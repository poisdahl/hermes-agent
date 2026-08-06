"""Core NeMo Relay adapter for Hermes tool execution."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from agent import relay_runtime

logger = logging.getLogger(__name__)


_MISSING = object()
_MAX_SEQUENTIAL_RELAY_WORKERS = 64
_MAX_ABANDONED_SEQUENTIAL_RELAY_WORKERS = 16
_sequential_worker_lock = threading.Lock()
_active_sequential_relay_workers = 0
_abandoned_sequential_relay_workers = 0


class SequentialRelayToolTimeout(BaseException):
    """Relay could not safely start a sequential Hermes tool before its bound.

    ``BaseException`` is intentional: plugin middleware commonly catches
    ``Exception`` and must not swallow this final no-dispatch gate.
    """

    def __init__(
        self,
        timeout_s: float,
        effect_disposition: str = "none",
        *,
        reason: str = "deadline",
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.effect_disposition = effect_disposition
        self.reason = reason
        if effect_disposition == "unknown":
            detail = (
                "ended after its Hermes tool effect began; the effect outcome "
                "is unknown"
            )
        elif reason == "interrupt":
            detail = "was interrupted before its Hermes tool effect began"
        else:
            detail = (
                f"exceeded its {self.timeout_s:g}s deadline before its Hermes "
                "tool effect began"
            )
        super().__init__(f"sequential Relay execution {detail}")


class SequentialRelayToolCapacityError(BaseException):
    """No bounded Relay-only worker slot is available for a tool call."""

    effect_disposition = "none"

    def __init__(self) -> None:
        super().__init__(
            "sequential Relay worker capacity is exhausted; Hermes tool was not run"
        )


class _SequentialRelayInvocation:
    """Run Relay off-thread while keeping the Hermes callback on its owner.

    Relay may invoke a synchronous callback from inside an async operation.  A
    daemon worker is safe for the Relay-only portion, but never for the actual
    Hermes tool: abandoning that thread could let external effects continue
    after a timeout result told the model that no tool was running.  This
    controller therefore hands the callback back to the original caller
    thread.  A deadline may abandon only Relay work before an effect claim, or
    Relay post-processing after the callback has completed.
    """

    def __init__(
        self,
        timeout_s: float,
        interrupted: Callable[[], bool] | None = None,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self._interrupted = interrupted
        self._condition = threading.Condition()
        self._deadline: float | None = None
        self._armed = False
        self._effect_started = False
        self._abandoned = False
        self._abandon_reason = "deadline"
        self._counted_abandoned = False
        self._worker_reserved = False
        self._request: Any = _MISSING
        self._request_consumed = False
        self._response: Any = _MISSING
        self._response_error: BaseException | None = None
        self._response_ready = False
        self._worker_done = False
        self._worker_result: Any = _MISSING
        self._worker_error: BaseException | None = None
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_task: asyncio.Task[Any] | None = None

    @property
    def effect_disposition(self) -> str:
        return "unknown" if self._effect_started else "none"

    @property
    def armed(self) -> bool:
        with self._condition:
            return self._armed

    def try_begin_effect(self, on_claim: Callable[[], Any] | None = None) -> None:
        """Atomically win the final deadline/interrupt race before dispatch."""
        with self._condition:
            if self._effect_started:
                raise RuntimeError("Hermes tool effect was claimed more than once")
            # Even an unmanaged Relay bypass must honor a late user interrupt.
            # Only the deadline portion depends on ``_arm()`` having run.
            self._raise_if_unavailable_locked()
            if on_claim is not None:
                on_claim()
            self._effect_started = True

    def _arm(self) -> None:
        with self._condition:
            if self._armed:
                raise RuntimeError("sequential Relay invocation reused")
            self._armed = True
            self._deadline = time.monotonic() + self.timeout_s

    def _interrupted_locked(self) -> bool:
        if self._interrupted is None:
            return False
        try:
            return bool(self._interrupted())
        except Exception:
            logger.debug("sequential Relay interrupt predicate failed", exc_info=True)
            return False

    def _timeout_locked(
        self, *, reason: str | None = None
    ) -> SequentialRelayToolTimeout:
        if reason is None:
            if self._abandoned:
                reason = self._abandon_reason
            else:
                reason = "interrupt" if self._interrupted_locked() else "deadline"
        return SequentialRelayToolTimeout(
            self.timeout_s,
            self.effect_disposition,
            reason=reason,
        )

    def _raise_if_unavailable_locked(self) -> None:
        if self._abandoned:
            raise self._timeout_locked()
        if self._interrupted_locked():
            raise self._timeout_locked(reason="interrupt")
        if self._deadline is not None and time.monotonic() >= self._deadline:
            raise self._timeout_locked(reason="deadline")

    def _remaining_locked(self) -> float:
        if self._deadline is None:
            return 0.0
        return max(0.0, self._deadline - time.monotonic())

    def _request_callback(self, args: Any) -> Any:
        """Relay-worker endpoint: publish one callback request and wait."""
        with self._condition:
            self._raise_if_unavailable_locked()
            if self._request is not _MISSING:
                raise RuntimeError("Hermes Relay tool callback invoked more than once")
            self._request = args
            self._condition.notify_all()
            self._condition.wait_for(lambda: self._response_ready or self._abandoned)
            if self._abandoned and not self._response_ready:
                raise self._timeout_locked()
            if self._response_error is not None:
                raise self._response_error
            return self._response

    def _publish_response(
        self,
        *,
        value: Any = _MISSING,
        error: BaseException | None = None,
    ) -> None:
        with self._condition:
            self._response = value
            self._response_error = error
            self._response_ready = True
            self._condition.notify_all()

    def _reserve_worker(self, value: Any) -> None:
        global _active_sequential_relay_workers
        rejected = False
        with _sequential_worker_lock:
            if (
                _active_sequential_relay_workers >= _MAX_SEQUENTIAL_RELAY_WORKERS
                or _abandoned_sequential_relay_workers
                >= _MAX_ABANDONED_SEQUENTIAL_RELAY_WORKERS
            ):
                rejected = True
            else:
                _active_sequential_relay_workers += 1
                self._worker_reserved = True
        if rejected:
            _close_awaitable(value)
            raise SequentialRelayToolCapacityError()

    def _release_worker(self) -> None:
        global _active_sequential_relay_workers
        global _abandoned_sequential_relay_workers
        with _sequential_worker_lock:
            if not self._worker_reserved:
                return
            self._worker_reserved = False
            _active_sequential_relay_workers = max(
                0, _active_sequential_relay_workers - 1
            )
            if self._counted_abandoned:
                _abandoned_sequential_relay_workers = max(
                    0, _abandoned_sequential_relay_workers - 1
                )
                self._counted_abandoned = False

    def _abandon_worker(self, *, reason: str = "deadline") -> None:
        global _abandoned_sequential_relay_workers
        loop: asyncio.AbstractEventLoop | None
        task: asyncio.Task[Any] | None
        with self._condition:
            first_abandon = not self._abandoned
            if first_abandon:
                self._abandon_reason = reason
            self._abandoned = True
            # Cancellation is edge-triggered. Repeated cleanup must not inject
            # a second CancelledError into a coroutine that caught the first
            # one to unwind or report a late callback attempt.
            loop = self._worker_loop if first_abandon else None
            task = self._worker_task if first_abandon else None
            self._condition.notify_all()
            if not self._worker_done and not self._counted_abandoned:
                with _sequential_worker_lock:
                    if self._worker_reserved:
                        _abandoned_sequential_relay_workers += 1
                        self._counted_abandoned = True
        if loop is not None and task is not None and not task.done():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass

    def _start_worker(self, value: Any) -> None:
        worker_context = contextvars.copy_context()

        def _worker() -> None:
            async def _drive() -> Any:
                task = asyncio.ensure_future(value)
                with self._condition:
                    self._worker_loop = asyncio.get_running_loop()
                    self._worker_task = task
                    abandon_now = self._abandoned
                    self._condition.notify_all()
                if abandon_now:
                    task.cancel()
                return await task

            try:
                result = worker_context.run(asyncio.run, _drive())
            except BaseException as exc:
                with self._condition:
                    self._worker_error = exc
            else:
                with self._condition:
                    self._worker_result = result
            finally:
                with self._condition:
                    # Release capacity before publishing completion so a
                    # caller awakened at saturation cannot be rejected by a
                    # worker that has already finished.
                    self._release_worker()
                    self._worker_done = True
                    self._condition.notify_all()

        thread = threading.Thread(
            target=_worker,
            name="hermes-sequential-relay",
            daemon=True,
        )
        try:
            thread.start()
        except BaseException as exc:
            if thread.ident is None:
                _close_awaitable(value)
                self._release_worker()
            else:
                self._abandon_worker(
                    reason="interrupt"
                    if isinstance(exc, KeyboardInterrupt)
                    else "deadline"
                )
            raise

    def run(self, value: Any, callback: Callable[[Any], Any]) -> Any:
        """Drive one Relay awaitable and service its callback on this thread."""
        if not inspect.isawaitable(value):
            return value
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            _close_awaitable(value)
            raise RuntimeError(
                "Synchronous Hermes Relay tool execution cannot run on an "
                "active event-loop thread"
            )

        self._arm()
        self._reserve_worker(value)

        try:
            self._start_worker(value)
            while True:
                request: Any = _MISSING
                timeout: SequentialRelayToolTimeout | None = None
                with self._condition:
                    while not self._worker_done and (
                        self._request is _MISSING or self._request_consumed
                    ):
                        if self._interrupted_locked():
                            timeout = self._timeout_locked(reason="interrupt")
                            break
                        remaining = self._remaining_locked()
                        if remaining <= 0:
                            timeout = self._timeout_locked(reason="deadline")
                            break
                        self._condition.wait(timeout=min(remaining, 0.05))
                    if timeout is None and self._request is not _MISSING:
                        if not self._request_consumed:
                            request = self._request
                            self._request_consumed = True
                        elif not self._worker_done:
                            # A response has been published; only Relay suffix
                            # work remains, still bounded by the original deadline.
                            if self._interrupted_locked():
                                timeout = self._timeout_locked(reason="interrupt")
                            elif self._remaining_locked() <= 0:
                                timeout = self._timeout_locked(reason="deadline")
                    if timeout is None and self._worker_done:
                        if self._worker_error is not None:
                            raise self._worker_error
                        return self._worker_result

                if timeout is not None:
                    self._abandon_worker(reason=timeout.reason)
                    raise timeout

                if request is not _MISSING:
                    try:
                        response = callback(request)
                    except BaseException as exc:
                        self._publish_response(error=exc)
                        self._abandon_worker(
                            reason=(
                                "interrupt"
                                if isinstance(exc, KeyboardInterrupt)
                                else "deadline"
                            )
                        )
                        raise
                    self._publish_response(value=response)
        except BaseException as exc:
            with self._condition:
                worker_unfinished = not self._worker_done
            if worker_unfinished:
                self._abandon_worker(
                    reason="interrupt"
                    if isinstance(exc, KeyboardInterrupt)
                    else "deadline"
                )
            raise


def execute(
    tool_name: str,
    args: dict[str, Any],
    callback: Callable[[dict[str, Any]], Any],
    *,
    session_id: str,
    metadata: dict[str, Any] | None = None,
    sequential_execution: _SequentialRelayInvocation | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run one tool call through Relay and return its final arguments."""
    runtime, session, parent = relay_runtime.resolve_execution_context(session_id)
    if runtime is None or session is None or not runtime.managed_execution_enabled():
        return callback(args), args

    observed_args = args
    raw_result: dict[str, Any] = {}
    callback_error: BaseException | None = None
    callback_context = contextvars.copy_context()

    def invoke(next_args: Any) -> Any:
        nonlocal callback_error, observed_args
        observed_args = next_args if isinstance(next_args, dict) else args
        try:
            result = callback_context.copy().run(callback, observed_args)
        except BaseException as exc:
            callback_error = exc
            raise
        raw_result["value"] = result
        raw_result["json"] = _jsonable(result)
        return raw_result["json"]

    relay_invoke = (
        sequential_execution._request_callback
        if sequential_execution is not None
        else invoke
    )
    relay_value = runtime.run_in_session_async(
        session,
        runtime.relay.tools.execute,
        tool_name,
        _jsonable(args),
        relay_invoke,
        handle=parent,
        metadata=_jsonable(metadata or {}),
    )

    try:
        managed = (
            sequential_execution.run(relay_value, invoke)
            if sequential_execution is not None
            else _run_awaitable(relay_value)
        )
    except BaseException as exc:
        if callback_error is not None and (
            exc is callback_error
            or relay_runtime._is_relay_wrapped_callback_error(exc, callback_error)
        ):
            raise callback_error
        if isinstance(exc, SequentialRelayToolTimeout) and "value" in raw_result:
            logger.warning(
                "NeMo Relay tool post-processing exceeded the sequential "
                "deadline after dispatch success; returning the Hermes tool result",
                exc_info=True,
            )
            return raw_result["value"], observed_args
        worker_only_error = (
            sequential_execution is not None
            and exc is sequential_execution._worker_error
        )
        if (
            callback_error is None
            and "value" in raw_result
            and (isinstance(exc, Exception) or worker_only_error)
        ):
            logger.warning(
                "NeMo Relay tool post-processing failed after dispatch success; "
                "returning the Hermes tool result",
                exc_info=True,
            )
            return raw_result["value"], observed_args
        raise

    if "value" in raw_result and _json_equal(managed, raw_result["json"]):
        return raw_result["value"], observed_args
    if isinstance(managed, str):
        return managed, observed_args
    return json.dumps(_jsonable(managed), ensure_ascii=False), observed_args


def _close_awaitable(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except Exception:
            pass
    try:
        return _jsonable(vars(value))
    except (TypeError, AttributeError):
        return str(value)


def _json_equal(left: Any, right: Any) -> bool:
    try:
        return json.dumps(
            _jsonable(left), sort_keys=True, separators=(",", ":")
        ) == json.dumps(_jsonable(right), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return left == right


def _run_awaitable(value: Any) -> Any:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    raise RuntimeError(
        "Synchronous Hermes Relay tool execution cannot run on an active event-loop thread"
    )
