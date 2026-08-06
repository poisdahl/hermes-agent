"""Behavior contracts for bounded sequential Relay tool execution.

These tests deliberately drive :func:`agent.relay_tools.execute` through a
small fake Relay runtime.  They exercise the thread hand-off and effect gate,
not implementation details such as source text or private worker counters.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from agent import relay_tools


class _FakeSession:
    handle = object()


class _FakeRuntime:
    """Minimal managed runtime that returns a caller-supplied awaitable."""

    def __init__(
        self,
        operation: Callable[[Callable[[Any], Any], dict[str, Any]], Any],
        *,
        managed: bool = True,
    ) -> None:
        self._operation = operation
        self._managed = managed
        self.relay = SimpleNamespace(tools=SimpleNamespace(execute=object()))

    def managed_execution_enabled(self) -> bool:
        return self._managed

    def run_in_session_async(
        self,
        session,
        relay_execute,
        tool_name,
        args,
        invoke,
        **kwargs,
    ):
        del session, relay_execute, tool_name, kwargs
        return self._operation(invoke, args)


def _patch_runtime(monkeypatch, runtime: _FakeRuntime) -> None:
    monkeypatch.setattr(
        relay_tools.relay_runtime,
        "resolve_execution_context",
        lambda _session_id: (runtime, _FakeSession(), None),
    )


def _controller(
    timeout_s: float,
    *,
    interrupted: Callable[[], bool] | None = None,
    human_wait_seconds: Callable[[], float] | None = None,
):
    return relay_tools._SequentialRelayInvocation(
        timeout_s,
        interrupted=interrupted,
        human_wait_seconds=human_wait_seconds,
    )


def _begin_effect(controller) -> None:
    controller.try_begin_effect()


def test_unmanaged_execution_bypasses_relay_on_the_caller_thread(monkeypatch):
    caller_thread = threading.get_ident()
    callback_threads: list[int] = []
    args = {"command": "pwd"}

    def _must_not_run(_invoke, _args):  # pragma: no cover - failure path
        raise AssertionError("an unmanaged runtime must be bypassed")

    _patch_runtime(monkeypatch, _FakeRuntime(_must_not_run, managed=False))

    result, observed_args = relay_tools.execute(
        "terminal",
        args,
        lambda value: callback_threads.append(threading.get_ident()) or value,
        session_id="session-1",
        sequential_execution=_controller(0.01),
    )

    assert result is args
    assert observed_args is args
    assert callback_threads == [caller_thread]


def test_unmanaged_bypass_preserves_legacy_dispatch_when_interrupt_is_set(
    monkeypatch,
):
    interrupt = threading.Event()
    effects: list[dict[str, Any]] = []
    execution = _controller(1.0, interrupted=interrupt.is_set)

    def _must_not_run(_invoke, _args):  # pragma: no cover - failure path
        raise AssertionError("an unmanaged runtime must be bypassed")

    def _callback(args):
        interrupt.set()
        execution.try_begin_effect(lambda: effects.append(args))
        return "done"

    _patch_runtime(monkeypatch, _FakeRuntime(_must_not_run, managed=False))
    args = {"path": "result.txt"}
    result, observed_args = relay_tools.execute(
        "write_file",
        args,
        _callback,
        session_id="session-1",
        sequential_execution=execution,
    )

    assert result == "done"
    assert observed_args is args
    assert effects == [args]


def test_managed_callback_runs_on_the_caller_thread(monkeypatch):
    caller_thread = threading.get_ident()
    callback_threads: list[int] = []
    execution = _controller(1.0)

    async def _invoke_once(invoke, args):
        return invoke(args)

    def _callback(args):
        callback_threads.append(threading.get_ident())
        _begin_effect(execution)
        return {"ok": args["value"]}

    _patch_runtime(monkeypatch, _FakeRuntime(_invoke_once))
    result, observed_args = relay_tools.execute(
        "tool",
        {"value": 7},
        _callback,
        session_id="session-1",
        sequential_execution=execution,
    )

    assert callback_threads == [caller_thread]
    assert result == {"ok": 7}
    assert observed_args == {"value": 7}


def test_callback_exception_is_raised_only_on_owner_thread(monkeypatch):
    original = RuntimeError("owner callback failed")
    execution = _controller(1.0)

    async def _invoke_once(invoke, args):
        return invoke(args)

    def _callback(_args):
        _begin_effect(execution)
        raise original

    _patch_runtime(monkeypatch, _FakeRuntime(_invoke_once))
    with pytest.raises(RuntimeError) as caught:
        relay_tools.execute(
            "tool",
            {},
            _callback,
            session_id="session-1",
            sequential_execution=execution,
        )

    with execution._condition:
        assert execution._condition.wait_for(
            lambda: execution._worker_done,
            timeout=2,
        )
        worker_error = execution._worker_error

    assert caught.value is original
    assert isinstance(worker_error, relay_tools._SequentialRelayCallbackError)
    assert worker_error is not original


def test_managed_handoff_preserves_caller_contextvars(monkeypatch):
    marker = contextvars.ContextVar("relay_test_marker", default="missing")
    token = marker.set("caller-value")
    execution = _controller(1.0)
    observed: list[str] = []

    async def _invoke_once(invoke, args):
        return invoke(args)

    def _callback(_args):
        _begin_effect(execution)
        observed.append(marker.get())
        return "done"

    _patch_runtime(monkeypatch, _FakeRuntime(_invoke_once))
    try:
        result, _ = relay_tools.execute(
            "tool",
            {},
            _callback,
            session_id="session-1",
            sequential_execution=execution,
        )
    finally:
        marker.reset(token)

    assert result == "done"
    assert observed == ["caller-value"]


def test_managed_execution_without_controller_keeps_original_path(monkeypatch):
    caller_thread = threading.get_ident()
    callback_threads: list[int] = []

    async def _invoke_once(invoke, args):
        return invoke(args)

    _patch_runtime(monkeypatch, _FakeRuntime(_invoke_once))
    result, observed_args = relay_tools.execute(
        "tool",
        {"value": 9},
        lambda args: callback_threads.append(threading.get_ident()) or args["value"],
        session_id="session-1",
        sequential_execution=None,
    )

    assert result == 9
    assert observed_args == {"value": 9}
    assert callback_threads == [caller_thread]


def test_timeout_before_invoke_has_no_late_effect(monkeypatch):
    callback_calls: list[dict[str, Any]] = []

    async def _never_invokes(_invoke, _args):
        await asyncio.Event().wait()

    _patch_runtime(monkeypatch, _FakeRuntime(_never_invokes))
    with pytest.raises(relay_tools.SequentialRelayToolTimeout) as caught:
        relay_tools.execute(
            "tool",
            {"value": 1},
            lambda args: callback_calls.append(args),
            session_id="session-1",
            sequential_execution=_controller(0.05),
        )

    assert caught.value.timeout_s == pytest.approx(0.05)
    assert caught.value.effect_disposition == "none"
    assert callback_calls == []


def test_cancel_swallowing_worker_cannot_invoke_after_timeout(monkeypatch):
    cancelled = threading.Event()
    release = threading.Event()
    late_attempt_finished = threading.Event()
    callback_calls: list[dict[str, Any]] = []
    late_errors: list[BaseException] = []

    async def _swallow_cancel_then_invoke(invoke, args):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            while not release.is_set():
                await asyncio.sleep(0.002)
        try:
            invoke(args)
        except BaseException as exc:
            late_errors.append(exc)
        finally:
            late_attempt_finished.set()

    _patch_runtime(monkeypatch, _FakeRuntime(_swallow_cancel_then_invoke))
    try:
        with pytest.raises(relay_tools.SequentialRelayToolTimeout) as caught:
            relay_tools.execute(
                "tool",
                {"value": 1},
                lambda args: callback_calls.append(args),
                session_id="session-1",
                sequential_execution=_controller(0.05),
            )
        assert caught.value.effect_disposition == "none"
        assert cancelled.wait(timeout=2)
    finally:
        release.set()

    assert late_attempt_finished.wait(timeout=2)
    assert callback_calls == []
    assert len(late_errors) == 1
    assert isinstance(late_errors[0], relay_tools.SequentialRelayToolTimeout)
    assert late_errors[0].effect_disposition == "none"


def test_expired_pending_request_is_rejected_before_owner_callback(monkeypatch):
    execution = _controller(1.0)
    callback_calls: list[dict[str, Any]] = []
    clock_values = iter((0.0, 2.0))

    async def _unused_worker():  # pragma: no cover - closed by fake starter
        return None

    def _publish_pending_request(value) -> None:
        value.close()
        with execution._condition:
            execution._request = {"late": True}
            execution._condition.notify_all()

    monkeypatch.setattr(relay_tools.time, "monotonic", lambda: next(clock_values))
    monkeypatch.setattr(execution, "_reserve_worker", lambda _value: None)
    monkeypatch.setattr(execution, "_start_worker", _publish_pending_request)

    with pytest.raises(relay_tools.SequentialRelayToolTimeout) as caught:
        execution.run(
            _unused_worker(),
            lambda args: callback_calls.append(args),
        )

    assert caught.value.reason == "deadline"
    assert caught.value.effect_disposition == "none"
    assert callback_calls == []


def test_effect_claim_is_atomic_with_deadline_expiry():
    """Expiry during the owner hand-off cannot turn a claimed effect into NONE."""

    execution = _controller(0.15)
    execution._arm()
    owner_entered = threading.Event()
    release_owner = threading.Event()
    result: list[None] = []

    def _on_claim() -> None:
        owner_entered.set()
        assert release_owner.wait(timeout=2)

    worker = threading.Thread(
        target=lambda: result.append(execution.try_begin_effect(_on_claim)),
        daemon=True,
    )
    worker.start()
    assert owner_entered.wait(timeout=2)
    # Let the original deadline pass while on_claim is still inside the same
    # atomic ownership transition, then permit the transition to finish.
    assert not release_owner.wait(timeout=0.25)
    release_owner.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert result == [None]


def test_human_wait_delta_extends_final_effect_gate(monkeypatch):
    now = 10.0
    human_wait = 7.0
    execution = _controller(1.0, human_wait_seconds=lambda: human_wait)
    monkeypatch.setattr(relay_tools.time, "monotonic", lambda: now)
    execution._arm()

    now = 12.0
    human_wait = 8.5
    execution.try_begin_effect()

    assert execution.effect_disposition == "unknown"


def test_prior_human_wait_does_not_extend_final_effect_gate(monkeypatch):
    now = 10.0
    human_wait = 7.0
    execution = _controller(1.0, human_wait_seconds=lambda: human_wait)
    monkeypatch.setattr(relay_tools.time, "monotonic", lambda: now)
    execution._arm()

    now = 12.0
    with pytest.raises(relay_tools.SequentialRelayToolTimeout) as caught:
        execution.try_begin_effect()

    assert caught.value.reason == "deadline"
    assert caught.value.effect_disposition == "none"


def test_interrupt_wins_over_human_wait_extension(monkeypatch):
    now = 10.0
    human_wait = 0.0
    interrupted = False
    execution = _controller(
        1.0,
        interrupted=lambda: interrupted,
        human_wait_seconds=lambda: human_wait,
    )
    monkeypatch.setattr(relay_tools.time, "monotonic", lambda: now)
    execution._arm()

    now = 12.0
    human_wait = 2.0
    interrupted = True
    with pytest.raises(relay_tools.SequentialRelayToolTimeout) as caught:
        execution.try_begin_effect()

    assert caught.value.reason == "interrupt"
    assert caught.value.effect_disposition == "none"


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
def test_invalid_human_wait_value_does_not_extend_deadline(monkeypatch, invalid):
    now = 10.0
    human_wait = 0.0
    execution = _controller(1.0, human_wait_seconds=lambda: human_wait)
    monkeypatch.setattr(relay_tools.time, "monotonic", lambda: now)
    execution._arm()

    now = 12.0
    human_wait = invalid
    with pytest.raises(relay_tools.SequentialRelayToolTimeout) as caught:
        execution.try_begin_effect()

    assert caught.value.reason == "deadline"
    assert caught.value.effect_disposition == "none"


def test_human_wait_reader_failure_does_not_extend_deadline(monkeypatch):
    now = 10.0
    fail = False

    def _human_wait_seconds():
        if fail:
            raise RuntimeError("counter unavailable")
        return 0.0

    execution = _controller(1.0, human_wait_seconds=_human_wait_seconds)
    monkeypatch.setattr(relay_tools.time, "monotonic", lambda: now)
    execution._arm()

    now = 12.0
    fail = True
    with pytest.raises(relay_tools.SequentialRelayToolTimeout) as caught:
        execution.try_begin_effect()

    assert caught.value.reason == "deadline"
    assert caught.value.effect_disposition == "none"


def test_final_effect_gate_rejects_after_slow_preflight(monkeypatch):
    preflight_started = threading.Event()
    release_preflight = threading.Event()
    effects: list[str] = []
    execution = _controller(0.05)

    async def _invoke_once(invoke, args):
        return invoke(args)

    def _callback(_args):
        preflight_started.set()
        assert release_preflight.wait(timeout=2)
        execution.try_begin_effect(lambda: effects.append("claimed"))
        effects.append("ran")
        return "unexpected"

    def _release_after_deadline() -> None:
        assert preflight_started.wait(timeout=2)
        assert not release_preflight.wait(timeout=0.12)
        release_preflight.set()

    helper = threading.Thread(target=_release_after_deadline, daemon=True)
    helper.start()
    _patch_runtime(monkeypatch, _FakeRuntime(_invoke_once))

    with pytest.raises(relay_tools.SequentialRelayToolTimeout) as caught:
        relay_tools.execute(
            "tool",
            {},
            _callback,
            session_id="session-1",
            sequential_execution=execution,
        )
    helper.join(timeout=2)

    assert caught.value.effect_disposition == "none"
    assert effects == []


def test_effect_started_callback_is_never_hard_abandoned(monkeypatch):
    execution = _controller(0.05)
    effect_started = threading.Event()
    release_effect = threading.Event()

    async def _invoke_once(invoke, args):
        return invoke(args)

    def _callback(_args):
        _begin_effect(execution)
        effect_started.set()
        assert release_effect.wait(timeout=2)
        return {"ok": True}

    def _release_well_after_deadline() -> None:
        assert effect_started.wait(timeout=2)
        assert not release_effect.wait(timeout=0.12)
        release_effect.set()

    helper = threading.Thread(target=_release_well_after_deadline, daemon=True)
    helper.start()
    _patch_runtime(monkeypatch, _FakeRuntime(_invoke_once))

    result, _ = relay_tools.execute(
        "write_file",
        {"path": "result.txt"},
        _callback,
        session_id="session-1",
        sequential_execution=execution,
    )
    helper.join(timeout=2)

    assert result == {"ok": True}
    assert not helper.is_alive()


def test_completed_callback_wins_over_relay_suffix_timeout(monkeypatch):
    execution = _controller(0.05)

    async def _invoke_then_wedge(invoke, args):
        invoke(args)
        await asyncio.Event().wait()

    def _callback(args):
        _begin_effect(execution)
        return {"ok": True, "args": args}

    _patch_runtime(monkeypatch, _FakeRuntime(_invoke_then_wedge))
    result, observed_args = relay_tools.execute(
        "write_file",
        {"path": "result.txt"},
        _callback,
        session_id="session-1",
        sequential_execution=execution,
    )

    assert result == {"ok": True, "args": {"path": "result.txt"}}
    assert observed_args == {"path": "result.txt"}


def test_completed_callback_wins_over_relay_suffix_cancellation(monkeypatch):
    execution = _controller(1.0)
    callback_calls: list[dict[str, Any]] = []

    async def _invoke_then_cancel(invoke, args):
        invoke(args)
        raise asyncio.CancelledError

    def _callback(args):
        _begin_effect(execution)
        callback_calls.append(args)
        return {"ok": True, "args": args}

    _patch_runtime(monkeypatch, _FakeRuntime(_invoke_then_cancel))
    result, observed_args = relay_tools.execute(
        "write_file",
        {"path": "result.txt"},
        _callback,
        session_id="session-1",
        sequential_execution=execution,
    )

    assert result == {"ok": True, "args": {"path": "result.txt"}}
    assert observed_args == {"path": "result.txt"}
    assert callback_calls == [{"path": "result.txt"}]


def test_owner_baseexception_abandons_worker_and_blocks_late_effect(monkeypatch):
    worker_started = threading.Event()
    worker_cancelled = threading.Event()
    permit_late_attempt = threading.Event()
    late_attempt_finished = threading.Event()
    worker_released = threading.Event()
    callback_calls: list[dict[str, Any]] = []
    late_errors: list[BaseException] = []

    def _owner_interrupt() -> bool:
        if worker_started.is_set():
            raise KeyboardInterrupt
        return False

    execution = _controller(1.0, interrupted=_owner_interrupt)
    original_release_worker = execution._release_worker

    def _observe_worker_release() -> None:
        original_release_worker()
        worker_released.set()

    monkeypatch.setattr(execution, "_release_worker", _observe_worker_release)
    with relay_tools._sequential_worker_lock:
        baseline_active = relay_tools._active_sequential_relay_workers
        baseline_abandoned = relay_tools._abandoned_sequential_relay_workers

    async def _relay_worker():
        worker_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            worker_cancelled.set()
            while not permit_late_attempt.is_set():
                await asyncio.sleep(0.002)
        try:
            execution._request_callback({"late": True})
        except BaseException as exc:
            late_errors.append(exc)
        finally:
            late_attempt_finished.set()

    cleanup_observed = False
    try:
        with pytest.raises(KeyboardInterrupt):
            execution.run(
                _relay_worker(),
                lambda args: callback_calls.append(args),
            )
        cleanup_observed = worker_cancelled.wait(timeout=0.5)
    finally:
        # Keep a failing pre-fix run leak-free. Do not cancel a second time
        # after observing production cleanup: the worker intentionally
        # swallows the first cancellation so it can prove a late callback is
        # rejected, and a second Task.cancel() would abort that probe itself.
        if not cleanup_observed:
            execution._abandon_worker()
        permit_late_attempt.set()

    assert late_attempt_finished.wait(timeout=2)
    assert worker_released.wait(timeout=2)
    with relay_tools._sequential_worker_lock:
        final_active = relay_tools._active_sequential_relay_workers
        final_abandoned = relay_tools._abandoned_sequential_relay_workers

    assert cleanup_observed, "owner BaseException did not cancel the Relay worker"
    assert callback_calls == []
    assert len(late_errors) == 1
    assert isinstance(late_errors[0], relay_tools.SequentialRelayToolTimeout)
    assert late_errors[0].effect_disposition == "none"
    assert final_active == baseline_active
    assert final_abandoned == baseline_abandoned


def test_sequential_worker_capacity_fails_closed_without_dispatch(
    monkeypatch,
    caplog,
):
    callback_calls: list[dict[str, Any]] = []

    async def _invoke_once(invoke, args):  # pragma: no cover - capacity wins
        return invoke(args)

    _patch_runtime(monkeypatch, _FakeRuntime(_invoke_once))
    monkeypatch.setattr(relay_tools, "_MAX_SEQUENTIAL_RELAY_WORKERS", 0)

    with caplog.at_level("WARNING", logger="agent.relay_tools"):
        with pytest.raises(relay_tools.SequentialRelayToolCapacityError) as caught:
            relay_tools.execute(
                "tool",
                {},
                lambda args: callback_calls.append(args),
                session_id="session-1",
                sequential_execution=_controller(1.0),
            )

    assert caught.value.effect_disposition == "none"
    assert callback_calls == []
    assert "active-worker limit" in caplog.text
    assert "active=" in caplog.text
    assert "/0" in caplog.text
    assert "abandoned=" in caplog.text
    assert "/16" in caplog.text


def test_unbounded_awaitable_keeps_original_value_and_loop_contract():
    async def _quick():
        return "done"

    assert relay_tools._run_awaitable("plain") == "plain"
    assert relay_tools._run_awaitable(_quick()) == "done"

    async def _driver() -> None:
        payload = _quick()
        try:
            with pytest.raises(RuntimeError, match="event-loop thread"):
                relay_tools._run_awaitable(payload)
        finally:
            payload.close()

    asyncio.run(_driver())
