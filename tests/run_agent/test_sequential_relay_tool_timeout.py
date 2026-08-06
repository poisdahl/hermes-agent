"""Integration contracts for the sequential Relay tool timeout.

The low-level Relay hand-off is covered in ``tests/agent``.  This module keeps
the configuration-to-agent wiring and the user-visible sequential tool loop
under test, including truthful effect metadata after a timeout.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import uuid
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import agent_init, relay_tools
from run_agent import AIAgent


_REAL_CONFIG = object()


class _ManagedRelayRuntime:
    """Invoke Relay's callback once through the real sequential controller."""

    def __init__(self) -> None:
        self.relay = SimpleNamespace(tools=SimpleNamespace(execute=object()))

    def managed_execution_enabled(self) -> bool:
        return True

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

        async def _invoke_once():
            return invoke(args)

        return _invoke_once()


class _UnmanagedRelayRuntime(_ManagedRelayRuntime):
    def managed_execution_enabled(self) -> bool:
        return False


def _install_managed_relay(monkeypatch) -> None:
    runtime = _ManagedRelayRuntime()
    session = SimpleNamespace(handle=object())
    monkeypatch.setattr(
        relay_tools.relay_runtime,
        "resolve_execution_context",
        lambda _session_id: (runtime, session, None),
    )


def _install_unmanaged_relay(monkeypatch) -> None:
    runtime = _UnmanagedRelayRuntime()
    session = SimpleNamespace(handle=object())
    monkeypatch.setattr(
        relay_tools.relay_runtime,
        "resolve_execution_context",
        lambda _session_id: (runtime, session, None),
    )


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _tool_call(name: str, call_id: str, args: dict | None = None):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args or {}),
        ),
    )


def _assistant_message(*calls):
    return SimpleNamespace(content="", tool_calls=list(calls))


def _build_agent(
    *tool_names: str,
    config: dict | object = _REAL_CONFIG,
) -> AIAgent:
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "run_agent.get_tool_definitions", return_value=_tool_defs(*tool_names)
            )
        )
        stack.enter_context(
            patch("run_agent.check_toolset_requirements", return_value={})
        )
        stack.enter_context(patch("run_agent.OpenAI"))
        if config is not _REAL_CONFIG:
            stack.enter_context(
                patch("hermes_cli.config.load_config", return_value=config)
            )
            stack.enter_context(
                patch("hermes_cli.config.load_config_readonly", return_value=config)
            )
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (660, 660.0),
        (0, None),
        ("0", None),
        ("12.5", 12.5),
        (0.125, 0.125),
    ],
)
def test_sequential_timeout_normalizer_accepts_supported_values(raw, expected):
    assert agent_init._normalize_sequential_tool_execution_timeout(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        True,
        False,
        -1,
        "not-a-number",
        math.nan,
        math.inf,
        -math.inf,
        None,
        {},
        [],
    ],
)
def test_sequential_timeout_normalizer_rejects_unsafe_values(raw):
    with pytest.raises(ValueError):
        agent_init._normalize_sequential_tool_execution_timeout(raw)


def test_default_sequential_timeout_is_660_seconds():
    agent = _build_agent("web_search", config={})

    assert agent_init._DEFAULT_SEQUENTIAL_TOOL_EXECUTION_TIMEOUT_S == 660.0
    assert agent._sequential_tool_execution_timeout_s == 660.0


@pytest.mark.parametrize(
    ("yaml_value", "expected"),
    [("0.125", 0.125), ("0", None)],
)
def test_real_hermes_home_config_reaches_agent(
    tmp_path,
    monkeypatch,
    yaml_value,
    expected,
):
    from hermes_cli import config as config_module

    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text(
        f"agent:\n  sequential_tool_execution_timeout: {yaml_value}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile))
    config_module._LOAD_CONFIG_CACHE.clear()

    agent = _build_agent("web_search")

    assert agent._sequential_tool_execution_timeout_s == expected


def test_invalid_real_config_warns_and_falls_back_to_660(
    tmp_path,
    monkeypatch,
    caplog,
):
    from hermes_cli import config as config_module

    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text(
        "agent:\n  sequential_tool_execution_timeout: invalid\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(profile))
    config_module._LOAD_CONFIG_CACHE.clear()

    with caplog.at_level("WARNING"):
        agent = _build_agent("web_search")

    assert agent._sequential_tool_execution_timeout_s == 660.0
    assert "agent.sequential_tool_execution_timeout" in caplog.text


def test_sequential_path_propagates_one_configured_controller():
    agent = _build_agent(
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 0.125}},
    )
    messages: list[dict] = []
    seen_controllers: list[object] = []

    def _relay_execute(_name, args, callback, **kwargs):
        seen_controllers.append(kwargs.get("sequential_execution"))
        return callback(dict(args)), dict(args)

    with (
        patch("agent.relay_tools.execute", side_effect=_relay_execute),
        patch("run_agent.handle_function_call", return_value='{"ok": true}'),
    ):
        agent._execute_tool_calls_sequential(
            _assistant_message(
                _tool_call("web_search", "call-sequential", {"query": "hermes"})
            ),
            messages,
            "task-1",
        )

    assert len(seen_controllers) == 1
    controller = seen_controllers[0]
    assert isinstance(controller, relay_tools._SequentialRelayInvocation)
    assert controller.timeout_s == pytest.approx(0.125)
    assert messages[0]["tool_call_id"] == "call-sequential"
    assert "ok" in messages[0]["content"]


def test_concurrent_path_does_not_create_sequential_controllers():
    agent = _build_agent(
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 0.125}},
    )
    messages: list[dict] = []
    seen_controllers: list[object] = []
    seen_lock = threading.Lock()

    def _relay_execute(_name, args, callback, **kwargs):
        with seen_lock:
            seen_controllers.append(kwargs.get("sequential_execution"))
        return callback(dict(args)), dict(args)

    calls = [
        _tool_call("web_search", f"call-{index}", {"query": str(index)})
        for index in range(2)
    ]
    with (
        patch("agent.relay_tools.execute", side_effect=_relay_execute),
        patch("run_agent.handle_function_call", return_value='{"ok": true}'),
    ):
        agent._execute_tool_calls_concurrent(
            _assistant_message(*calls),
            messages,
            "task-1",
        )

    assert seen_controllers == [None, None]
    assert {message["tool_call_id"] for message in messages} == {
        "call-0",
        "call-1",
    }


def test_disabled_bound_passes_no_controller_and_propagates_original_exception():
    from agent import tool_executor

    agent = _build_agent(
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 0}},
    )
    original_error = RuntimeError("unbounded middleware failure")
    seen_controllers: list[object] = []

    def _relay_execute(_name, args, callback, **kwargs):
        seen_controllers.append(kwargs.get("sequential_execution"))
        return callback(dict(args)), dict(args)

    def _raise_original(_args):
        raise original_error

    with (
        patch("agent.relay_tools.execute", side_effect=_relay_execute),
        patch("hermes_cli.plugins.resolve_pre_tool_block", return_value=None),
        pytest.raises(RuntimeError) as caught,
    ):
        tool_executor._run_agent_tool_execution_middleware(
            agent,
            function_name="web_search",
            function_args={"query": "x"},
            effective_task_id="task-1",
            tool_call_id="call-unbounded",
            execute=_raise_original,
            sequential_timeout_s=agent._sequential_tool_execution_timeout_s,
        )

    assert agent._sequential_tool_execution_timeout_s is None
    assert seen_controllers == [None]
    assert caught.value is original_error


def test_concurrent_exception_keeps_unbounded_contract_without_controller():
    agent = _build_agent(
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 0.125}},
    )
    messages: list[dict] = []
    seen_controllers: list[object] = []
    seen_lock = threading.Lock()

    def _relay_execute(_name, args, callback, **kwargs):
        with seen_lock:
            seen_controllers.append(kwargs.get("sequential_execution"))
        return callback(dict(args)), dict(args)

    agent._invoke_tool = MagicMock(
        side_effect=RuntimeError("legacy concurrent failure")
    )
    calls = _assistant_message(
        _tool_call("web_search", "call-concurrent-error-1", {"query": "one"}),
        _tool_call("web_search", "call-concurrent-error-2", {"query": "two"}),
    )
    with patch("agent.relay_tools.execute", side_effect=_relay_execute):
        agent._execute_tool_calls_concurrent(calls, messages, "task-1")

    assert seen_controllers == [None, None]
    assert agent._invoke_tool.call_count == 2
    assert len(messages) == 2
    for message in messages:
        assert "legacy concurrent failure" in message["content"]
        assert "sequential_relay" not in message["content"]
        assert "effect_disposition" not in message


def test_final_effect_gate_runs_after_slow_tool_preflight():
    from agent import tool_executor

    agent = _build_agent("write_file", config={})
    preflight_started = threading.Event()
    release_preflight = threading.Event()
    effects: list[dict] = []

    def _slow_preflight(*_args, **_kwargs):
        preflight_started.set()
        assert release_preflight.wait(timeout=2)

    def _release_after_deadline() -> None:
        assert preflight_started.wait(timeout=2)
        assert not release_preflight.wait(timeout=0.12)
        release_preflight.set()

    def _relay_execute(_name, args, callback, **kwargs):
        execution = kwargs["sequential_execution"]

        async def _request_callback():
            return execution._request_callback(dict(args))

        return execution.run(_request_callback(), callback), dict(args)

    helper = threading.Thread(target=_release_after_deadline, daemon=True)
    helper.start()
    with (
        patch("agent.relay_tools.execute", side_effect=_relay_execute),
        patch.object(
            tool_executor,
            "_begin_tool_execution",
            side_effect=_slow_preflight,
        ),
        patch("hermes_cli.plugins.resolve_pre_tool_block", return_value=None),
    ):
        outcome = tool_executor._run_agent_tool_execution_middleware(
            agent,
            function_name="write_file",
            function_args={"path": "result.txt", "content": "value"},
            effective_task_id="task-1",
            tool_call_id="call-preflight",
            execute=lambda args: effects.append(args) or "unexpected",
            sequential_timeout_s=0.05,
        )
    helper.join(timeout=2)

    assert not helper.is_alive()
    assert outcome.timed_out is True
    assert outcome.dispatched is False
    assert outcome.effect_disposition == "none"
    assert effects == []


def test_claimed_effect_exception_is_unknown_and_next_call_runs(monkeypatch):
    _install_managed_relay(monkeypatch)
    agent = _build_agent(
        "terminal",
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 1}},
    )
    messages: list[dict] = []

    def _dispatch(name, *_args, **_kwargs):
        if name == "terminal":
            raise RuntimeError("dispatch failed after effect claim")
        return '{"ok": true}'

    calls = _assistant_message(
        _tool_call("terminal", "call-effect-error", {"command": "do-work"}),
        _tool_call("web_search", "call-after-error", {"query": "next"}),
    )
    with (
        patch("run_agent.handle_function_call", side_effect=_dispatch) as dispatch,
        patch("agent.tool_executor._emit_terminal_post_tool_call") as terminal_row,
    ):
        agent._execute_tool_calls_sequential(calls, messages, "task-1")

    by_id = {message["tool_call_id"]: message for message in messages}
    assert by_id["call-effect-error"]["effect_disposition"] == "unknown"
    assert by_id["call-effect-error"]["content"] == (
        "Error executing tool 'terminal': dispatch failed after effect claim"
    )
    assert "ok" in by_id["call-after-error"]["content"]
    assert dispatch.call_count == 2
    assert terminal_row.call_count == 1
    assert terminal_row.call_args.kwargs["tool_call_id"] == "call-effect-error"


def test_managed_pre_dispatch_exception_keeps_legacy_error_and_one_hook(
    monkeypatch,
):
    runtime = _ManagedRelayRuntime()
    original = RuntimeError("Relay failed before callback")

    def _raise_before_callback(*_args, **_kwargs):
        async def _raise():
            raise original

        return _raise()

    runtime.run_in_session_async = _raise_before_callback
    session = SimpleNamespace(handle=object())
    monkeypatch.setattr(
        relay_tools.relay_runtime,
        "resolve_execution_context",
        lambda _session_id: (runtime, session, None),
    )
    agent = _build_agent(
        "terminal",
        config={"agent": {"sequential_tool_execution_timeout": 1}},
    )
    messages: list[dict] = []

    with (
        patch("run_agent.handle_function_call") as dispatch,
        patch("agent.tool_executor._emit_terminal_post_tool_call") as terminal_row,
    ):
        agent._execute_tool_calls_sequential(
            _assistant_message(
                _tool_call(
                    "terminal",
                    "call-pre-dispatch-error",
                    {"command": "pwd"},
                )
            ),
            messages,
            "task-1",
        )

    assert messages[0]["content"] == (
        "Error executing tool 'terminal': Relay failed before callback"
    )
    assert messages[0]["effect_disposition"] == "none"
    dispatch.assert_not_called()
    assert terminal_row.call_count == 1
    assert terminal_row.call_args.kwargs["tool_call_id"] == "call-pre-dispatch-error"


def test_claimed_inline_runtime_exception_preserves_legacy_propagation(
    monkeypatch,
):
    _install_managed_relay(monkeypatch)
    agent = _build_agent(
        "todo",
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 1}},
    )
    messages: list[dict] = []
    calls = _assistant_message(
        _tool_call("todo", "call-inline-error", {"todos": []}),
        _tool_call("web_search", "call-after-inline-error", {"query": "next"}),
    )

    original = RuntimeError("inline dispatch failed after effect claim")
    with (
        patch(
            "tools.todo_tool.todo_tool",
            side_effect=original,
        ) as inline_dispatch,
        patch(
            "run_agent.handle_function_call",
            return_value='{"ok": true}',
        ) as next_call,
        pytest.raises(RuntimeError) as caught,
    ):
        agent._execute_tool_calls_sequential(calls, messages, "task-1")

    assert caught.value is original
    assert caught.value.effect_disposition == "unknown"
    assert messages == []
    inline_dispatch.assert_called_once()
    next_call.assert_not_called()


def test_claimed_effect_keyboard_interrupt_marks_only_current_call_unknown(
    monkeypatch,
):
    _install_managed_relay(monkeypatch)
    agent = _build_agent(
        "terminal",
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 1}},
    )
    messages: list[dict] = []

    def _mark_interrupted(*_args, **_kwargs) -> None:
        agent._interrupt_requested = True

    agent.interrupt = MagicMock(side_effect=_mark_interrupted)
    calls = _assistant_message(
        _tool_call("terminal", "call-effect-interrupt", {"command": "do-work"}),
        _tool_call("web_search", "call-never-started-1", {"query": "one"}),
        _tool_call("web_search", "call-never-started-2", {"query": "two"}),
    )
    with patch(
        "run_agent.handle_function_call",
        side_effect=KeyboardInterrupt,
    ) as dispatch:
        with pytest.raises(KeyboardInterrupt):
            agent._execute_tool_calls_sequential(calls, messages, "task-1")

    by_id = {message["tool_call_id"]: message for message in messages}
    assert by_id["call-effect-interrupt"]["effect_disposition"] == "unknown"
    assert by_id["call-never-started-1"]["effect_disposition"] == "none"
    assert by_id["call-never-started-2"]["effect_disposition"] == "none"
    assert dispatch.call_count == 1
    agent.interrupt.assert_called_once()


def test_claimed_effect_cancelled_error_completes_transcript_before_reraise(
    monkeypatch,
):
    _install_managed_relay(monkeypatch)
    agent = _build_agent(
        "terminal",
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 1}},
    )
    messages: list[dict] = []
    cancellation = asyncio.CancelledError("cancelled after effect claim")

    def _mark_interrupted(*_args, **_kwargs) -> None:
        agent._interrupt_requested = True

    agent.interrupt = MagicMock(side_effect=_mark_interrupted)
    calls = _assistant_message(
        _tool_call("terminal", "call-effect-cancelled", {"command": "do-work"}),
        _tool_call("web_search", "call-cancelled-rest-1", {"query": "one"}),
        _tool_call("web_search", "call-cancelled-rest-2", {"query": "two"}),
    )

    caught = None
    with (
        patch("run_agent.handle_function_call", side_effect=cancellation) as dispatch,
        patch("agent.tool_executor._emit_terminal_post_tool_call") as terminal_row,
    ):
        try:
            agent._execute_tool_calls_sequential(calls, messages, "task-1")
        except asyncio.CancelledError as exc:
            caught = exc

    assert caught is cancellation
    assert [message["tool_call_id"] for message in messages] == [
        "call-effect-cancelled",
        "call-cancelled-rest-1",
        "call-cancelled-rest-2",
    ]
    by_id = {message["tool_call_id"]: message for message in messages}
    assert by_id["call-effect-cancelled"]["effect_disposition"] == "unknown"
    assert by_id["call-cancelled-rest-1"]["effect_disposition"] == "none"
    assert by_id["call-cancelled-rest-2"]["effect_disposition"] == "none"
    assert terminal_row.call_count == 1
    assert terminal_row.call_args.kwargs["tool_call_id"] == "call-effect-cancelled"
    assert terminal_row.call_args.kwargs["status"] == "cancelled"
    assert dispatch.call_count == 1
    agent.interrupt.assert_called_once()


def test_persistence_failure_does_not_swallow_claimed_cancelled_error(
    monkeypatch,
):
    _install_managed_relay(monkeypatch)
    agent = _build_agent(
        "terminal",
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 1}},
    )
    messages: list[dict] = []
    cancellation = asyncio.CancelledError("cancelled before persistence")

    def _mark_interrupted(*_args, **_kwargs) -> None:
        agent._interrupt_requested = True

    agent.interrupt = MagicMock(side_effect=_mark_interrupted)
    agent._flush_messages_to_session_db = MagicMock(return_value=False)
    agent.tool_complete_callback = MagicMock()
    calls = _assistant_message(
        _tool_call("terminal", "call-persist-current", {"command": "do-work"}),
        _tool_call("web_search", "call-persist-rest", {"query": "never"}),
    )

    with (
        patch("run_agent.handle_function_call", side_effect=cancellation) as dispatch,
        pytest.raises(asyncio.CancelledError) as caught,
    ):
        agent._execute_tool_calls_sequential(calls, messages, "task-1")

    assert caught.value is cancellation
    assert [message["tool_call_id"] for message in messages] == [
        "call-persist-current",
        "call-persist-rest",
    ]
    assert messages[0]["effect_disposition"] == "unknown"
    assert messages[1]["effect_disposition"] == "none"
    assert agent._incremental_persistence_failed is True
    assert agent._flush_messages_to_session_db.call_count == 2
    agent.tool_complete_callback.assert_not_called()
    agent.interrupt.assert_called_once()
    dispatch.assert_called_once()


def test_unmanaged_bypass_claimed_interrupt_completes_transcript_before_reraise(
    monkeypatch,
):
    _install_unmanaged_relay(monkeypatch)
    agent = _build_agent(
        "terminal",
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 1}},
    )
    messages: list[dict] = []
    interruption = KeyboardInterrupt("interrupted after unmanaged effect claim")

    def _mark_interrupted(*_args, **_kwargs) -> None:
        agent._interrupt_requested = True

    agent.interrupt = MagicMock(side_effect=_mark_interrupted)
    calls = _assistant_message(
        _tool_call("terminal", "call-unmanaged-interrupt", {"command": "do-work"}),
        _tool_call("web_search", "call-unmanaged-rest", {"query": "never"}),
    )

    caught = None
    with (
        patch("run_agent.handle_function_call", side_effect=interruption) as dispatch,
        patch("agent.tool_executor._emit_terminal_post_tool_call") as terminal_row,
    ):
        try:
            agent._execute_tool_calls_sequential(calls, messages, "task-1")
        except KeyboardInterrupt as exc:
            caught = exc

    assert caught is interruption
    assert [message["tool_call_id"] for message in messages] == [
        "call-unmanaged-interrupt",
        "call-unmanaged-rest",
    ]
    assert messages[0]["effect_disposition"] == "unknown"
    assert messages[1]["effect_disposition"] == "none"
    assert terminal_row.call_count == 1
    assert terminal_row.call_args.kwargs["tool_call_id"] == "call-unmanaged-interrupt"
    assert terminal_row.call_args.kwargs["status"] == "cancelled"
    assert dispatch.call_count == 1
    agent.interrupt.assert_called_once()


def test_post_start_gate_timeout_emits_one_completion_after_canonical_row(
    monkeypatch,
):
    _install_managed_relay(monkeypatch)
    agent = _build_agent(
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 0.05}},
    )
    timeline: list[tuple] = []
    preflight_started = threading.Event()
    release_preflight = threading.Event()

    class _RecordingMessages(list):
        def append(self, message):
            super().append(message)
            timeline.append(("canonical-row", message))

    messages = _RecordingMessages()

    def _progress(event, name, *_args, **kwargs) -> None:
        timeline.append(("projection", event, name, kwargs))

    def _slow_start_callback(*_args, **_kwargs) -> None:
        preflight_started.set()
        assert release_preflight.wait(timeout=2)

    def _release_after_deadline() -> None:
        assert preflight_started.wait(timeout=2)
        assert not release_preflight.wait(timeout=0.12)
        release_preflight.set()

    agent.tool_progress_callback = _progress
    agent.tool_start_callback = _slow_start_callback
    helper = threading.Thread(target=_release_after_deadline, daemon=True)
    helper.start()
    with patch("run_agent.handle_function_call") as dispatch:
        agent._execute_tool_calls_sequential(
            _assistant_message(
                _tool_call("web_search", "call-post-start-timeout", {"query": "x"})
            ),
            messages,
            "task-1",
        )
    helper.join(timeout=2)

    started = [
        event
        for event in timeline
        if event[:3] == ("projection", "tool.started", "web_search")
    ]
    completed = [
        event
        for event in timeline
        if event[:3] == ("projection", "tool.completed", "web_search")
    ]
    rows = [event for event in timeline if event[0] == "canonical-row"]

    assert not helper.is_alive()
    assert len(started) == 1
    assert len(rows) == 1
    assert len(completed) == 1
    assert timeline.index(started[0]) < timeline.index(rows[0])
    assert timeline.index(rows[0]) < timeline.index(completed[0])
    assert completed[0][3]["is_error"] is True
    assert rows[0][1]["tool_call_id"] == "call-post-start-timeout"
    assert rows[0][1]["effect_disposition"] == "none"
    dispatch.assert_not_called()


@pytest.mark.parametrize("terminal_kind", ["timeout", "capacity"])
def test_pre_callback_terminal_result_has_no_start_and_one_completion(
    monkeypatch,
    terminal_kind,
):
    if terminal_kind == "timeout":
        runtime = _ManagedRelayRuntime()

        def _never_reaches_callback(*_args, **_kwargs):
            async def _wedge():
                await asyncio.Event().wait()

            return _wedge()

        runtime.run_in_session_async = _never_reaches_callback
        session = SimpleNamespace(handle=object())
        monkeypatch.setattr(
            relay_tools.relay_runtime,
            "resolve_execution_context",
            lambda _session_id: (runtime, session, None),
        )
    else:
        _install_managed_relay(monkeypatch)
        monkeypatch.setattr(relay_tools, "_MAX_SEQUENTIAL_RELAY_WORKERS", 0)

    agent = _build_agent(
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 0.02}},
    )
    timeline: list[tuple] = []

    class _RecordingMessages(list):
        def append(self, message):
            super().append(message)
            timeline.append(("canonical-row", message))

    messages = _RecordingMessages()
    agent.tool_progress_callback = lambda event, name, *_args, **_kwargs: (
        timeline.append(("progress", event, name))
    )
    agent.tool_start_callback = lambda *_args: timeline.append(("structured-start",))
    agent.tool_complete_callback = lambda *_args: timeline.append((
        "structured-complete",
    ))

    with patch("run_agent.handle_function_call") as dispatch:
        agent._execute_tool_calls_sequential(
            _assistant_message(
                _tool_call("web_search", f"call-{terminal_kind}", {"query": "x"})
            ),
            messages,
            "task-1",
        )

    starts = [
        event
        for event in timeline
        if event[0] == "structured-start" or event[:2] == ("progress", "tool.started")
    ]
    progress_completions = [
        event
        for event in timeline
        if event[:3] == ("progress", "tool.completed", "web_search")
    ]
    structured_completions = [
        event for event in timeline if event[0] == "structured-complete"
    ]
    canonical_rows = [event for event in timeline if event[0] == "canonical-row"]

    assert starts == []
    assert len(canonical_rows) == 1
    assert len(progress_completions) == 1
    assert len(structured_completions) == 1
    assert timeline.index(canonical_rows[0]) < timeline.index(progress_completions[0])
    assert timeline.index(canonical_rows[0]) < timeline.index(structured_completions[0])
    assert messages[0]["effect_disposition"] == "none"
    dispatch.assert_not_called()


@pytest.mark.parametrize("effect_disposition", ["none", "unknown"])
def test_timeout_handler_maps_effect_disposition_and_next_call_runs(
    effect_disposition,
):
    agent = _build_agent(
        "write_file",
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 0.125}},
    )
    messages: list[dict] = []
    mutation_records: list[tuple] = []
    agent._turn_failed_file_mutations = {}
    agent._turn_file_mutation_paths = set()
    original_recorder = agent._record_file_mutation_result

    def _record(*args, **kwargs):
        mutation_records.append((args, kwargs))
        return original_recorder(*args, **kwargs)

    agent._record_file_mutation_result = _record

    def _relay_execute(name, args, callback, **kwargs):
        del kwargs
        if name == "write_file":
            raise relay_tools.SequentialRelayToolTimeout(
                0.125,
                effect_disposition,
            )
        return callback(dict(args)), dict(args)

    calls = _assistant_message(
        _tool_call(
            "write_file",
            "call-timeout",
            {"path": "result.txt", "content": "value"},
        ),
        _tool_call("web_search", "call-next", {"query": str(uuid.uuid4())}),
    )
    with (
        patch("agent.relay_tools.execute", side_effect=_relay_execute),
        patch(
            "run_agent.handle_function_call",
            return_value='{"ok": true}',
        ) as dispatch,
    ):
        agent._execute_tool_calls_sequential(calls, messages, "task-1")

    by_id = {message["tool_call_id"]: message for message in messages}
    timeout_message = by_id["call-timeout"]
    timeout_payload = json.loads(timeout_message["content"])
    assert timeout_payload["error_type"] == "sequential_relay_timeout"
    assert timeout_payload["timeout_seconds"] == pytest.approx(0.125)
    assert timeout_payload["effect_disposition"] == effect_disposition
    assert timeout_message["effect_disposition"] == effect_disposition

    assert "ok" in by_id["call-next"]["content"]
    assert dispatch.call_count == 1
    assert all(record[0][0] != "write_file" for record in mutation_records)
    assert agent._turn_failed_file_mutations == {}


@pytest.mark.parametrize("cancellation_kind", ["keyboard", "cancelled"])
def test_segmented_hard_cancellation_pairs_all_later_calls(cancellation_kind):
    from agent import tool_executor

    calls = [
        _tool_call("web_search", "parallel-before-1"),
        _tool_call("web_search", "parallel-before-2"),
        _tool_call("terminal", "sequential-barrier"),
        _tool_call("web_search", "parallel-after-1"),
        _tool_call("web_search", "parallel-after-2"),
    ]
    segments = [
        ("parallel", calls[:2]),
        ("sequential", calls[2:3]),
        ("parallel", calls[3:]),
    ]
    messages: list[dict] = []
    agent = SimpleNamespace(_incremental_persistence_failed=False)
    cancellation = (
        KeyboardInterrupt("stop at barrier")
        if cancellation_kind == "keyboard"
        else asyncio.CancelledError("cancel at barrier")
    )

    def _complete_parallel(_agent, segment_message, output, *_args, **_kwargs):
        for call in segment_message.tool_calls:
            output.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": '{"ok": true}',
                "effect_disposition": "none",
            })

    def _cancel_barrier(_agent, segment_message, output, *_args, **_kwargs):
        call = segment_message.tool_calls[0]
        output.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": "[Tool execution cancelled after start]",
            "effect_disposition": "unknown",
        })
        raise cancellation

    with (
        patch.object(
            tool_executor,
            "execute_tool_calls_concurrent",
            side_effect=_complete_parallel,
        ) as concurrent,
        patch.object(
            tool_executor,
            "execute_tool_calls_sequential",
            side_effect=_cancel_barrier,
        ) as sequential,
        patch.object(
            tool_executor,
            "_flush_session_db_after_tool_progress",
            return_value=True,
        ) as flush,
        pytest.raises(type(cancellation)) as caught,
    ):
        tool_executor.execute_tool_calls_segmented(
            agent,
            _assistant_message(*calls),
            messages,
            "task-1",
            segments=segments,
        )

    assert caught.value is cancellation
    assert [message["tool_call_id"] for message in messages] == [
        "parallel-before-1",
        "parallel-before-2",
        "sequential-barrier",
        "parallel-after-1",
        "parallel-after-2",
    ]
    assert messages[2]["effect_disposition"] == "unknown"
    for message in messages[3:]:
        assert message["effect_disposition"] == "none"
        assert "cancelled" in message["content"]
    assert concurrent.call_count == 1
    assert sequential.call_count == 1
    flush.assert_called_once_with(
        agent,
        messages,
        stage="cancelled remaining segmented tool results",
    )


def test_segmented_unbounded_cancellation_pairs_missing_current_and_future(
    monkeypatch,
):
    from agent import tool_executor

    _install_unmanaged_relay(monkeypatch)
    agent = _build_agent(
        "terminal",
        "web_search",
        config={"agent": {"sequential_tool_execution_timeout": 0}},
    )
    calls = [
        _tool_call("terminal", "unbounded-current", {"command": "do-work"}),
        _tool_call("web_search", "future-parallel-1", {"query": "one"}),
        _tool_call("web_search", "future-parallel-2", {"query": "two"}),
    ]
    segments = [
        ("sequential", calls[:1]),
        ("parallel", calls[1:]),
    ]
    messages: list[dict] = []
    cancellation = asyncio.CancelledError("unbounded callback cancelled")

    with (
        patch("run_agent.handle_function_call", side_effect=cancellation) as dispatch,
        patch.object(tool_executor, "execute_tool_calls_concurrent") as concurrent,
        patch.object(
            tool_executor,
            "_flush_session_db_after_tool_progress",
            return_value=True,
        ) as flush,
        pytest.raises(asyncio.CancelledError) as caught,
    ):
        tool_executor.execute_tool_calls_segmented(
            agent,
            _assistant_message(*calls),
            messages,
            "task-1",
            segments=segments,
        )

    assert caught.value is cancellation
    assert [message["tool_call_id"] for message in messages] == [
        "unbounded-current",
        "future-parallel-1",
        "future-parallel-2",
    ]
    assert messages[0]["effect_disposition"] == "unknown"
    assert "may have started" in messages[0]["content"]
    for message in messages[1:]:
        assert message["effect_disposition"] == "none"
        assert "cancelled" in message["content"]
    dispatch.assert_called_once()
    concurrent.assert_not_called()
    flush.assert_called_once_with(
        agent,
        messages,
        stage="cancelled remaining segmented tool results",
    )


def test_segmented_final_unbounded_keyboard_interrupt_flushes_prepaired_row(
    monkeypatch,
):
    from agent import tool_executor

    _install_unmanaged_relay(monkeypatch)
    agent = _build_agent(
        "terminal",
        config={"agent": {"sequential_tool_execution_timeout": 0}},
    )
    call = _tool_call("terminal", "final-unbounded", {"command": "do-work"})
    messages: list[dict] = []
    interruption = KeyboardInterrupt("final segment interrupted")

    with (
        patch("run_agent.handle_function_call", side_effect=interruption) as dispatch,
        patch.object(
            tool_executor,
            "_flush_session_db_after_tool_progress",
            return_value=True,
        ) as flush,
        pytest.raises(KeyboardInterrupt) as caught,
    ):
        tool_executor.execute_tool_calls_segmented(
            agent,
            _assistant_message(call),
            messages,
            "task-1",
            segments=[("sequential", [call])],
        )

    assert caught.value is interruption
    assert [message["tool_call_id"] for message in messages] == ["final-unbounded"]
    assert messages[0]["effect_disposition"] == "unknown"
    assert "had already started" in messages[0]["content"]
    dispatch.assert_called_once()
    flush.assert_called_once_with(
        agent,
        messages,
        stage="cancelled remaining segmented tool results",
    )


def test_segmented_parallel_interrupt_marks_all_missing_current_calls_unknown():
    from agent import tool_executor

    calls = [
        _tool_call("web_search", "parallel-current-1"),
        _tool_call("web_search", "parallel-current-2"),
        _tool_call("terminal", "future-barrier"),
    ]
    segments = [
        ("parallel", calls[:2]),
        ("sequential", calls[2:]),
    ]
    messages: list[dict] = []
    agent = SimpleNamespace(_incremental_persistence_failed=False)
    interruption = KeyboardInterrupt("interrupt parallel segment")

    with (
        patch.object(
            tool_executor,
            "execute_tool_calls_concurrent",
            side_effect=interruption,
        ) as concurrent,
        patch.object(tool_executor, "execute_tool_calls_sequential") as sequential,
        patch.object(
            tool_executor,
            "_flush_session_db_after_tool_progress",
            return_value=True,
        ) as flush,
        pytest.raises(KeyboardInterrupt) as caught,
    ):
        tool_executor.execute_tool_calls_segmented(
            agent,
            _assistant_message(*calls),
            messages,
            "task-1",
            segments=segments,
        )

    assert caught.value is interruption
    assert [message["tool_call_id"] for message in messages] == [
        "parallel-current-1",
        "parallel-current-2",
        "future-barrier",
    ]
    for message in messages[:2]:
        assert message["effect_disposition"] == "unknown"
        assert "may have started" in message["content"]
    assert messages[2]["effect_disposition"] == "none"
    assert "cancelled" in messages[2]["content"]
    concurrent.assert_called_once()
    sequential.assert_not_called()
    flush.assert_called_once_with(
        agent,
        messages,
        stage="cancelled remaining segmented tool results",
    )
