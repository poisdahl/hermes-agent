"""Regression coverage for blocked Relay-managed Hermes callbacks."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest


def _run_starvation_scenario() -> None:
    from agent import relay_runtime, relay_tools
    from tools.thread_context import propagate_context_to_thread

    relay_runtime._reset_for_tests()
    lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
        profile_key=relay_runtime.current_profile_key(),
        session_id="relay-starvation-session",
        platform="cli",
    )
    turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
        lease,
        turn_id="relay-starvation-turn",
        task_id="relay-starvation-task",
    )
    host = lease.host
    assert isinstance(host, relay_runtime.RelayRuntime)
    assert lease.session is not None
    consumer = "test.relay_callback_starvation"
    host.retain_managed_execution(consumer)
    assert host.managed_execution_enabled()
    assert turn.relay_enabled

    release_callbacks = threading.Event()
    both_callbacks_entered = threading.Event()
    third_callback_entered = threading.Event()
    third_call_finished = threading.Event()
    counter_lock = threading.Lock()
    entered_count = 0
    errors: list[BaseException] = []
    third_results: list[tuple[object, dict]] = []

    def assert_managed_context() -> None:
        resolved_host, resolved_session, _ = relay_runtime.resolve_execution_context(
            lease.session_id
        )
        assert resolved_host is host
        assert resolved_session is lease.session

    def blocking_callback(args: dict) -> dict:
        nonlocal entered_count
        with counter_lock:
            entered_count += 1
            if entered_count == 2:
                both_callbacks_entered.set()
        if not release_callbacks.wait(timeout=30):
            raise AssertionError("test did not release the blocked callback")
        return {"released": args["index"]}

    def run_blocking_call(index: int) -> None:
        try:
            assert_managed_context()
            relay_tools.execute(
                f"blocking-tool-{index}",
                {"index": index},
                blocking_callback,
                session_id=lease.session_id,
            )
        except BaseException as exc:
            errors.append(exc)

    def third_callback(_args: dict) -> dict:
        third_callback_entered.set()
        return {"progress": "independent"}

    def run_third_call() -> None:
        try:
            assert_managed_context()
            third_results.append(
                relay_tools.execute(
                    "independent-tool",
                    {},
                    third_callback,
                    session_id=lease.session_id,
                )
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            third_call_finished.set()

    blocking_threads = [
        threading.Thread(
            target=propagate_context_to_thread(run_blocking_call),
            args=(index,),
            daemon=True,
            name=f"relay-blocking-caller-{index}",
        )
        for index in range(2)
    ]
    third_thread: threading.Thread | None = None

    try:
        for thread in blocking_threads:
            thread.start()
        assert both_callbacks_entered.wait(timeout=5), errors

        third_thread = threading.Thread(
            target=propagate_context_to_thread(run_third_call),
            daemon=True,
            name="relay-independent-caller",
        )
        third_thread.start()

        assert third_callback_entered.wait(timeout=5), errors
        assert third_call_finished.wait(timeout=5), errors
        assert not release_callbacks.is_set()
        assert all(thread.is_alive() for thread in blocking_threads)
        assert not errors
        assert third_results == [({"progress": "independent"}, {})]
    finally:
        release_callbacks.set()
        for thread in [*blocking_threads, third_thread]:
            if thread is not None:
                thread.join(timeout=10)
        all_callers_finished = all(
            thread is None or not thread.is_alive()
            for thread in [*blocking_threads, third_thread]
        )
        operations_finished = host._operations_idle.wait(timeout=10)

        host.release_managed_execution(consumer)
        relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="success")
        relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
        idle_before_reset = host._operations_idle.is_set()

        relay_runtime._reset_for_tests()

    assert all_callers_finished
    assert operations_finished
    assert idle_before_reset
    assert not errors


def test_blocked_managed_callbacks_do_not_starve_later_call() -> None:
    """Let an unrelated managed call progress past blocked callbacks (#79568).

    Relay 0.6 ran synchronous Python callbacks on a shared native worker pool,
    so two blocked callbacks could starve every later managed call. The child
    process fixes the pool at two workers before Relay imports; Relay 0.7 must
    let the third callback finish before either blocker is released.
    """
    pytest.importorskip("nemo_relay")
    env = os.environ.copy()
    env.pop("HERMES_NEMO_RELAY_PLUGINS_TOML", None)
    env["TOKIO_WORKER_THREADS"] = "2"
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-m", "tests.agent.test_relay_callback_starvation"],
        cwd=repo_root,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, (
        f"starvation scenario failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


if __name__ == "__main__":
    _run_starvation_scenario()
