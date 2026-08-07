"""Regression test for the jobs.json cross-process lock.

Background: ``hermes cron pause`` runs in its own process (CLI → cronjob tool →
``pause_job`` → ``update_job`` → ``save_jobs``), entirely separate from the
gateway process that also writes ``jobs.json`` (``mark_job_run`` /
``advance_next_run`` / due-fast-forward). The module's ``threading.Lock`` only
serializes writers *inside one process*, so a CLI pause issued while the gateway
was live could be silently lost to a concurrent gateway write — the job kept
firing even though the CLI reported "Paused".

``_jobs_lock()`` closes that gap with a short-held cross-process advisory file
lock. This test proves the lock actually excludes a *separate process*, which an
in-process ``threading.Lock`` cannot do.
"""

import contextlib
import copy
import errno
import os
import stat
import subprocess
import sys
import textwrap
import time
from typing import Any, cast

import pytest

from cron import jobs
import utils as file_utils


# Repo root (parent of the ``cron`` package) so the child process can import it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(jobs.__file__)))


@pytest.mark.skipif(jobs.fcntl is None, reason="POSIX fcntl/flock required")
def test_jobs_lock_excludes_another_process(tmp_path, monkeypatch):
    cron_dir = tmp_path / "cron"
    output_dir = cron_dir / "output"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", output_dir)

    ready = tmp_path / "child_holds_lock"
    release = tmp_path / "child_may_release"
    blocker_started = tmp_path / "blocker_started"
    blocker_acquired = tmp_path / "blocker_acquired"
    holder = tmp_path / "holder.py"
    holder.write_text(
        textwrap.dedent(
            f"""
            import sys, time, pathlib
            sys.path.insert(0, {_REPO_ROOT!r})
            from cron import jobs

            jobs.CRON_DIR = pathlib.Path({str(cron_dir)!r})
            jobs.JOBS_FILE = jobs.CRON_DIR / "jobs.json"
            jobs.OUTPUT_DIR = jobs.CRON_DIR / "output"

            with jobs._jobs_lock():
                pathlib.Path({str(ready)!r}).write_text("1")
                # Hold the lock until the parent signals (bounded so a wedged
                # test can never hang CI).
                for _ in range(1000):
                    if pathlib.Path({str(release)!r}).exists():
                        break
                    time.sleep(0.01)
            """
        )
    )

    blocker = tmp_path / "blocker.py"
    blocker.write_text(
        textwrap.dedent(
            f"""
            import sys, pathlib
            sys.path.insert(0, {_REPO_ROOT!r})
            from cron import jobs

            jobs.CRON_DIR = pathlib.Path({str(cron_dir)!r})
            jobs.JOBS_FILE = jobs.CRON_DIR / "jobs.json"
            jobs.OUTPUT_DIR = jobs.CRON_DIR / "output"

            pathlib.Path({str(blocker_started)!r}).write_text("1")
            with jobs._jobs_lock():
                pathlib.Path({str(blocker_acquired)!r}).write_text("1")
            """
        )
    )

    child = subprocess.Popen([sys.executable, str(holder)])
    blocker_child = None
    try:
        # Wait until the child is inside the critical section.
        for _ in range(1000):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.exists(), "child never acquired _jobs_lock()"

        # While the child holds it, a non-blocking acquire of the SAME lock file
        # from this process must fail. A threading.Lock could never block here.
        lock_file = jobs._jobs_lock_file()
        fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT)
        try:
            with pytest.raises(OSError):
                jobs.fcntl.flock(fd, jobs.fcntl.LOCK_EX | jobs.fcntl.LOCK_NB)
        finally:
            os.close(fd)

        # A second _jobs_lock() caller in another process should block until the
        # holder releases, rather than falling through with only a process-local
        # threading lock.
        blocker_child = subprocess.Popen([sys.executable, str(blocker)])
        for _ in range(1000):
            if blocker_started.exists():
                break
            time.sleep(0.01)
        assert blocker_started.exists(), "blocker process never started"
        time.sleep(0.05)
        assert not blocker_acquired.exists(), "second process entered _jobs_lock() while held"
    finally:
        release.write_text("1")
        child.wait(timeout=15)
        if blocker_child is not None:
            blocker_child.wait(timeout=15)

    assert blocker_acquired.exists(), "second process did not acquire _jobs_lock() after release"

    # Once the child has released, the lock is freely acquirable again.
    with jobs._jobs_lock():
        pass


@pytest.mark.skipif(jobs.fcntl is None, reason="POSIX fcntl/flock required")
def test_real_degraded_process_preserves_disjoint_sibling_update(
    tmp_path, monkeypatch
):
    """Two real processes retain disjoint edits when one broad lock degrades."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    job_a = {"id": "aaaaaaaaaaaa", "name": "base-a", "enabled": True}
    job_b = {"id": "bbbbbbbbbbbb", "name": "base-b", "enabled": True}
    jobs.save_jobs([job_a, job_b])

    writer_a_loaded = tmp_path / "writer_a_loaded"
    writer_b_saved = tmp_path / "writer_b_saved"
    writer_a = tmp_path / "writer_a.py"
    writer_b = tmp_path / "writer_b.py"
    writer_a.write_text(
        textwrap.dedent(
            f"""
            import pathlib, sys, time
            sys.path.insert(0, {_REPO_ROOT!r})
            from cron import jobs

            jobs.CRON_DIR = pathlib.Path({str(cron_dir)!r})
            jobs.JOBS_FILE = jobs.CRON_DIR / "jobs.json"
            jobs.OUTPUT_DIR = jobs.CRON_DIR / "output"

            with jobs._jobs_lock():
                loaded = jobs.load_jobs()
                pathlib.Path({str(writer_a_loaded)!r}).write_text("1")
                for _ in range(1000):
                    if pathlib.Path({str(writer_b_saved)!r}).exists():
                        break
                    time.sleep(0.01)
                else:
                    raise RuntimeError("degraded sibling never published")
                loaded[0]["name"] = "writer-a-update"
                jobs.save_jobs(loaded)
            """
        ),
        encoding="utf-8",
    )
    writer_b.write_text(
        textwrap.dedent(
            f"""
            import pathlib, sys
            sys.path.insert(0, {_REPO_ROOT!r})
            from cron import jobs

            jobs.CRON_DIR = pathlib.Path({str(cron_dir)!r})
            jobs.JOBS_FILE = jobs.CRON_DIR / "jobs.json"
            jobs.OUTPUT_DIR = jobs.CRON_DIR / "output"
            jobs._JOBS_LOCK_TIMEOUT_SECONDS = 0.05

            with jobs._jobs_lock():
                if not jobs._jobs_lock_state.degraded:
                    raise RuntimeError("writer B unexpectedly acquired broad lock")
                loaded = jobs.load_jobs()
                loaded[1]["enabled"] = False
                jobs.save_jobs(loaded)
                pathlib.Path({str(writer_b_saved)!r}).write_text("1")
            """
        ),
        encoding="utf-8",
    )

    processes = []
    try:
        process_a = subprocess.Popen(
            [sys.executable, str(writer_a)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process_a)
        for _ in range(1000):
            if writer_a_loaded.exists() or process_a.poll() is not None:
                break
            time.sleep(0.01)
        assert writer_a_loaded.exists(), "writer A never loaded while holding lock"

        process_b = subprocess.Popen(
            [sys.executable, str(writer_b)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process_b)
        stdout_b, stderr_b = process_b.communicate(timeout=15)
        stdout_a, stderr_a = process_a.communicate(timeout=15)
        assert process_b.returncode == 0, (stdout_b, stderr_b)
        assert process_a.returncode == 0, (stdout_a, stderr_a)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

    on_disk = {job["id"]: job for job in jobs.load_jobs()}
    assert on_disk[job_a["id"]]["name"] == "writer-a-update"
    assert on_disk[job_b["id"]]["enabled"] is False


@pytest.mark.skipif(jobs.fcntl is None, reason="POSIX fcntl/flock required")
def test_jobs_commit_lock_fails_closed_on_contention(tmp_path, monkeypatch):
    """The short publication lock must reject, not degrade, on contention."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    monkeypatch.setattr(jobs, "_JOBS_COMMIT_LOCK_TIMEOUT_SECONDS", 0.0)
    jobs.ensure_dirs()

    with jobs._jobs_commit_lock():
        with pytest.raises(RuntimeError, match="refusing to publish"):
            with jobs._jobs_commit_lock():
                pytest.fail("a contending writer entered the commit section")

    # A failed contender must close its descriptor and leave the lock usable.
    with jobs._jobs_commit_lock():
        pass
    lock_stat = os.stat(jobs._jobs_commit_lock_file())
    parent_stat = os.stat(cron_dir)
    assert stat.S_IMODE(lock_stat.st_mode) == 0o600
    assert (lock_stat.st_uid, lock_stat.st_gid) == (
        parent_stat.st_uid,
        parent_stat.st_gid,
    )


def test_jobs_commit_lock_uses_windows_fallback_and_requires_backend(
    tmp_path, monkeypatch
):
    """The msvcrt branch locks/unlocks; no backend fails closed."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.ensure_dirs()

    class FakeMsvcrt:
        LK_NBLCK = "nonblocking-lock"
        LK_UNLCK = "unlock"

        def __init__(self):
            self.calls = []

        def locking(self, fd, mode, length):
            if os.fstat(fd).st_size < 1:
                raise OSError("Windows byte-range lock requires a real byte")
            position = os.lseek(fd, 0, os.SEEK_CUR)
            if position != 0:
                raise OSError(f"Windows lock offset must be zero, got {position}")
            self.calls.append((fd, mode, length))
            if mode == self.LK_NBLCK:
                # Prove the release path seeks independently instead of
                # assuming the acquisition backend preserved the offset.
                os.lseek(fd, 1, os.SEEK_SET)

    fake_msvcrt = FakeMsvcrt()
    monkeypatch.setattr(jobs, "fcntl", None)
    monkeypatch.setattr(jobs, "msvcrt", fake_msvcrt)
    with jobs._jobs_commit_lock():
        pass
    assert [call[1:] for call in fake_msvcrt.calls] == [
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_UNLCK, 1),
    ]

    monkeypatch.setattr(jobs, "msvcrt", None)
    monkeypatch.setattr(jobs, "_JOBS_COMMIT_LOCK_TIMEOUT_SECONDS", 0.0)
    with pytest.raises(RuntimeError, match="refusing to publish"):
        with jobs._jobs_commit_lock():
            pytest.fail("commit proceeded without a cross-process lock backend")


@pytest.mark.skipif(
    os.name != "posix" or jobs.fcntl is None,
    reason="POSIX no-follow descriptor locking required",
)
def test_jobs_commit_lock_rejects_symlink_without_touching_target(
    tmp_path, monkeypatch
):
    """The publication lock must never chmod/chown/fock a symlink target."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.ensure_dirs()

    victim = tmp_path / "victim"
    victim.write_text("do not touch", encoding="utf-8")
    victim.chmod(0o640)
    jobs._jobs_commit_lock_file().symlink_to(victim)

    with pytest.raises(RuntimeError, match="refusing to publish"):
        with jobs._jobs_commit_lock():
            pytest.fail("symlink-backed commit lock was accepted")

    assert victim.read_text(encoding="utf-8") == "do not touch"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o640


@pytest.mark.skipif(
    os.name != "posix" or jobs.fcntl is None,
    reason="POSIX regular descriptor link-count validation required",
)
def test_jobs_commit_lock_rejects_hardlink_without_touching_target(
    tmp_path, monkeypatch
):
    """A planted hard link cannot redirect lock metadata changes to a victim."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.ensure_dirs()

    victim = tmp_path / "victim"
    victim.write_text("do not touch", encoding="utf-8")
    victim.chmod(0o640)
    os.link(victim, jobs._jobs_commit_lock_file())

    with pytest.raises(RuntimeError, match="refusing to publish"):
        with jobs._jobs_commit_lock():
            pytest.fail("hard-linked commit lock was accepted")

    assert victim.read_text(encoding="utf-8") == "do not touch"
    assert stat.S_IMODE(victim.stat().st_mode) == 0o640


@pytest.mark.skipif(
    os.name != "posix" or jobs.fcntl is None,
    reason="POSIX descriptor identity validation required",
)
def test_jobs_commit_lock_rejects_post_open_path_swap(tmp_path, monkeypatch):
    """Locking an unlinked inode must not license publication via a new path."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.ensure_dirs()

    lock_path = jobs._jobs_commit_lock_file()
    lock_path.write_text("original", encoding="utf-8")
    replacement = cron_dir / ".replacement-lock"
    replacement.write_text("replacement", encoding="utf-8")
    real_fchmod = jobs.os.fchmod
    swapped = False

    def swap_path_after_fd_open(fd, mode):
        nonlocal swapped
        real_fchmod(fd, mode)
        if not swapped:
            swapped = True
            old_inode = cron_dir / ".opened-lock-inode"
            os.replace(lock_path, old_inode)
            os.replace(replacement, lock_path)

    monkeypatch.setattr(jobs.os, "fchmod", swap_path_after_fd_open)

    with pytest.raises(RuntimeError, match="refusing to publish"):
        with jobs._jobs_commit_lock():
            pytest.fail("path-swapped commit lock was accepted")

    assert lock_path.read_text(encoding="utf-8") == "replacement"


@pytest.mark.skipif(
    os.name != "posix" or jobs.fcntl is None,
    reason="POSIX descriptor identity validation required",
)
def test_save_revalidates_commit_lock_identity_at_publication(
    tmp_path, monkeypatch
):
    """A lock-path swap during staging must abort before jobs.json replace."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    original = [{"id": "aaaaaaaaaaaa", "name": "original"}]
    jobs.save_jobs(original)
    lock_path = jobs._jobs_commit_lock_file()
    replacement = cron_dir / ".replacement-lock"
    replacement.write_text("replacement", encoding="utf-8")
    real_stamp = jobs._jobs_file_stamp
    swapped = False

    def swap_lock_path_during_generation_check(path):
        nonlocal swapped
        stamp = real_stamp(path)
        if not swapped:
            swapped = True
            os.replace(lock_path, cron_dir / ".opened-lock-inode")
            os.replace(replacement, lock_path)
        return stamp

    monkeypatch.setattr(jobs, "_jobs_file_stamp", swap_lock_path_during_generation_check)

    with pytest.raises(RuntimeError, match="identity changed; refusing to publish"):
        jobs.save_jobs([{"id": "aaaaaaaaaaaa", "name": "must not publish"}])

    assert jobs.json.loads(jobs.JOBS_FILE.read_text(encoding="utf-8"))["jobs"] == original
    assert list(cron_dir.glob(".jobs_*.tmp")) == []


@pytest.mark.parametrize("bare_list", [False, True])
def test_load_jobs_auto_repair_preserves_nested_save_paths(
    tmp_path, monkeypatch, bare_list
):
    """Control-character and bare-list repairs still publish valid wrapped JSON."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.ensure_dirs()

    job = {"id": "aaaaaaaaaaaa", "name": "bare" if bare_list else "bad\x01name"}
    if bare_list:
        raw = jobs.json.dumps([job])
    else:
        raw = '{"jobs": [{"id": "aaaaaaaaaaaa", "name": "bad\x01name"}]}'
    jobs.JOBS_FILE.write_text(raw, encoding="utf-8")

    # Public mutation paths call load_jobs() while already holding _jobs_lock;
    # auto-repair's nested save must remain re-entrant under the commit lock.
    with jobs._jobs_lock():
        assert jobs.load_jobs() == [job]

    repaired_text = jobs.JOBS_FILE.read_text(encoding="utf-8")
    repaired = jobs.json.loads(repaired_text)
    assert repaired["jobs"] == [job]
    assert "\x01" not in repaired_text


def test_auto_repair_reloads_after_lock_and_preserves_sibling_create(
    tmp_path, monkeypatch
):
    """A repair begun outside _jobs_lock cannot overwrite a newer generation."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.ensure_dirs()

    job_a = {"id": "aaaaaaaaaaaa", "name": "original"}
    job_b = {"id": "bbbbbbbbbbbb", "name": "sibling-created"}
    jobs.JOBS_FILE.write_text(jobs.json.dumps([job_a]), encoding="utf-8")
    replacement = cron_dir / ".sibling-repair-generation"
    replacement.write_text(jobs.json.dumps([job_a, job_b]), encoding="utf-8")
    real_loads = jobs.json.loads
    replaced = False

    def replace_path_after_first_decode(payload, *args, **kwargs):
        nonlocal replaced
        data = real_loads(payload, *args, **kwargs)
        if not replaced:
            replaced = True
            os.replace(replacement, jobs.JOBS_FILE)
        return data

    monkeypatch.setattr(jobs.json, "loads", replace_path_after_first_decode)

    loaded = jobs.load_jobs()

    assert loaded == [job_a, job_b]
    repaired = real_loads(jobs.JOBS_FILE.read_text(encoding="utf-8"))
    assert repaired["jobs"] == [job_a, job_b]


def test_nested_auto_repair_returns_and_records_published_generation(
    tmp_path, monkeypatch
):
    """A repair inside an outer mutation lock leaves exact save provenance."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.ensure_dirs()

    job_a = {
        "id": "aaaaaaaaaaaa",
        "name": "original",
        "schedule": {"kind": "interval", "minutes": 5},
    }
    job_b = {"id": "bbbbbbbbbbbb", "name": "sibling-created"}
    jobs.JOBS_FILE.write_text(jobs.json.dumps([job_a]), encoding="utf-8")
    replacement = cron_dir / ".nested-repair-generation"
    replacement.write_text(jobs.json.dumps([job_a, job_b]), encoding="utf-8")
    real_loads = jobs.json.loads
    replaced = False

    def replace_path_after_first_decode(payload, *args, **kwargs):
        nonlocal replaced
        data = real_loads(payload, *args, **kwargs)
        if not replaced:
            replaced = True
            os.replace(replacement, jobs.JOBS_FILE)
        return data

    monkeypatch.setattr(jobs.json, "loads", replace_path_after_first_decode)

    with jobs._jobs_lock():
        loaded = jobs.load_jobs()
        assert loaded == [job_a, job_b]
        assert jobs._jobs_lock_state.load_jobs == loaded

        loaded[0]["schedule"]["minutes"] = 10
        assert jobs._jobs_lock_state.load_jobs[0]["schedule"]["minutes"] == 5

    repaired = real_loads(jobs.JOBS_FILE.read_text(encoding="utf-8"))
    assert repaired["jobs"] == [job_a, job_b]


def test_snapshot_parse_reopens_after_relaxed_failure_and_accepts_new_generation(
    tmp_path, monkeypatch
):
    """Strict+relaxed failure retries a replacement instead of poisoning load."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.ensure_dirs()

    expected = [{"id": "aaaaaaaaaaaa", "name": "valid replacement"}]
    jobs.JOBS_FILE.write_text('{"jobs": [', encoding="utf-8")
    replacement = cron_dir / ".valid-generation"
    replacement.write_text(
        jobs.json.dumps({"jobs": expected, "updated_at": "sibling"}),
        encoding="utf-8",
    )
    real_loads = jobs.json.loads
    replaced = False

    def replace_after_relaxed_failure(payload, *args, **kwargs):
        nonlocal replaced
        try:
            return real_loads(payload, *args, **kwargs)
        except jobs.json.JSONDecodeError:
            if kwargs.get("strict") is False and not replaced:
                replaced = True
                os.replace(replacement, jobs.JOBS_FILE)
            raise

    monkeypatch.setattr(jobs.json, "loads", replace_after_relaxed_failure)

    assert jobs.load_jobs() == expected


def test_stable_corrupt_snapshot_retries_each_generation_attempt(
    tmp_path, monkeypatch
):
    """A stable malformed file is reopened a bounded number of times."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.ensure_dirs()
    jobs.JOBS_FILE.write_text('{"jobs": [', encoding="utf-8")

    real_loads = jobs.json.loads
    attempts = []

    def count_parse_attempts(payload, *args, **kwargs):
        attempts.append(kwargs.get("strict", True))
        return real_loads(payload, *args, **kwargs)

    monkeypatch.setattr(jobs.json, "loads", count_parse_attempts)

    with pytest.raises(RuntimeError, match="corrupted and unrepairable"):
        jobs.load_jobs()

    assert attempts == [True, False] * jobs._JOBS_GENERATION_MAX_ATTEMPTS


def test_save_fails_closed_on_stable_unparseable_current_generation(
    tmp_path, monkeypatch
):
    """A stable corrupt sibling generation must never license overwrite."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    job = {"id": "aaaaaaaaaaaa", "name": "base"}
    corrupt = b'{"jobs": ['

    try:
        jobs.save_jobs([job])
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        assert jobs.load_jobs() == [job]

        jobs.JOBS_FILE.write_bytes(corrupt)
        with pytest.raises(RuntimeError, match="corrupt|read"):
            jobs._save_jobs_unlocked([{**job, "name": "ours"}])
    finally:
        jobs._jobs_lock_state.depth = 0
        jobs._jobs_lock_state.degraded = False
        jobs._jobs_lock_state.load_jobs = None

    assert jobs.JOBS_FILE.read_bytes() == corrupt
    assert list(cron_dir.glob(".jobs_*.tmp")) == []


def test_save_fails_closed_on_unreadable_changed_generation(tmp_path, monkeypatch):
    """A failed coherent re-read cannot be replaced using a path-only stamp."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    job = {"id": "aaaaaaaaaaaa", "name": "base"}

    try:
        jobs.save_jobs([job])
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        assert jobs.load_jobs() == [job]

        # A same-content replacement changes the inode without introducing a
        # merge conflict. The only safe outcome when that generation cannot be
        # read is to leave it byte-for-byte untouched.
        replacement = cron_dir / ".unreadable-generation"
        replacement.write_bytes(jobs.JOBS_FILE.read_bytes())
        os.replace(replacement, jobs.JOBS_FILE)
        current = jobs.JOBS_FILE.read_bytes()

        def deny_snapshot(_path):
            raise PermissionError("generation became unreadable")

        monkeypatch.setattr(jobs, "_read_jobs_file_snapshot", deny_snapshot)
        with pytest.raises(RuntimeError, match="read cron database"):
            jobs._save_jobs_unlocked([{**job, "name": "ours"}])
    finally:
        jobs._jobs_lock_state.depth = 0
        jobs._jobs_lock_state.degraded = False
        jobs._jobs_lock_state.load_jobs = None

    assert jobs.JOBS_FILE.read_bytes() == current
    assert list(cron_dir.glob(".jobs_*.tmp")) == []


def test_direct_save_without_prior_load_replaces_requested_snapshot(
    tmp_path, monkeypatch
):
    """load_jobs=None remains the intentional whole-snapshot save contract."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    old_jobs = [
        {"id": "aaaaaaaaaaaa", "name": "old-a"},
        {"id": "bbbbbbbbbbbb", "name": "old-b"},
    ]
    replacement = [{"id": "cccccccccccc", "name": "replacement"}]
    jobs.save_jobs(old_jobs)
    jobs.save_jobs(replacement)
    assert jobs.load_jobs() == replacement


def test_direct_save_without_prior_load_recovers_corrupt_current_snapshot(
    tmp_path, monkeypatch
):
    """Explicit whole-snapshot save can replace bytes that JSON cannot parse."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    jobs.ensure_dirs()
    jobs.JOBS_FILE.write_bytes(b"\xff\xfe{not valid JSON")

    replacement = [{"id": "aaaaaaaaaaaa", "name": "recovered"}]
    jobs.save_jobs(replacement)

    assert jobs.load_jobs() == replacement


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode semantics required")
def test_direct_save_without_prior_load_replaces_unreadable_content(
    tmp_path, monkeypatch
):
    """Whole-snapshot recovery needs target identity, not readable old bytes."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    jobs.save_jobs([{"id": "old", "name": "unreadable"}])
    jobs.JOBS_FILE.chmod(0o000)
    try:
        if os.access(jobs.JOBS_FILE, os.R_OK):
            pytest.skip("test user can still read mode-000 files")
        replacement = [{"id": "aaaaaaaaaaaa", "name": "recovered"}]
        jobs.save_jobs(replacement)
        assert jobs.load_jobs() == replacement
    finally:
        if jobs.JOBS_FILE.exists():
            jobs.JOBS_FILE.chmod(0o600)


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode semantics required")
def test_copy_fallback_reapplies_secure_jobs_file_mode(tmp_path, monkeypatch):
    """EXDEV/EBUSY fallback cannot retain a pre-existing broad target mode."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    jobs.save_jobs([{"id": "aaaaaaaaaaaa", "name": "original"}])
    jobs.JOBS_FILE.chmod(0o666)

    def force_cross_device_fallback(_source, _target):
        raise OSError(errno.EXDEV, "simulated cross-device replace")

    def fail_metadata_copy(_source, _target):
        raise OSError("simulated copystat failure")

    monkeypatch.setattr(file_utils.os, "replace", force_cross_device_fallback)
    monkeypatch.setattr(file_utils.shutil, "copystat", fail_metadata_copy)

    jobs.save_jobs([{"id": "aaaaaaaaaaaa", "name": "updated"}])

    assert stat.S_IMODE(jobs.JOBS_FILE.stat().st_mode) == 0o600
    assert jobs.load_jobs()[0]["name"] == "updated"


def test_three_way_merge_preserves_current_when_desired_is_unchanged():
    """Concurrent pause/delete wins where this writer left its base untouched."""
    base = [
        {"id": "aaaaaaaaaaaa", "name": "keep", "enabled": True},
        {"id": "bbbbbbbbbbbb", "name": "delete", "enabled": True},
    ]
    desired = copy.deepcopy(base)
    current = [{"id": "aaaaaaaaaaaa", "name": "keep", "enabled": False}]
    before = (copy.deepcopy(base), copy.deepcopy(desired), copy.deepcopy(current))

    merged = jobs._merge_jobs_three_way(base, desired, current)

    assert merged == current
    assert (base, desired, current) == before
    merged[0]["enabled"] = True
    assert current[0]["enabled"] is False


def test_three_way_merge_applies_desired_when_current_is_unchanged():
    """This writer's update/delete wins where the sibling kept the base."""
    base = [
        {"id": "aaaaaaaaaaaa", "name": "old"},
        {"id": "bbbbbbbbbbbb", "name": "delete"},
    ]
    desired = [{"id": "aaaaaaaaaaaa", "name": "updated"}]

    assert jobs._merge_jobs_three_way(base, desired, copy.deepcopy(base)) == desired


def test_three_way_merge_unions_distinct_and_identical_creates():
    desired = [
        {"id": "aaaaaaaaaaaa", "name": "ours"},
        {"id": "cccccccccccc", "name": "identical"},
    ]
    current = [
        {"id": "bbbbbbbbbbbb", "name": "theirs"},
        {"id": "cccccccccccc", "name": "identical"},
    ]

    merged = jobs._merge_jobs_three_way([], desired, current)

    assert merged == [
        desired[0],
        desired[1],
        current[0],
    ]
    merged[0]["name"] = "mutated result"
    assert desired[0]["name"] == "ours"


@pytest.mark.parametrize(
    ("desired", "current"),
    [
        (
            [{"id": "aaaaaaaaaaaa", "name": "same update"}],
            [{"id": "aaaaaaaaaaaa", "name": "same update"}],
        ),
        ([], []),
    ],
)
def test_three_way_merge_accepts_identical_concurrent_change(desired, current):
    base = [{"id": "aaaaaaaaaaaa", "name": "base"}]

    assert jobs._merge_jobs_three_way(base, desired, current) == desired


def test_three_way_merge_fast_path_preserves_current_order():
    first = {"id": "aaaaaaaaaaaa", "name": "first"}
    second = {"id": "bbbbbbbbbbbb", "name": "second"}
    base = [first, second]
    current = [second, first]

    assert jobs._merge_jobs_three_way(base, copy.deepcopy(base), current) == current


def test_three_way_merge_preserves_current_reorder_with_desired_update():
    first = {"id": "aaaaaaaaaaaa", "name": "first"}
    second = {"id": "bbbbbbbbbbbb", "name": "second"}
    desired_first = {**first, "name": "first updated"}

    assert jobs._merge_jobs_three_way(
        [first, second],
        [desired_first, second],
        [second, first],
    ) == [second, desired_first]


def test_three_way_merge_preserves_sibling_create_position_with_desired_update():
    first = {"id": "aaaaaaaaaaaa", "name": "first"}
    second = {"id": "bbbbbbbbbbbb", "name": "second"}
    sibling = {"id": "cccccccccccc", "name": "sibling"}
    desired_first = {**first, "name": "first updated"}

    assert jobs._merge_jobs_three_way(
        [first, second],
        [desired_first, second],
        [first, sibling, second],
    ) == [desired_first, sibling, second]


def test_three_way_merge_fails_closed_on_conflicting_reorders():
    first = {"id": "aaaaaaaaaaaa", "name": "first"}
    second = {"id": "bbbbbbbbbbbb", "name": "second"}
    third = {"id": "cccccccccccc", "name": "third"}

    with pytest.raises(RuntimeError, match="ordering"):
        jobs._merge_jobs_three_way(
            [first, second, third],
            [second, first, third],
            [first, third, second],
        )


def test_recorded_merge_base_isolated_from_nested_caller_mutation():
    loaded = [
        {
            "id": "aaaaaaaaaaaa",
            "name": "base",
            "schedule": {"kind": "interval", "minutes": 5},
        }
    ]
    try:
        jobs._jobs_lock_state.depth = 1
        jobs._record_load_snapshot(loaded)
        schedule = loaded[0]["schedule"]
        assert isinstance(schedule, dict)
        schedule["minutes"] = 10

        assert jobs._jobs_lock_state.load_jobs[0]["schedule"]["minutes"] == 5
    finally:
        jobs._jobs_lock_state.depth = 0
        jobs._jobs_lock_state.load_jobs = None


def test_three_way_merge_allows_legacy_id_repair_with_sibling_create():
    legacy = {"name": "legacy", "enabled": False}
    repaired = {"id": "repaired0001", **legacy}
    sibling = {"id": "bbbbbbbbbbbb", "name": "sibling create"}

    merged = jobs._merge_jobs_three_way(
        [legacy],
        [repaired],
        [legacy, sibling],
    )

    assert merged == [repaired, sibling]


def test_three_way_merge_repairs_poisoned_duplicate_without_coalescing():
    first = {"id": "duplicate-id", "name": "first"}
    second = {"id": "duplicate-id", "name": "second"}
    repaired_second = {"id": "repaired0002", "name": "second"}
    sibling = {"id": "bbbbbbbbbbbb", "name": "sibling create"}

    merged = jobs._merge_jobs_three_way(
        [first, second],
        [first, repaired_second],
        [first, second, sibling],
    )

    assert merged == [first, repaired_second, sibling]


def test_three_way_merge_preserves_opaque_row_position_across_disjoint_updates():
    first = {"id": "aaaaaaaaaaaa", "name": "first"}
    legacy = {"name": "legacy", "enabled": False}
    second = {"id": "bbbbbbbbbbbb", "name": "second"}
    desired_first = {**first, "name": "first updated"}
    current_second = {**second, "name": "second updated"}

    assert jobs._merge_jobs_three_way(
        [first, legacy, second],
        [desired_first, legacy, second],
        [first, legacy, current_second],
    ) == [desired_first, legacy, current_second]


def test_three_way_merge_preserves_current_opaque_row_reorder():
    first = {"id": "aaaaaaaaaaaa", "name": "first"}
    legacy = {"name": "legacy", "enabled": False}
    second = {"id": "bbbbbbbbbbbb", "name": "second"}
    desired_first = {**first, "name": "first updated"}
    current_second = {**second, "name": "second updated"}

    assert jobs._merge_jobs_three_way(
        [first, legacy, second],
        [desired_first, legacy, second],
        [first, current_second, legacy],
    ) == [desired_first, current_second, legacy]


def test_three_way_merge_rejects_conflicting_opaque_row_reorders():
    first = {"id": "aaaaaaaaaaaa", "name": "first"}
    legacy = {"name": "legacy", "enabled": False}
    second = {"id": "bbbbbbbbbbbb", "name": "second"}

    with pytest.raises(RuntimeError, match="ordering"):
        jobs._merge_jobs_three_way(
            [first, legacy, second],
            [legacy, {**first, "name": "ours"}, second],
            [first, {**second, "name": "theirs"}, legacy],
        )


def test_three_way_merge_rejects_identified_insert_against_opaque_change():
    first = {"id": "aaaaaaaaaaaa", "name": "first"}
    legacy = {"name": "legacy", "enabled": False}
    second = {"id": "bbbbbbbbbbbb", "name": "second"}
    inserted = {"id": "cccccccccccc", "name": "inserted"}
    changed_legacy = {**legacy, "enabled": True}

    with pytest.raises(RuntimeError, match="unidentifiable-record changes"):
        jobs._merge_jobs_three_way(
            [first, legacy, second],
            [first, inserted, legacy, second],
            [first, changed_legacy, second],
        )


def test_three_way_merge_rejects_two_concurrent_mid_list_inserts():
    first = {"id": "aaaaaaaaaaaa", "name": "first"}
    second = {"id": "bbbbbbbbbbbb", "name": "second"}
    ours = {"id": "cccccccccccc", "name": "ours"}
    theirs = {"id": "dddddddddddd", "name": "theirs"}

    with pytest.raises(RuntimeError, match="ordering"):
        jobs._merge_jobs_three_way(
            [first, second],
            [first, ours, second],
            [first, theirs, second],
        )


def test_three_way_merge_rejects_reorder_against_mid_list_insert():
    first = {"id": "aaaaaaaaaaaa", "name": "first"}
    second = {"id": "bbbbbbbbbbbb", "name": "second"}
    third = {"id": "cccccccccccc", "name": "third"}
    inserted = {"id": "dddddddddddd", "name": "inserted"}

    with pytest.raises(RuntimeError, match="ordering"):
        jobs._merge_jobs_three_way(
            [first, second, third],
            [second, first, third],
            [first, inserted, second, third],
        )


def test_three_way_merge_keeps_mid_insert_with_sibling_suffix_create():
    first = {"id": "aaaaaaaaaaaa", "name": "first"}
    second = {"id": "bbbbbbbbbbbb", "name": "second"}
    ours = {"id": "cccccccccccc", "name": "ours"}
    theirs = {"id": "dddddddddddd", "name": "theirs"}

    assert jobs._merge_jobs_three_way(
        [first, second],
        [first, ours, second],
        [first, second, theirs],
    ) == [first, ours, second, theirs]


def test_three_way_merge_overlays_sibling_delete_on_mid_insert():
    first = {"id": "aaaaaaaaaaaa", "name": "first"}
    second = {"id": "bbbbbbbbbbbb", "name": "second"}
    inserted = {"id": "cccccccccccc", "name": "inserted"}

    assert jobs._merge_jobs_three_way(
        [first, second],
        [first, inserted, second],
        [first],
    ) == [first, inserted]


@pytest.mark.parametrize(
    "sibling_legacy",
    [
        {"name": "sibling legacy", "enabled": False},
        "sibling non-object legacy row",
    ],
)
def test_three_way_merge_preserves_sibling_added_unidentifiable_record(
    sibling_legacy,
):
    base = [{"id": "aaaaaaaaaaaa", "name": "base"}]
    desired = [{"id": "aaaaaaaaaaaa", "name": "ours"}]

    merged = jobs._merge_jobs_three_way(
        base,
        desired,
        [base[0], sibling_legacy],
    )

    assert merged == [desired[0], sibling_legacy]


def test_three_way_merge_fails_closed_on_divergent_unidentifiable_changes():
    base = [{"name": "legacy", "enabled": False}]
    desired = [{"id": "repaired0001", "name": "legacy", "enabled": False}]
    current = [{"name": "legacy changed", "enabled": False}]

    with pytest.raises(RuntimeError, match="unidentifiable cron job"):
        jobs._merge_jobs_three_way(base, desired, current)


def test_three_way_merge_fails_closed_on_divergent_non_object_changes():
    with pytest.raises(RuntimeError, match="unidentifiable cron job"):
        jobs._merge_jobs_three_way(
            cast(Any, ["legacy non-object row"]),
            cast(Any, ["our replacement row"]),
            cast(Any, ["their replacement row"]),
        )


@pytest.mark.parametrize(
    "current",
    [
        [{"id": "repair-id-z", "name": "legacy", "enabled": False}],
        [],
    ],
)
def test_three_way_merge_rejects_concurrent_repair_or_delete_of_opaque_base(
    current,
):
    base = [{"name": "legacy", "enabled": False}]
    desired = [{"id": "repair-id-y", "name": "legacy", "enabled": False}]

    with pytest.raises(RuntimeError, match="unidentifiable cron job"):
        jobs._merge_jobs_three_way(base, desired, current)


def test_three_way_merge_rejects_two_distinct_duplicate_id_repairs():
    first = {"id": "duplicate-id", "name": "first"}
    second = {"id": "duplicate-id", "name": "second"}

    with pytest.raises(RuntimeError, match="unidentifiable cron job"):
        jobs._merge_jobs_three_way(
            [first, second],
            [first, {**second, "id": "repair-id-y"}],
            [first, {**second, "id": "repair-id-z"}],
        )


@pytest.mark.parametrize(
    ("desired", "current"),
    [
        (
            [{"id": "aaaaaaaaaaaa", "name": "ours"}],
            [{"id": "aaaaaaaaaaaa", "name": "theirs"}],
        ),
        ([], [{"id": "aaaaaaaaaaaa", "name": "theirs"}]),
        ([{"id": "aaaaaaaaaaaa", "name": "ours"}], []),
    ],
)
def test_three_way_merge_fails_closed_on_same_id_conflict(desired, current):
    base = [{"id": "aaaaaaaaaaaa", "name": "base"}]

    with pytest.raises(RuntimeError, match="conflicting concurrent cron job change"):
        jobs._merge_jobs_three_way(base, desired, current)


def test_three_way_merge_rejects_conflicting_same_id_create():
    with pytest.raises(RuntimeError, match="conflicting concurrent cron job change"):
        jobs._merge_jobs_three_way(
            [],
            [{"id": "aaaaaaaaaaaa", "name": "ours"}],
            [{"id": "aaaaaaaaaaaa", "name": "theirs"}],
        )


def test_degraded_lock_recovers_concurrently_created_job(tmp_path, monkeypatch):
    """#80624: a job a sibling process wrote during a degraded (unlocked)
    window must not be silently discarded by this process's own save.

    ``_jobs_lock()`` intentionally falls through to in-process-only locking
    when the cross-process flock times out or is unavailable (#60703) — that
    is a deliberate liveness tradeoff, not a bug. But before this fix, a save
    made during that window would blindly overwrite jobs.json with whatever
    stale list this process last loaded, discarding any job a sibling process
    (e.g. the CLI) wrote in between. Reproduces that exact race in-process by
    forcing ``_jobs_lock_state`` into the degraded state _jobs_lock() would
    have left it in, without depending on OS-specific flock timing.
    """
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    job_a = {"id": "aaaaaaaaaaaa", "name": "existing"}
    job_b = {"id": "bbbbbbbbbbbb", "name": "cli-created"}

    try:
        jobs.save_jobs([job_a])

        # This process enters a degraded critical section (as _jobs_lock()
        # does after a flock timeout) and loads the current, job_b-less state.
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        assert jobs.load_jobs() == [job_a]
        stale_base = copy.deepcopy(jobs._jobs_lock_state.load_jobs)

        # A sibling process (e.g. the CLI) creates job_b concurrently, via its
        # own independent, fully-scoped _jobs_lock() cycle.
        jobs._jobs_lock_state.depth = 0
        jobs.save_jobs([job_a, job_b])

        # This process resumes its degraded section with the stale view it
        # actually observed (no job_b) and saves — pre-fix, this silently
        # wiped job_b out of jobs.json.
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        jobs._jobs_lock_state.load_jobs = stale_base
        jobs._save_jobs_unlocked([job_a])
    finally:
        jobs._jobs_lock_state.depth = 0
        jobs._jobs_lock_state.degraded = False
        jobs._jobs_lock_state.load_jobs = None

    on_disk_ids = {j["id"] for j in jobs.load_jobs()}
    assert on_disk_ids == {job_a["id"], job_b["id"]}, (
        "sibling-created job was clobbered by a degraded-lock write (#80624)"
    )


def test_healthy_lock_write_recovers_sibling_degraded_create(tmp_path, monkeypatch):
    """#80624 reverse direction: a *healthy* lock holder can still clobber a
    sibling's degraded write if that sibling raced in and out while this
    process's own critical section was open. flock only excludes other
    processes that also hold it — it does nothing to stop a process that
    gave up waiting for it. The reconcile check must not be gated on this
    process's own ``degraded`` flag, or this direction of the race reopens
    the exact #80624 symptom.
    """
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    job_a = {"id": "aaaaaaaaaaaa", "name": "existing"}
    job_b = {"id": "bbbbbbbbbbbb", "name": "cli-created"}

    try:
        jobs.save_jobs([job_a])

        # This process opens a *healthy* critical section (real flock held)
        # and loads the current, job_b-less state.
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = False
        assert jobs.load_jobs() == [job_a]
        stale_base = copy.deepcopy(jobs._jobs_lock_state.load_jobs)

        # A sibling process couldn't get the flock, degraded, and created
        # job_b anyway via its own independent _jobs_lock() cycle. (This
        # nested call shares the same thread-local as the outer section only
        # because the test simulates two processes in one thread — real
        # processes each have their own _jobs_lock_state, so this reset
        # doesn't happen in production; restored below to keep the test
        # faithful to the real per-process state.)
        jobs._jobs_lock_state.depth = 0
        jobs.save_jobs([job_a, job_b])

        # This process resumes its still-healthy section with the stale view
        # it actually observed (no job_b) and saves.
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = False
        jobs._jobs_lock_state.load_jobs = stale_base
        jobs._save_jobs_unlocked([job_a])
    finally:
        jobs._jobs_lock_state.depth = 0
        jobs._jobs_lock_state.degraded = False
        jobs._jobs_lock_state.load_jobs = None

    on_disk_ids = {j["id"] for j in jobs.load_jobs()}
    assert on_disk_ids == {job_a["id"], job_b["id"]}, (
        "sibling's degraded create was clobbered by a healthy-lock write (#80624)"
    )


def test_two_saves_in_one_lock_keep_sibling_create_in_caller_provenance(
    tmp_path, monkeypatch
):
    """A recovered C-only create survives a second save from unchanged caller D."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    job_a = {"id": "aaaaaaaaaaaa", "name": "base"}
    job_b = {"id": "bbbbbbbbbbbb", "name": "sibling-created"}

    try:
        jobs.save_jobs([job_a])
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        caller_jobs = jobs.load_jobs()
        caller_base = copy.deepcopy(jobs._jobs_lock_state.load_jobs)

        jobs._jobs_lock_state.depth = 0
        jobs.save_jobs([job_a, job_b])

        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        jobs._jobs_lock_state.load_jobs = caller_base
        caller_jobs[0]["name"] = "first update"
        jobs._save_jobs_unlocked(caller_jobs)
        assert [job["id"] for job in caller_jobs] == [job_a["id"]]

        caller_jobs[0]["name"] = "second update"
        jobs._save_jobs_unlocked(caller_jobs)
    finally:
        jobs._jobs_lock_state.depth = 0
        jobs._jobs_lock_state.degraded = False
        jobs._jobs_lock_state.load_jobs = None

    on_disk = {job["id"]: job for job in jobs.load_jobs()}
    assert set(on_disk) == {job_a["id"], job_b["id"]}
    assert on_disk[job_a["id"]]["name"] == "second update"


def test_two_saves_in_one_lock_keep_sibling_delete_in_caller_provenance(
    tmp_path, monkeypatch
):
    """A C-only deletion is not resurrected by the caller's second save."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    job_a = {"id": "aaaaaaaaaaaa", "name": "base"}
    job_b = {"id": "bbbbbbbbbbbb", "name": "sibling-deleted"}

    try:
        jobs.save_jobs([job_a, job_b])
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        caller_jobs = jobs.load_jobs()
        caller_base = copy.deepcopy(jobs._jobs_lock_state.load_jobs)

        jobs._jobs_lock_state.depth = 0
        jobs.save_jobs([job_a])

        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        jobs._jobs_lock_state.load_jobs = caller_base
        caller_jobs[0]["name"] = "first update"
        jobs._save_jobs_unlocked(caller_jobs)

        caller_jobs[0]["name"] = "second update"
        jobs._save_jobs_unlocked(caller_jobs)
    finally:
        jobs._jobs_lock_state.depth = 0
        jobs._jobs_lock_state.degraded = False
        jobs._jobs_lock_state.load_jobs = None

    on_disk = jobs.load_jobs()
    assert [job["id"] for job in on_disk] == [job_a["id"]]
    assert on_disk[0]["name"] == "second update"


def test_loaded_base_matches_data_read_from_open_descriptor(tmp_path, monkeypatch):
    """A replacement after decoding must not pair old data with the new path stamp.

    Atomic replace leaves an already-open descriptor on the old inode.  The load
    snapshot therefore has to derive both its JSON and identity from that same
    descriptor; a later stat of the path can already describe a sibling's file.
    """
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    job_a = {"id": "aaaaaaaaaaaa", "name": "existing"}
    job_b = {"id": "bbbbbbbbbbbb", "name": "cli-created"}
    jobs.save_jobs([job_a])

    replacement = cron_dir / ".sibling-replacement"
    replacement.write_text(
        jobs.json.dumps({"jobs": [job_a, job_b], "updated_at": "sibling"}),
        encoding="utf-8",
    )
    real_loads = jobs.json.loads
    replaced = False

    def replace_path_after_decode(payload, *args, **kwargs):
        nonlocal replaced
        data = real_loads(payload, *args, **kwargs)
        if not replaced:
            replaced = True
            os.replace(replacement, jobs.JOBS_FILE)
        return data

    monkeypatch.setattr(jobs.json, "loads", replace_path_after_decode)

    try:
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        loaded = jobs.load_jobs()
        assert loaded == [job_a]
        jobs._save_jobs_unlocked([{**job_a, "name": "updated"}])
    finally:
        jobs._jobs_lock_state.depth = 0
        jobs._jobs_lock_state.degraded = False
        jobs._jobs_lock_state.load_jobs = None

    on_disk = {job["id"]: job for job in jobs.load_jobs()}
    assert set(on_disk) == {job_a["id"], job_b["id"]}
    assert on_disk[job_a["id"]]["name"] == "updated"


def test_save_rechecks_generation_after_staging_without_resurrecting_delete(
    tmp_path, monkeypatch
):
    """A sibling create after reconcile survives; an intentional delete does not.

    The replacement is injected from ``mkstemp`` so it lands after the first
    reconciliation and before publication.  Retrying must recompute from the
    caller's original desired list, otherwise a previously recovered or deleted
    job can be resurrected across attempts.
    """
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    job_a = {"id": "aaaaaaaaaaaa", "name": "existing"}
    job_b = {"id": "bbbbbbbbbbbb", "name": "delete-me"}
    job_c = {"id": "cccccccccccc", "name": "sibling-created"}
    jobs.save_jobs([job_a, job_b])

    replacement = cron_dir / ".sibling-replacement"
    replacement.write_text(
        jobs.json.dumps({"jobs": [job_a, job_b, job_c], "updated_at": "sibling"}),
        encoding="utf-8",
    )
    real_mkstemp = jobs.tempfile.mkstemp
    replaced = False

    def replace_path_while_staging(*args, **kwargs):
        nonlocal replaced
        staged = real_mkstemp(*args, **kwargs)
        if kwargs.get("prefix") == ".jobs_" and not replaced:
            replaced = True
            os.replace(replacement, jobs.JOBS_FILE)
        return staged

    monkeypatch.setattr(jobs.tempfile, "mkstemp", replace_path_while_staging)

    try:
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        assert jobs.load_jobs() == [job_a, job_b]
        jobs._save_jobs_unlocked([{**job_a, "name": "updated"}])
    finally:
        jobs._jobs_lock_state.depth = 0
        jobs._jobs_lock_state.degraded = False
        jobs._jobs_lock_state.load_jobs = None

    on_disk = {job["id"]: job for job in jobs.load_jobs()}
    assert set(on_disk) == {job_a["id"], job_c["id"]}
    assert on_disk[job_a["id"]]["name"] == "updated"


def test_save_fails_closed_when_manual_writer_keeps_replacing(tmp_path, monkeypatch):
    """Bounded retries leave the last external generation intact on exhaustion."""
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    job_a = {"id": "aaaaaaaaaaaa", "name": "existing"}
    jobs.save_jobs([job_a])
    real_mkstemp = jobs.tempfile.mkstemp
    generation = 0

    def replace_path_on_every_stage(*args, **kwargs):
        nonlocal generation
        staged = real_mkstemp(*args, **kwargs)
        if kwargs.get("prefix") == ".jobs_":
            generation += 1
            sibling = {
                "id": f"{generation:012d}",
                "name": f"manual-generation-{generation}",
            }
            replacement = cron_dir / f".manual-replacement-{generation}"
            replacement.write_text(
                jobs.json.dumps(
                    {"jobs": [job_a, sibling], "updated_at": "manual-writer"}
                ),
                encoding="utf-8",
            )
            os.replace(replacement, jobs.JOBS_FILE)
        return staged

    monkeypatch.setattr(jobs.tempfile, "mkstemp", replace_path_on_every_stage)

    try:
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        assert jobs.load_jobs() == [job_a]
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            jobs._save_jobs_unlocked([{**job_a, "name": "stale-writer"}])
    finally:
        jobs._jobs_lock_state.depth = 0
        jobs._jobs_lock_state.degraded = False
        jobs._jobs_lock_state.load_jobs = None

    assert generation == jobs._JOBS_GENERATION_MAX_ATTEMPTS
    on_disk = {job["id"]: job for job in jobs.load_jobs()}
    assert set(on_disk) == {job_a["id"], f"{generation:012d}"}
    assert on_disk[job_a["id"]]["name"] == "existing"
    assert list(cron_dir.glob(".jobs_*.tmp")) == []


def test_get_due_jobs_repairs_idless_base_while_preserving_sibling_create(
    tmp_path, monkeypatch
):
    """A real due scan must not abort when C retains the legacy B row."""
    from datetime import datetime, timedelta, timezone

    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    now = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    legacy = {
        "name": "legacy idless",
        "schedule": {},
        "enabled": False,
    }
    healthy = {
        "id": "aaaaaaaaaaaa",
        "name": "healthy due",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": (now - timedelta(seconds=30)).isoformat(),
        "enabled": True,
    }
    sibling = {
        "id": "bbbbbbbbbbbb",
        "name": "sibling create",
        "schedule": {},
        "enabled": False,
    }
    jobs.save_jobs([legacy, healthy])

    replacement = cron_dir / ".sibling-replacement"
    replacement.write_text(
        jobs.json.dumps(
            {"jobs": [legacy, healthy, sibling], "updated_at": "sibling"}
        ),
        encoding="utf-8",
    )
    replaced = False

    class FixedUUID:
        hex = "repaired0001"

    def replace_when_due_scan_repairs_id():
        nonlocal replaced
        if not replaced:
            replaced = True
            os.replace(replacement, jobs.JOBS_FILE)
        return FixedUUID()

    monkeypatch.setattr(jobs.uuid, "uuid4", replace_when_due_scan_repairs_id)

    due = jobs.get_due_jobs()

    assert [job["id"] for job in due] == [healthy["id"]]
    on_disk = {job["id"]: job for job in jobs.load_jobs()}
    assert set(on_disk) == {"repaired0001", healthy["id"], sibling["id"]}


def test_get_due_jobs_repairs_duplicate_base_while_preserving_sibling_create(
    tmp_path, monkeypatch
):
    """Duplicate-ID rows stay distinct when a sibling generation arrives."""
    from datetime import datetime, timedelta, timezone

    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    now = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    first = {
        "id": "duplicate-id",
        "name": "first duplicate",
        "schedule": {},
        "enabled": False,
    }
    second = {
        "id": "duplicate-id",
        "name": "second duplicate",
        "schedule": {},
        "enabled": False,
    }
    healthy = {
        "id": "aaaaaaaaaaaa",
        "name": "healthy due",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": (now - timedelta(seconds=30)).isoformat(),
        "enabled": True,
    }
    sibling = {
        "id": "bbbbbbbbbbbb",
        "name": "sibling create",
        "schedule": {},
        "enabled": False,
    }
    jobs.save_jobs([first, second, healthy])

    replacement = cron_dir / ".sibling-replacement"
    replacement.write_text(
        jobs.json.dumps(
            {
                "jobs": [first, second, healthy, sibling],
                "updated_at": "sibling",
            }
        ),
        encoding="utf-8",
    )
    replaced = False

    class FixedUUID:
        hex = "repaired0002"

    def replace_when_due_scan_repairs_duplicate():
        nonlocal replaced
        if not replaced:
            replaced = True
            os.replace(replacement, jobs.JOBS_FILE)
        return FixedUUID()

    monkeypatch.setattr(jobs.uuid, "uuid4", replace_when_due_scan_repairs_duplicate)

    due = jobs.get_due_jobs()

    assert replaced is True
    assert [job["id"] for job in due] == [healthy["id"]]
    on_disk = {job["id"]: job for job in jobs.load_jobs()}
    assert set(on_disk) == {
        "duplicate-id",
        "repaired0002",
        healthy["id"],
        sibling["id"],
    }
    assert on_disk["duplicate-id"]["name"] == "first duplicate"
    assert on_disk["repaired0002"]["name"] == "second duplicate"


@pytest.mark.parametrize(
    "opaque_row",
    ["legacy scalar row", ["legacy", "list", "row"]],
)
def test_get_due_jobs_preserves_non_object_row_while_processing_healthy_job(
    tmp_path, monkeypatch, opaque_row
):
    """Opaque legacy rows remain stored but cannot starve a healthy due job."""
    from datetime import datetime, timedelta, timezone

    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    now = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    healthy = {
        "id": "aaaaaaaaaaaa",
        "name": "healthy due",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": (now - timedelta(minutes=10)).isoformat(),
        "enabled": True,
    }
    jobs.save_jobs(cast(Any, [opaque_row, healthy]))

    due = jobs.get_due_jobs()

    assert [job["id"] for job in due] == [healthy["id"]]
    on_disk = jobs.load_jobs()
    assert on_disk[0] == opaque_row
    assert on_disk[1]["id"] == healthy["id"]
    assert jobs._ensure_aware(
        jobs.datetime.fromisoformat(on_disk[1]["next_run_at"])
    ) > jobs._hermes_now()


def test_get_due_jobs_bounds_uuid_collisions_during_identity_repair(
    tmp_path, monkeypatch
):
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")
    monkeypatch.setattr(jobs, "_JOB_ID_REPAIR_MAX_ATTEMPTS", 3)

    existing = {
        "id": "aaaaaaaaaaaa",
        "name": "existing",
        "schedule": {},
        "enabled": False,
    }
    legacy = {"name": "legacy idless", "schedule": {}, "enabled": False}
    jobs.save_jobs([existing, legacy])
    calls = 0

    class CollidingUUID:
        hex = existing["id"]

    def collide_with_reserved_id():
        nonlocal calls
        calls += 1
        return CollidingUUID()

    monkeypatch.setattr(jobs.uuid, "uuid4", collide_with_reserved_id)

    with pytest.raises(RuntimeError, match="Could not generate a unique ID"):
        jobs.get_due_jobs()

    assert calls == 3
    assert jobs.load_jobs() == [existing, legacy]


@pytest.mark.parametrize("sibling_change", ["pause", "delete", "edit", "duplicate"])
def test_get_due_jobs_revalidates_no_save_degraded_sibling_change(
    tmp_path, monkeypatch, sibling_change
):
    """A due decision from B cannot outlive a same-record change in C."""
    from datetime import datetime, timedelta, timezone

    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    now = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    due_job = {
        "id": "aaaaaaaaaaaa",
        "name": "due",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": (now - timedelta(seconds=30)).isoformat(),
        "enabled": True,
    }
    jobs.save_jobs([due_job])

    if sibling_change == "pause":
        sibling_jobs = [{**due_job, "enabled": False}]
    elif sibling_change == "delete":
        sibling_jobs = []
    elif sibling_change == "edit":
        sibling_jobs = [{**due_job, "name": "sibling edit"}]
    else:
        sibling_jobs = [due_job, {**due_job, "name": "duplicate"}]
    replacement = cron_dir / ".sibling-due-replacement"
    replacement.write_text(
        jobs.json.dumps({"jobs": sibling_jobs, "updated_at": "sibling"}),
        encoding="utf-8",
    )

    original_apply_skill_fields = jobs._apply_skill_fields
    replaced = False

    def replace_after_degraded_load(job):
        nonlocal replaced
        if not replaced:
            assert jobs._jobs_lock_state.degraded is True
            os.replace(replacement, jobs.JOBS_FILE)
            replaced = True
        return original_apply_skill_fields(job)

    monkeypatch.setattr(jobs, "_apply_skill_fields", replace_after_degraded_load)
    try:
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        assert jobs._get_due_jobs_locked() == []
    finally:
        jobs._jobs_lock_state.depth = 0
        jobs._jobs_lock_state.degraded = False
        jobs._jobs_lock_state.load_jobs = None

    assert replaced is True
    assert jobs.load_jobs() == sibling_jobs


def test_get_due_jobs_revalidates_after_save_merges_sibling_pause(
    tmp_path, monkeypatch
):
    """A disjoint repair save may preserve C on disk but must also drop stale due."""
    from datetime import datetime, timedelta, timezone

    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    now = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    due_job = {
        "id": "aaaaaaaaaaaa",
        "name": "due",
        "schedule": {"kind": "interval", "minutes": 5},
        "next_run_at": (now - timedelta(seconds=30)).isoformat(),
        "enabled": True,
    }
    malformed = {
        "id": "bbbbbbbbbbbb",
        "name": "repair me",
        "schedule": None,
        "enabled": False,
    }
    jobs.save_jobs([due_job, malformed])

    paused = {**due_job, "enabled": False}
    replacement = cron_dir / ".sibling-save-replacement"
    replacement.write_text(
        jobs.json.dumps(
            {"jobs": [paused, malformed], "updated_at": "sibling"}
        ),
        encoding="utf-8",
    )
    original_apply_skill_fields = jobs._apply_skill_fields
    replaced = False

    def replace_after_degraded_load(job):
        nonlocal replaced
        if not replaced:
            assert jobs._jobs_lock_state.degraded is True
            os.replace(replacement, jobs.JOBS_FILE)
            replaced = True
        return original_apply_skill_fields(job)

    monkeypatch.setattr(jobs, "_apply_skill_fields", replace_after_degraded_load)
    try:
        jobs._jobs_lock_state.depth = 1
        jobs._jobs_lock_state.degraded = True
        assert jobs._get_due_jobs_locked() == []
    finally:
        jobs._jobs_lock_state.depth = 0
        jobs._jobs_lock_state.degraded = False
        jobs._jobs_lock_state.load_jobs = None

    persisted = {job["id"]: job for job in jobs.load_jobs()}
    assert replaced is True
    assert persisted[due_job["id"]]["enabled"] is False
    assert persisted[malformed["id"]]["schedule"] == {}


def test_due_oneshot_reuses_published_save_snapshot_without_second_commit_lock(
    tmp_path, monkeypatch
):
    """A persisted run_claim must not be stranded by a second lock/read failure."""
    from datetime import datetime, timezone

    cron_dir = tmp_path / "cron"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", cron_dir / "output")

    now = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    oneshot = {
        "id": "aaaaaaaaaaaa",
        "name": "due once",
        "schedule": {"kind": "once", "run_at": now.isoformat()},
        "next_run_at": now.isoformat(),
        "enabled": True,
    }
    jobs.save_jobs([oneshot])

    original_commit_lock = jobs._jobs_commit_lock
    acquisitions = 0

    @contextlib.contextmanager
    def fail_hypothetical_second_commit_lock():
        nonlocal acquisitions
        acquisitions += 1
        if acquisitions > 1:
            raise RuntimeError("hypothetical second commit-lock failure")
        with original_commit_lock() as validate_commit_lock:
            yield validate_commit_lock

    monkeypatch.setattr(
        jobs, "_jobs_commit_lock", fail_hypothetical_second_commit_lock
    )

    due = jobs.get_due_jobs()

    assert acquisitions == 1
    assert [job["id"] for job in due] == [oneshot["id"]]
    persisted = jobs.load_jobs()[0]
    assert persisted["run_claim"] == due[0]["run_claim"]
