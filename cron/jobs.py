"""
Cron job storage and management.

Jobs are stored in ~/.hermes/cron/jobs.json
Output is saved to ~/.hermes/cron/output/{job_id}/{timestamp}.md
"""

import contextlib
import copy
from contextvars import ContextVar
from dataclasses import dataclass
import json
import logging
import shutil
import stat
import tempfile
import threading
import time
import os
import re
import uuid

# Cross-process advisory file locking for jobs.json critical sections.
# fcntl is Unix-only; on Windows fall back to msvcrt. Either may be absent,
# in which case _jobs_lock() degrades to in-process locking only (the old
# behaviour) rather than failing.
try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None
from datetime import datetime, timedelta
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Optional, Dict, List, Any, Set, Tuple, Union

logger = logging.getLogger(__name__)

from hermes_time import now as _hermes_now
from utils import atomic_replace, atomic_write_text

# ``croniter`` compiles ~15 ms of regexes at import and only matters for
# 5-field cron expressions. Resolve lazily; ``HAS_CRONITER`` stays a module
# attribute (tests monkeypatch it, and a monkeypatched value wins because
# ``_ensure_croniter`` only probes while it's still None).
croniter = None
HAS_CRONITER: Optional[bool] = None


def _ensure_croniter() -> bool:
    """Import croniter on first use; honor a pre-set HAS_CRONITER override."""
    global croniter, HAS_CRONITER
    if HAS_CRONITER is None:
        try:
            from croniter import croniter as _croniter
            croniter = _croniter
            HAS_CRONITER = True
        except ImportError:
            HAS_CRONITER = False
    return bool(HAS_CRONITER)

# =============================================================================
# Configuration
# =============================================================================

# Cron is per-profile by design (issue #4707). Each profile owns its own cron
# store under its own HERMES_HOME, and a profile-scoped gateway runs that
# profile's jobs under that same HERMES_HOME — so a job authored in profile
# `coder` lives in `~/.hermes/profiles/coder/cron/jobs.json` and executes with
# `coder`'s `.env`, `config.yaml`, and skills. We deliberately anchor on
# `get_hermes_home()` (the active profile home), NOT `get_default_hermes_root()`
# (the shared root). Anchoring at the root would funnel every profile's jobs
# into one shared `jobs.json` and run them under whatever HERMES_HOME the
# ticker process happens to have — leaking config/credentials/skills across
# profiles (the security boundary #4707 was filed for). Do NOT change this to
# the default root: that re-breaks per-profile isolation. See also the dynamic
# `_get_hermes_home()` / `_get_lock_paths()` resolution in cron/scheduler.py.
HERMES_DIR = get_hermes_home().resolve()
# These constants remain the default-profile fallback and a compatibility
# surface for existing callers/tests. Cross-profile callers must scope paths
# with use_cron_store() instead of mutating them process-wide.
CRON_DIR = HERMES_DIR / "cron"
JOBS_FILE = CRON_DIR / "jobs.json"
# Heartbeat file the in-process ticker touches on every loop iteration. The
# gateway process and the (separate) ``hermes cron status`` process share it
# so status can tell whether the ticker THREAD is alive, not just whether the
# gateway PROCESS exists — a ticker that dies silently inside a live gateway
# would otherwise report healthy (#32612, #32895).
TICKER_HEARTBEAT_FILE = CRON_DIR / "ticker_heartbeat"
# Last tick that completed WITHOUT raising. Distinguishing this from the plain
# heartbeat lets status detect a ticker that is alive but failing every tick.
TICKER_SUCCESS_FILE = CRON_DIR / "ticker_last_success"
# Default ticker loop interval (seconds). The single source of truth shared by
# the in-process ticker (cron/scheduler_provider.py) and the staleness
# threshold in `hermes cron status` (hermes_cli/cron.py), so the two never
# drift apart.
TICKER_INTERVAL_SECONDS = 60

# In-process lock protecting load_jobs→modify→save_jobs cycles.
# Required when tick() runs jobs in parallel threads — without this,
# concurrent mark_job_run / advance_next_run calls can clobber each other.
_jobs_file_lock = threading.RLock()
_jobs_lock_state = threading.local()

# Upper bound on waiting for the cross-process .jobs.lock flock (#60703).
# Every cron function in the process funnels through _jobs_lock(), and the
# flock is taken while holding the process-wide RLock — so an unbounded wait
# on a lock held by a wedged sibling process silently freezes the ticker
# heartbeat and every job forever.  30s is orders of magnitude above any
# legitimate critical section (field updates only) while keeping the ticker's
# worst-case stall well under one status-alarm threshold.
_JOBS_LOCK_TIMEOUT_SECONDS = 30.0
# The final read/reconcile/replace section is much shorter than a full cron
# mutation.  Every writer takes this second lock even when the broader lock
# degraded, so two current Hermes processes cannot race between reconciliation
# and publication.  Failure is closed here: publishing a known-stale snapshot
# is worse than making the caller retry a save.
_JOBS_COMMIT_LOCK_TIMEOUT_SECONDS = 5.0
_JOBS_GENERATION_MAX_ATTEMPTS = 3
_JOB_ID_REPAIR_MAX_ATTEMPTS = 16
OUTPUT_DIR = CRON_DIR / "output"
ONESHOT_GRACE_SECONDS = 120


@dataclass(frozen=True)
class _CronStorePaths:
    cron_dir: Path
    jobs_file: Path
    output_dir: Path


_JobsFileStamp = Tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _JobsFileSnapshot:
    exists: bool
    data: Any
    stamp: Optional[_JobsFileStamp]
    strict_retry: bool = False


_cron_store_override: ContextVar[Optional[_CronStorePaths]] = ContextVar(
    "cron_store_override",
    default=None,
)


# Import-time snapshot of the compatibility constants, so deliberate
# re-pointing of the module surface (monkeypatched CRON_DIR/JOBS_FILE/
# OUTPUT_DIR — the documented escape hatch existing tests/embedders use)
# is distinguishable from the constants merely being stale.
_IMPORT_STORE = _CronStorePaths(CRON_DIR, JOBS_FILE, OUTPUT_DIR)


def _current_cron_store() -> _CronStorePaths:
    """Return paths pinned to this execution context's profile.

    Precedence, most explicit first:

    1. an active use_cron_store() override (ContextVar);
    2. deliberately re-pointed module constants — if CRON_DIR/JOBS_FILE/
       OUTPUT_DIR no longer match their import-time values, someone chose
       the documented process-wide compatibility surface; honor it;
    3. the ACTIVE profile home, resolved fresh via get_hermes_home()
       (context-local override, then the HERMES_HOME env var) — so a test
       or embedder that re-points HERMES_HOME after this module was
       imported reads/writes ITS OWN store, not whatever jobs.json the
       import happened to freeze (the filed incident: fixtures that patched
       the env too late silently rewrote the user's real jobs file);
    4. the import-time constants (home unchanged since import — the common
       path, returned unchanged).
    """
    override = _cron_store_override.get()
    if override is not None:
        return override
    live_constants = _CronStorePaths(CRON_DIR, JOBS_FILE, OUTPUT_DIR)
    if live_constants != _IMPORT_STORE:
        return live_constants
    home = get_hermes_home().resolve()
    if home == HERMES_DIR:
        return live_constants
    cron_dir = home / "cron"
    return _CronStorePaths(cron_dir, cron_dir / "jobs.json", cron_dir / "output")


@contextlib.contextmanager
def use_cron_store(home: Union[str, Path]):
    """Route cron storage to ``home`` without mutating process globals."""
    cron_dir = Path(home).expanduser().resolve() / "cron"
    token = _cron_store_override.set(
        _CronStorePaths(
            cron_dir=cron_dir,
            jobs_file=cron_dir / "jobs.json",
            output_dir=cron_dir / "output",
        )
    )
    try:
        yield
    finally:
        _cron_store_override.reset(token)


def get_cron_output_dir() -> Path:
    """Return the output directory for the active cron store context."""
    return _current_cron_store().output_dir


# Fallback stale-recovery window for a one-shot's running-claim (#59229) when
# the cron inactivity timeout is disabled (HERMES_CRON_TIMEOUT=0 → unlimited),
# in which case no finite run bound exists to derive from. Also acts as the
# floor for the derived value so a very short configured timeout can't make the
# claim expire mid-run.
ONESHOT_RUN_CLAIM_TTL_SECONDS = 1800

# The derived TTL is the cron inactivity timeout times this headroom multiplier.
# A healthy run clears its claim via mark_job_run() long before the TTL; the
# TTL only recovers a claim left by a tick that DIED mid-run. HERMES_CRON_TIMEOUT
# is an *inactivity* limit, not a wall-clock cap — a job that keeps producing
# output legitimately runs past it — so the multiplier gives comfortable
# headroom over any healthy run before we treat a claim as stale.
_ONESHOT_RUN_CLAIM_TTL_HEADROOM = 3

_DEFAULT_CRON_INACTIVITY_TIMEOUT = 600.0


def _oneshot_run_claim_ttl_seconds() -> float:
    """Resolve the one-shot running-claim stale-recovery TTL.

    Derived from ``HERMES_CRON_TIMEOUT`` (the cron inactivity timeout the
    scheduler enforces on each run) so the safety valve tracks how long a run
    is actually allowed to go quiet, instead of a magic constant:

    - unset / invalid → default 600s inactivity limit → TTL = 1800s
    - ``0`` (unlimited runs) → no finite bound to derive from → fall back to
      ``ONESHOT_RUN_CLAIM_TTL_SECONDS``
    - positive N → ``max(N * headroom, ONESHOT_RUN_CLAIM_TTL_SECONDS)`` so a
      tiny configured timeout can never expire a claim mid-run.
    """
    raw = os.getenv("HERMES_CRON_TIMEOUT", "").strip()
    timeout = _DEFAULT_CRON_INACTIVITY_TIMEOUT
    if raw:
        try:
            timeout = float(raw)
        except (ValueError, TypeError):
            timeout = _DEFAULT_CRON_INACTIVITY_TIMEOUT
    if timeout <= 0:
        # Unlimited runs — cannot bound; use the fixed fallback floor.
        return float(ONESHOT_RUN_CLAIM_TTL_SECONDS)
    return max(
        timeout * _ONESHOT_RUN_CLAIM_TTL_HEADROOM,
        float(ONESHOT_RUN_CLAIM_TTL_SECONDS),
    )


def _job_running_in_this_process(job_id: str) -> bool:
    """Return True when the scheduler in THIS process is still running ``job_id``.

    Direct liveness signal for stale-entry recovery (#62002): the run_claim
    TTL alone cannot distinguish "the claiming tick died" from "the run is
    alive but slow" — a run stalled on network I/O (or a laptop that slept
    mid-run) legitimately outlives the TTL. The in-process ticker and the run
    share this process, so the scheduler's running set settles the common
    single-gateway case without any claim-age guesswork.

    Imported lazily: the scheduler imports this module at load, so a
    module-level import here would be circular.
    """
    try:
        from cron.scheduler import get_running_job_ids
        return job_id in get_running_job_ids()
    except Exception:
        logger.warning(
            "Cron running-set liveness check failed for job %r; keeping the "
            "entry to avoid deleting a possibly live one-shot run",
            job_id,
            exc_info=True,
        )
        return True


def _jobs_lock_file() -> Path:
    """Return the advisory lock path for the current cron directory."""
    return _current_cron_store().cron_dir / ".jobs.lock"


def _jobs_commit_lock_file() -> Path:
    """Return the short-held publication lock for the current cron store."""
    return _current_cron_store().cron_dir / ".jobs.commit.lock"


@contextlib.contextmanager
def _jobs_commit_lock():
    """Serialize the coherent re-read and atomic publication of jobs.json.

    The broader ``_jobs_lock`` deliberately fails open after its timeout so a
    wedged process cannot stop the scheduler forever (#60703).  That means a
    healthy writer and a degraded writer can both reach ``save_jobs``.  This
    separate lock is held only for the final read/reconcile/fsync/replace and
    is therefore safe to fail closed: if it cannot be acquired promptly, no
    potentially stale payload is published.
    """
    lock_path = _jobs_commit_lock_file()
    lock_fd = None
    locked = False
    acquisition_error = None
    try:
        try:
            parent_stat = os.stat(lock_path.parent)
            if os.name == "posix":
                nofollow = getattr(os, "O_NOFOLLOW", None)
                cloexec = getattr(os, "O_CLOEXEC", None)
                if nofollow is None or cloexec is None:
                    raise OSError(
                        "secure no-follow/close-on-exec open flags unavailable"
                    )
                raw_fd = os.open(
                    lock_path,
                    os.O_RDWR | os.O_CREAT | nofollow | cloexec,
                    0o600,
                )
                try:
                    lock_stat = os.fstat(raw_fd)
                    if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
                        raise OSError(
                            "cron jobs commit lock is not a singly-linked regular file"
                        )
                    # Operate only on the descriptor returned by the no-follow
                    # open. A path chmod/chown here can be redirected by a
                    # symlink or a rename between validation and mutation.
                    os.fchmod(raw_fd, 0o600)
                    _preserve_fd_ownership(
                        raw_fd, parent_stat, "cron jobs commit lock"
                    )
                    lock_fd = os.fdopen(raw_fd, "r+", encoding="utf-8")
                    raw_fd = -1
                finally:
                    if raw_fd >= 0:
                        os.close(raw_fd)
            else:
                lock_fd = open(lock_path, "a+", encoding="utf-8")
            if fcntl is None and msvcrt is None:
                raise OSError("no cross-process file locking backend available")
            if fcntl is None and msvcrt is not None:
                # msvcrt.locking() locks a byte range from the current offset.
                # Match the other Windows locks in hermes_cli/auth.py and
                # gateway/status.py: seed a real byte and reset the offset.
                lock_fd.seek(0, os.SEEK_END)
                if lock_fd.tell() == 0:
                    lock_fd.write(" ")
                    lock_fd.flush()
                lock_fd.seek(0)
            else:
                lock_fd.seek(0)
            deadline = time.monotonic() + _JOBS_COMMIT_LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    if fcntl is not None:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    elif msvcrt is not None:
                        lock_fd.seek(0)
                        getattr(msvcrt, "locking")(
                            lock_fd.fileno(), getattr(msvcrt, "LK_NBLCK"), 1
                        )
                    else:  # guarded above; keeps platform narrowing explicit
                        raise AssertionError("unreachable locking backend state")
                    locked = True
                    break
                except (OSError, IOError) as exc:
                    acquisition_error = exc
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.01)
        except (OSError, IOError) as exc:
            acquisition_error = exc

        if not locked:
            raise RuntimeError(
                "Could not acquire the cron jobs commit lock within "
                f"{_JOBS_COMMIT_LOCK_TIMEOUT_SECONDS:g}s; refusing to publish "
                "a potentially stale jobs.json snapshot"
            ) from acquisition_error
        if lock_fd is None:  # narrowed above; defensive for static analysis
            raise AssertionError("locked cron jobs commit lock has no descriptor")

        def validate_lock_identity() -> None:
            if os.name != "posix":
                return
            try:
                fd_stat = os.fstat(lock_fd.fileno())
                path_stat = os.stat(lock_path, follow_symlinks=False)
                if (
                    not stat.S_ISREG(fd_stat.st_mode)
                    or fd_stat.st_nlink != 1
                    or not stat.S_ISREG(path_stat.st_mode)
                    or (fd_stat.st_dev, fd_stat.st_ino)
                    != (path_stat.st_dev, path_stat.st_ino)
                ):
                    raise OSError(
                        "cron jobs commit lock path changed after descriptor open"
                    )
            except (OSError, IOError) as exc:
                raise RuntimeError(
                    "Cron jobs commit lock identity changed; refusing to publish "
                    "a potentially stale jobs.json snapshot"
                ) from exc

        validate_lock_identity()
        yield validate_lock_identity
    finally:
        if lock_fd is not None:
            if locked:
                try:
                    if fcntl is not None:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    elif msvcrt is not None:
                        lock_fd.seek(0)
                        getattr(msvcrt, "locking")(
                            lock_fd.fileno(), getattr(msvcrt, "LK_UNLCK"), 1
                        )
                except (OSError, IOError):
                    pass
            try:
                lock_fd.close()
            except OSError:
                pass


@contextlib.contextmanager
def _jobs_lock():
    """Serialize a load_jobs→modify→save_jobs critical section.

    Combines the in-process threading lock (cheap mutual exclusion between
    the gateway's parallel tick threads) with a cross-process advisory file
    lock on ``<cron dir>/.jobs.lock`` (mutual exclusion between the gateway process
    and standalone ``hermes`` CLI invocations, which previously shared no lock
    at all — a `cron pause` could be silently clobbered by a concurrent
    gateway write, leaving a "paused" job still firing).

    The flock is blocking, but every critical section that uses it is short
    (field updates only — no agent execution), so contention resolves in
    milliseconds. If neither fcntl nor msvcrt is available the manager still
    provides in-process locking, matching the historical behaviour.

    Nested calls in the same thread reuse the held lock so legacy callers that
    invoke save_jobs() inside a broader mutation section don't deadlock or try
    to reacquire the advisory file lock.

    Whenever the cross-process flock could not actually be acquired (timeout,
    OSError, or neither fcntl/msvcrt available), ``_jobs_lock_state.degraded``
    records that fact for diagnostics.  Save still uses a separate short-held
    commit lock and coherent reconciliation to protect publication (#80624).
    """
    depth = getattr(_jobs_lock_state, "depth", 0)
    if depth:
        _jobs_lock_state.depth = depth + 1
        try:
            yield
        finally:
            _jobs_lock_state.depth -= 1
        return

    with _jobs_file_lock:
        _jobs_lock_state.depth = 1
        _jobs_lock_state.degraded = False
        _jobs_lock_state.load_jobs = None
        lock_fd = None
        locked = False
        try:
            try:
                ensure_dirs()
                lock_fd = open(_jobs_lock_file(), "a+", encoding="utf-8")
                lock_fd.seek(0)
                if fcntl is not None:
                    # Bounded acquisition (#60703): a plain blocking
                    # fcntl.flock(LOCK_EX) here has NO timeout, and it is
                    # taken while holding the process-wide _jobs_file_lock
                    # RLock above.  If another process wedges while holding
                    # .jobs.lock (e.g. an old gateway draining through a
                    # restart), a single blocked acquirer freezes EVERY cron
                    # function in this process — including the ticker's
                    # get_due_jobs() — silently and forever: the heartbeat
                    # file stops updating and all jobs stop firing with no
                    # error logged.  Poll LOCK_NB against a deadline instead;
                    # on timeout, log loudly and fall through to the same
                    # in-process-only degraded mode used when locking is
                    # unavailable.  Publication remains serialized later by
                    # the much shorter commit lock; the broad mutation lock is
                    # what must never freeze the scheduler permanently.
                    _deadline = time.monotonic() + _JOBS_LOCK_TIMEOUT_SECONDS
                    while True:
                        try:
                            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            locked = True
                            break
                        except (OSError, IOError):
                            if time.monotonic() >= _deadline:
                                logger.error(
                                    "Timed out after %.0fs waiting for the cron "
                                    "jobs lock (%s) — another process is holding "
                                    "it. Proceeding with in-process locking only "
                                    "so the scheduler stays alive (#60703).",
                                    _JOBS_LOCK_TIMEOUT_SECONDS,
                                    _jobs_lock_file(),
                                )
                                try:
                                    lock_fd.close()
                                except OSError:
                                    pass
                                lock_fd = None
                                break
                            time.sleep(0.1)
                elif msvcrt is not None:
                    getattr(msvcrt, "locking")(lock_fd.fileno(), getattr(msvcrt, "LK_LOCK"), 1)
                    locked = True
            except (OSError, IOError) as e:
                # Never let a locking failure take down cron writes — fall back to
                # in-process-only protection (still held via _jobs_file_lock).
                logger.warning("jobs.json cross-process lock unavailable (%s); "
                               "proceeding with in-process lock only", e)
            _jobs_lock_state.degraded = not locked
            try:
                yield
            finally:
                if lock_fd is not None:
                    try:
                        if fcntl is not None:
                            fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        elif msvcrt is not None:
                            getattr(msvcrt, "locking")(lock_fd.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)
                    except (OSError, IOError):
                        pass
                    finally:
                        lock_fd.close()
        finally:
            _jobs_lock_state.depth = 0
            _jobs_lock_state.degraded = False
            _jobs_lock_state.load_jobs = None

# Fields on a cron job that must never change after creation. ``id`` is used
# as a filesystem path component under ``OUTPUT_DIR``; allowing it to be
# updated lets an unsafe value (``../escape``, absolute path, nested) leak
# into output writes/deletes.
_IMMUTABLE_JOB_FIELDS = frozenset({"id"})


def _job_output_dir(job_id: str) -> Path:
    """Resolve a job's output directory, rejecting any path-escape attempt.

    Job IDs are filesystem path components under ``OUTPUT_DIR``. A legacy or
    crafted ID containing ``..``, absolute paths, or nested separators would
    allow output writes/deletes to escape the cron output sandbox. Reject
    anything that isn't a single safe path component.
    """
    text = str(job_id or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"Invalid cron job id for output path: {job_id!r}")
    if Path(text).is_absolute() or Path(text).drive:
        raise ValueError(f"Invalid cron job id for output path: {job_id!r}")
    return _current_cron_store().output_dir / text


def _normalize_skill_list(skill: Optional[str] = None, skills: Optional[Any] = None) -> List[str]:
    """Normalize legacy/single-skill and multi-skill inputs into a unique ordered list."""
    if skills is None:
        raw_items = [skill] if skill else []
    elif isinstance(skills, str):
        raw_items = [skills]
    else:
        raw_items = list(skills)

    normalized: List[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _apply_skill_fields(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return a job dict with canonical `skills` and legacy `skill` fields aligned."""
    normalized = dict(job)
    skills = _normalize_skill_list(normalized.get("skill"), normalized.get("skills"))
    normalized["skills"] = skills
    normalized["skill"] = skills[0] if skills else None
    return normalized


def _coerce_job_text(value: Any, fallback: str = "") -> str:
    """Coerce legacy/hand-edited nullable cron fields to strings for readers."""
    if value is None:
        return fallback
    return str(value)


def _schedule_display_for_job(job: Dict[str, Any]) -> str:
    display = _coerce_job_text(job.get("schedule_display")).strip()
    if display:
        return display

    schedule = job.get("schedule")
    if isinstance(schedule, dict):
        for key in ("display", "value", "expr", "run_at"):
            text = _coerce_job_text(schedule.get(key)).strip()
            if text:
                return text
    elif schedule is not None:
        return str(schedule)

    return "?"


def _normalize_job_record(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return a read-safe cron job shape for UI/API/tool/scheduler consumers.

    Older or hand-edited jobs can have nullable fields like ``prompt``,
    ``name``, or ``schedule_display``.  Keep storage untouched on read, but
    ensure consumers never crash while formatting or running those records.
    """
    normalized = _apply_skill_fields(job)
    job_id = _coerce_job_text(normalized.get("id"), "unknown")
    prompt = _coerce_job_text(normalized.get("prompt"))
    normalized["id"] = job_id
    normalized["prompt"] = prompt

    name = _coerce_job_text(normalized.get("name")).strip()
    if not name:
        script = _coerce_job_text(normalized.get("script")).strip()
        label_source = (
            prompt
            or (normalized["skills"][0] if normalized.get("skills") else "")
            or script
            or job_id
            or "cron job"
        )
        name = label_source[:50].strip() or "cron job"
    normalized["name"] = name
    normalized["schedule_display"] = _schedule_display_for_job(normalized)

    state = _coerce_job_text(normalized.get("state")).strip()
    if not state:
        state = "scheduled" if normalized.get("enabled", True) else "paused"
    normalized["state"] = state

    return normalized


def _secure_dir(path: Path):
    """Set directory to owner-only access (0700). No-op on Windows."""
    try:
        os.chmod(path, 0o700)
    except (OSError, NotImplementedError):
        pass  # Windows or other platforms where chmod is not supported


def _secure_file(path: Path):
    """Set file to owner-only read/write (0600). No-op on Windows."""
    try:
        if path.exists():
            os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def _ownership_to_restore(
    before: Optional[os.stat_result],
) -> Optional[Tuple[int, int]]:
    """Return the prior owner only when a privileged POSIX writer changed it."""
    if before is None or os.name != "posix":
        return None
    geteuid = getattr(os, "geteuid", None)
    getegid = getattr(os, "getegid", None)
    if geteuid is None or getegid is None:
        return None
    euid, egid = geteuid(), getegid()
    if euid != 0:
        return None
    owner = (before.st_uid, before.st_gid)
    return None if owner == (euid, egid) else owner


def _preserve_file_ownership(path: Path, before: Optional[os.stat_result]) -> None:
    """Restore a rewritten file's previous owner (privileged POSIX writer only).

    The atomic-write pattern (mkstemp + replace) makes the rewritten file owned
    by the *writer's* euid. When a root shell runs a state-writing cron CLI
    command (``docker exec hermes hermes cron create ...`` — ``docker exec``
    defaults to root) against a store owned by the unprivileged gateway user,
    the replace flips ``jobs.json`` to ``root:root`` mode 600 and the gateway's
    ticker (uid 1000) is silently locked out of every subsequent tick (#68483).

    Root can always hand ownership back, so do exactly that: when the euid is 0
    and the pre-replace owner differs, chown the new file to the previous
    uid/gid. Unprivileged writers are a no-op (their own rewrite already heals
    a root-owned file back to their uid, and they couldn't chown anyway).
    No-op on Windows. Best-effort: a failure must never break the save.
    """
    owner = _ownership_to_restore(before)
    if owner is None:
        return
    try:
        os.chown(path, *owner)
    except OSError as e:
        logger.warning(
            "Could not restore ownership of %s to uid=%s gid=%s after rewrite: %s "
            "— if the gateway runs as a different user, its cron ticker may now "
            "be locked out (see issue #68483).",
            path,
            owner[0],
            owner[1],
            e,
        )


def _preserve_fd_ownership(
    fd: int, before: Optional[os.stat_result], description: str
) -> None:
    """Restore owner through an already-validated descriptor when root writes.

    Commit-lock security depends on never resolving its pathname again for a
    metadata mutation: a path-based chown can be redirected after open. This
    descriptor variant deliberately mirrors ``_preserve_file_ownership``'s
    root-only, best-effort ownership policy.
    """
    owner = _ownership_to_restore(before)
    fchown = getattr(os, "fchown", None)
    if owner is None or fchown is None:
        return
    try:
        fchown(fd, *owner)
    except OSError as exc:
        logger.warning(
            "Could not restore ownership of %s to uid=%s gid=%s: %s",
            description,
            owner[0],
            owner[1],
            exc,
        )


def ensure_dirs():
    """Ensure cron directories exist with secure permissions."""
    store = _current_cron_store()
    store.cron_dir.mkdir(parents=True, exist_ok=True)
    store.output_dir.mkdir(parents=True, exist_ok=True)
    _secure_dir(store.cron_dir)
    _secure_dir(store.output_dir)


# =============================================================================
# Schedule Parsing
# =============================================================================

def parse_duration(s: str) -> int:
    """
    Parse duration string into minutes.
    
    Examples:
        "30m" → 30
        "2h" → 120
        "1d" → 1440
    """
    s = s.strip().lower()
    match = re.match(r'^(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$', s)
    if not match:
        raise ValueError(f"Invalid duration: '{s}'. Use format like '30m', '2h', or '1d'")
    
    value = int(match.group(1))
    unit = match.group(2)[0]  # First char: m, h, or d
    
    multipliers = {'m': 1, 'h': 60, 'd': 1440}
    return value * multipliers[unit]


def parse_schedule(schedule: str) -> Dict[str, Any]:
    """
    Parse schedule string into structured format.
    
    Returns dict with:
        - kind: "once" | "interval" | "cron"
        - For "once": "run_at" (ISO timestamp)
        - For "interval": "minutes" (int)
        - For "cron": "expr" (cron expression)
    
    Examples:
        "30m"              → once in 30 minutes
        "2h"               → once in 2 hours
        "every 30m"        → recurring every 30 minutes
        "every 2h"         → recurring every 2 hours
        "0 9 * * *"        → cron expression
        "2026-02-03T14:00" → once at timestamp
    """
    schedule = schedule.strip()
    original = schedule
    schedule_lower = schedule.lower()
    
    # "every X" pattern → recurring interval
    if schedule_lower.startswith("every "):
        duration_str = schedule[6:].strip()
        minutes = parse_duration(duration_str)
        return {
            "kind": "interval",
            "minutes": minutes,
            "display": f"every {minutes}m"
        }
    
    # Check for cron expression (5 or 6 space-separated fields)
    # Cron fields: minute hour day month weekday [year]
    parts = schedule.split()
    if len(parts) >= 5 and all(
        re.match(r'^[\d\*\-,/]+$', p) for p in parts[:5]
    ):
        if not _ensure_croniter():
            raise ValueError("Cron expressions require 'croniter' package. Install with: pip install croniter")
        # Validate cron expression
        try:
            croniter(schedule)
        except Exception as e:
            raise ValueError(f"Invalid cron expression '{schedule}': {e}")
        return {
            "kind": "cron",
            "expr": schedule,
            "display": schedule
        }
    
    # ISO timestamp (contains T or looks like date)
    if 'T' in schedule or re.match(r'^\d{4}-\d{2}-\d{2}', schedule):
        try:
            # Parse and validate
            dt = datetime.fromisoformat(schedule.replace('Z', '+00:00'))
            # Make naive timestamps timezone-aware at parse time so the stored
            # value doesn't depend on the system timezone matching at check time.
            #
            # Anchor to the CONFIGURED Hermes timezone, not the server's local
            # timezone. The due-check (`get_due_jobs`) compares `next_run_at`
            # against `hermes_time.now()`, which uses the configured zone. If a
            # naive "20:07" were interpreted as server-local (e.g. UTC) while
            # now() runs in Asia/Kolkata, the stored instant would land hours
            # off from the user's wall-clock intent — far enough that one-shots
            # never become due and recurring jobs fire at the wrong time. Using
            # the configured zone makes "20:07" mean 20:07 on the same clock the
            # scheduler checks against (#51021).
            if dt.tzinfo is None:
                hermes_tz = _hermes_now().tzinfo
                dt = dt.replace(tzinfo=hermes_tz)
            return {
                "kind": "once",
                "run_at": dt.isoformat(),
                "display": f"once at {dt.strftime('%Y-%m-%d %H:%M')}"
            }
        except ValueError as e:
            raise ValueError(f"Invalid timestamp '{schedule}': {e}")
    
    # Duration like "30m", "2h", "1d" → one-shot from now
    try:
        minutes = parse_duration(schedule)
        run_at = _hermes_now() + timedelta(minutes=minutes)
        return {
            "kind": "once",
            "run_at": run_at.isoformat(),
            "display": f"once in {original}"
        }
    except ValueError:
        pass
    
    raise ValueError(
        f"Invalid schedule '{original}'. Use:\n"
        f"  - Duration: '30m', '2h', '1d' (one-shot)\n"
        f"  - Interval: 'every 30m', 'every 2h' (recurring)\n"
        f"  - Cron: '0 9 * * *' (cron expression)\n"
        f"  - Timestamp: '2026-02-03T14:00:00' (one-shot at time)"
    )


def _ensure_aware(dt: datetime) -> datetime:
    """Return a timezone-aware datetime in Hermes configured timezone.

    Backward compatibility:
    - Older stored timestamps may be naive.
    - Naive values are interpreted as *system-local wall time* (the timezone
      `datetime.now()` used when they were created), then converted to the
      configured Hermes timezone.

    This preserves relative ordering for legacy naive timestamps across
    timezone changes and avoids false not-due results.
    """
    target_tz = _hermes_now().tzinfo
    if dt.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        return dt.replace(tzinfo=local_tz).astimezone(target_tz)
    return dt.astimezone(target_tz)


def _timezone_offset_mismatch(stored: datetime, current: datetime) -> bool:
    """Return True when a stored aware timestamp uses a different UTC offset.

    Naive stored timestamps return False: they carry no offset to compare, and
    are normalized by ``_ensure_aware`` instead — they intentionally never take
    the offset-repair path.
    """
    if stored.tzinfo is None or current.tzinfo is None:
        return False
    return stored.utcoffset() != current.utcoffset()


def _stored_wall_clock_is_future(stored: datetime, current: datetime) -> bool:
    """Return True when the stored local wall-clock time has not arrived yet.

    Cron schedules express local wall-clock intent. If Hermes/system local time
    changes after next_run_at was persisted, an old offset can make a future
    wall-clock run look due at the converted absolute time (for example
    21:00+10 becomes 13:00+02). Comparing naive wall-clock values lets us
    distinguish that migration case from a genuinely missed run whose scheduled
    wall time has already passed.
    """
    return stored.replace(tzinfo=None) > current.replace(tzinfo=None)


def _recoverable_oneshot_run_at(
    schedule: Dict[str, Any],
    now: datetime,
    *,
    last_run_at: Optional[str] = None,
) -> Optional[str]:
    """Return a one-shot run time if it is still eligible to fire.

    One-shot jobs get a small grace window so jobs created a few seconds after
    their requested minute still run on the next tick. Once a one-shot has
    already run, it is never eligible again.
    """
    if not isinstance(schedule, dict) or schedule.get("kind") != "once":
        return None
    if last_run_at:
        return None

    run_at = schedule.get("run_at")
    if not run_at:
        return None

    try:
        run_at_dt = _ensure_aware(datetime.fromisoformat(run_at))
    except Exception:
        return None
    if run_at_dt >= now - timedelta(seconds=ONESHOT_GRACE_SECONDS):
        return run_at
    return None


def _compute_grace_seconds(schedule: dict) -> int:
    """Compute how late a job can be and still catch up instead of fast-forwarding.

    Uses half the schedule period, clamped between 120 seconds and 2 hours.
    This ensures daily jobs can catch up if missed by up to 2 hours,
    while frequent jobs (every 5-10 min) still fast-forward quickly.
    """
    MIN_GRACE = 120
    MAX_GRACE = 7200  # 2 hours

    kind = schedule.get("kind")

    if kind == "interval":
        period_seconds = schedule.get("minutes", 1) * 60
        grace = period_seconds // 2
        return max(MIN_GRACE, min(grace, MAX_GRACE))

    if kind == "cron" and _ensure_croniter():
        expr = schedule.get("expr")
        if expr:
            try:
                now = _hermes_now()
                cron = croniter(expr, now)
                first = cron.get_next(datetime)
                second = cron.get_next(datetime)
                period_seconds = int((second - first).total_seconds())
                grace = period_seconds // 2
                return max(MIN_GRACE, min(grace, MAX_GRACE))
            except Exception:
                pass

    return MIN_GRACE


def compute_next_run(schedule: Dict[str, Any], last_run_at: Optional[str] = None) -> Optional[str]:
    """
    Compute the next run time for a schedule.

    Returns ISO timestamp string, or None if no more runs.
    """
    now = _hermes_now()

    if not isinstance(schedule, dict):
        return None
    kind = schedule.get("kind")
    if kind is None:
        return None

    if kind == "once":
        return _recoverable_oneshot_run_at(schedule, now, last_run_at=last_run_at)

    elif kind == "interval":
        minutes = schedule.get("minutes")
        if minutes is None:
            return None
        if last_run_at:
            try:
                last = _ensure_aware(datetime.fromisoformat(last_run_at))
                next_run = last + timedelta(minutes=minutes)
            except Exception:
                next_run = now + timedelta(minutes=minutes)
        else:
            # First run is now + interval
            next_run = now + timedelta(minutes=minutes)
        return next_run.isoformat()

    elif kind == "cron":
        expr = schedule.get("expr")
        if not expr:
            return None
        if not _ensure_croniter():
            logger.warning(
                "Cannot compute next run for cron schedule %r: 'croniter' is "
                "not installed. croniter is a core dependency as of v0.9.x; "
                "reinstall hermes-agent or run 'pip install croniter' in your "
                "runtime env.",
                expr,
            )
            return None
        # Use last_run_at as the croniter base when available, consistent
        # with interval jobs.  This ensures that after a crash/restart,
        # the next run is anchored to the actual last execution time
        # rather than to an arbitrary restart time.
        base_time = now
        if last_run_at:
            try:
                base_time = _ensure_aware(datetime.fromisoformat(last_run_at))
            except Exception:
                base_time = now
        cron = croniter(expr, base_time)
        next_run = cron.get_next(datetime)
        return next_run.isoformat()

    return None


# =============================================================================
# Ticker heartbeat (liveness signal for `hermes cron status`)
# =============================================================================

def _atomic_write_epoch(path: Path) -> None:
    """Atomically write the current epoch time to ``path``.

    Delegates to :func:`utils.atomic_write_text` (tmpfile + fsync +
    ``atomic_replace``, same pattern as ``save_jobs``) so a concurrent reader
    in another process (``hermes cron status``) never sees a torn/truncated
    file. Best-effort: failures are swallowed by callers.
    """
    ensure_dirs()
    atomic_write_text(path, str(time.time()), tmp_prefix=".hb_")


def _atomic_write_counter(path: Path, value: int) -> None:
    """Atomically persist a non-negative integer counter."""
    ensure_dirs()
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".count_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(max(0, value)))
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def record_ticker_heartbeat(success: bool = False) -> None:
    """Record a ticker liveness signal, and optionally a successful-tick signal.

    The ticker calls this once per loop iteration. ``success=True`` additionally
    bumps the *last successful tick* marker. We track two distinct signals so
    `hermes cron status` can tell a thread that is merely *alive and looping*
    (heartbeat fresh, success stale) from one that is actually *firing jobs*
    (both fresh) — a ticker stuck failing every tick would otherwise keep the
    plain heartbeat fresh and falsely report healthy (#32612, #32895).

    Resolution uses ``_current_cron_store()`` so the heartbeat is correctly
    scoped to the active profile's store — critical under multiplex_profiles
    where each profile needs its own liveness signal (#69377).

    Best-effort: a write failure must never disrupt the tick loop.
    """
    store = _current_cron_store()
    try:
        _atomic_write_epoch(store.cron_dir / "ticker_heartbeat")
    except Exception:
        pass
    if success:
        try:
            _atomic_write_epoch(store.cron_dir / "ticker_last_success")
        except Exception:
            pass


def _epoch_file_age(path: Path) -> Optional[float]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return max(0.0, time.time() - float(raw))
    except Exception:
        return None


def get_ticker_heartbeat_age() -> Optional[float]:
    """Seconds since the ticker loop last iterated, or None if unknown.

    None = heartbeat file missing/unreadable (older build, never ran, or a
    torn read). Callers treat None as "cannot determine", not "dead".

    Resolution uses ``_current_cron_store()`` so the heartbeat is correctly
    scoped to the active profile — critical under multiplex_profiles where
    ``hermes cron status`` must report per-profile liveness (#69377).
    """
    store = _current_cron_store()
    return _epoch_file_age(store.cron_dir / "ticker_heartbeat")


def get_ticker_success_age() -> Optional[float]:
    """Seconds since the ticker last completed a tick WITHOUT raising, or None.

    Resolution uses ``_current_cron_store()`` so the heartbeat is correctly
    scoped to the active profile — critical under multiplex_profiles where
    ``hermes cron status`` must report per-profile liveness (#69377).
    """
    store = _current_cron_store()
    return _epoch_file_age(store.cron_dir / "ticker_last_success")


def record_catch_up_occurrence() -> None:
    """Increment the profile-local stale-schedule catch-up counter, best effort."""
    path = _current_cron_store().cron_dir / "catch_up_occurrences"
    try:
        try:
            value = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            value = 0
        _atomic_write_counter(path, max(0, value) + 1)
    except Exception:
        pass


def record_ticker_error(message: str) -> None:
    """Persist the most recent tick failure so other processes can surface it.

    The ticker thread lives inside the gateway process; ``hermes cron
    status``/``list`` run in a separate process and previously could only
    infer "ticks may be failing" from marker staleness, with no clue WHY.
    A root-owned ``jobs.json`` (#68483) failed every tick for ~14h with the
    reason visible only in the gateway's errors.log. Writing the last error
    next to the heartbeat markers gives the CLI something concrete to show.

    Best-effort: a write failure must never disrupt the tick loop.
    """
    store = _current_cron_store()
    path = store.cron_dir / "ticker_last_error"
    try:
        ensure_dirs()
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".terr_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{time.time()}\n{message.strip()}\n")
                f.flush()
                os.fsync(f.fileno())
            atomic_replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception:
        pass


def get_catch_up_occurrence_count() -> int:
    """Return the profile-local stale-schedule catch-up count."""
    path = _current_cron_store().cron_dir / "catch_up_occurrences"
    try:
        return max(0, int(path.read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        return 0


def clear_ticker_error() -> None:
    """Remove the last-tick-error marker after a successful tick. Best-effort."""
    store = _current_cron_store()
    try:
        (store.cron_dir / "ticker_last_error").unlink()
    except OSError:
        pass


def get_ticker_last_error() -> Optional[str]:
    """Return the most recent recorded tick error message, or None."""
    store = _current_cron_store()
    try:
        raw = (store.cron_dir / "ticker_last_error").read_text(encoding="utf-8")
    except Exception:
        return None
    lines = raw.splitlines()
    if len(lines) < 2:
        return None
    message = "\n".join(lines[1:]).strip()
    return message or None


# =============================================================================
# Job CRUD Operations
# =============================================================================

def _stamp_from_stat(st: os.stat_result) -> _JobsFileStamp:
    """Return the generation fields used to identify one jobs.json inode."""
    return (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_ctime_ns, st.st_size)


def _jobs_file_stamp(jobs_file: Path) -> Optional[_JobsFileStamp]:
    """Return the generation currently reachable through ``jobs_file``."""
    try:
        return _stamp_from_stat(jobs_file.stat())
    except FileNotFoundError:
        return None


def _read_jobs_file_snapshot(jobs_file: Path) -> _JobsFileSnapshot:
    """Read JSON and identity from one descriptor without mutating lock state.

    Atomic replacement changes which inode the path names while an already-open
    descriptor continues to expose the old bytes.  Capturing ``fstat`` from the
    descriptor that supplied the JSON keeps those two facts coherent.  The
    before/after check also retries a rare in-place writer instead of accepting
    bytes paired with metadata from two different moments.
    """
    last_decode_error = None
    last_attempt_was_decode_error = False
    for _attempt in range(_JOBS_GENERATION_MAX_ATTEMPTS):
        try:
            with open(jobs_file, "rb") as f:
                before_stat = os.fstat(f.fileno())
                payload = f.read()
                after_stat = os.fstat(f.fileno())
        except FileNotFoundError:
            return _JobsFileSnapshot(exists=False, data=None, stamp=None)
        before = _stamp_from_stat(before_stat)
        after = _stamp_from_stat(after_stat)
        if before != after:
            last_attempt_was_decode_error = False
            continue
        try:
            payload_text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            last_decode_error = exc
            last_attempt_was_decode_error = True
            continue

        strict_retry = False
        try:
            data = json.loads(payload_text)
        except json.JSONDecodeError:
            strict_retry = True
            try:
                data = json.loads(payload_text, strict=False)
            except json.JSONDecodeError as exc:
                # Do not keep retrying the bytes from this descriptor. A
                # sibling may already have atomically installed a valid
                # generation, so close/reopen on the next bounded attempt.
                last_decode_error = exc
                last_attempt_was_decode_error = True
                continue
        return _JobsFileSnapshot(
            exists=True,
            data=data,
            stamp=after,
            strict_retry=strict_retry,
        )
    if last_attempt_was_decode_error and last_decode_error is not None:
        raise last_decode_error
    raise RuntimeError("Cron database changed repeatedly while it was being read")


def _jobs_from_snapshot(snapshot: _JobsFileSnapshot) -> List[Dict[str, Any]]:
    """Validate and return a snapshot's jobs without repairing or recording it."""
    if not snapshot.exists:
        return []
    if isinstance(snapshot.data, dict):
        result = snapshot.data.get("jobs", [])
    elif isinstance(snapshot.data, list):
        result = snapshot.data
    else:
        raise RuntimeError(
            "Cron database corrupted: expected {'jobs': [...]}, got "
            f"{type(snapshot.data).__name__}"
        )
    if not isinstance(result, list):
        raise RuntimeError(
            "Cron database corrupted: expected 'jobs' to be a list, got "
            f"{type(result).__name__}"
        )
    return result


def _record_load_snapshot(jobs: List[Dict[str, Any]]) -> None:
    """Record a deep-copied B snapshot for a later three-way save.

    Only meaningful inside a _jobs_lock() critical section (depth > 0); lets
    _save_jobs_unlocked() compare the caller's desired D with the exact B it
    loaded and the latest current C. The deep copy is essential because most
    cron callers mutate the returned list and nested job dictionaries in place.
    """
    if not getattr(_jobs_lock_state, "depth", 0):
        return
    _jobs_lock_state.load_jobs = copy.deepcopy(jobs)


def _read_validated_jobs_snapshot(
    jobs_file: Path,
) -> Tuple[_JobsFileSnapshot, List[Dict[str, Any]]]:
    """Read one coherent generation and expose storage failures uniformly."""
    try:
        snapshot = _read_jobs_file_snapshot(jobs_file)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"Cron database corrupted and unrepairable: {exc}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to read cron database: {exc}") from exc
    return snapshot, _jobs_from_snapshot(snapshot)


def _snapshot_repair_reason(snapshot: _JobsFileSnapshot) -> Optional[str]:
    if not snapshot.exists:
        return None
    if isinstance(snapshot.data, dict) and snapshot.strict_retry:
        return "had invalid control characters"
    if isinstance(snapshot.data, list):
        return "bare list wrapped as dict"
    return None


def _repair_jobs_snapshot(jobs_file: Path) -> List[Dict[str, Any]]:
    """Re-read under the broad lock and normalize only the latest generation."""
    with _jobs_lock():
        snapshot, current_jobs = _read_validated_jobs_snapshot(jobs_file)
        _record_load_snapshot(current_jobs)
        reason = _snapshot_repair_reason(snapshot)
        if reason is None:
            return current_jobs
        repaired_jobs = _save_jobs_unlocked(current_jobs)
        logger.warning("Auto-repaired jobs.json (%s)", reason)
        return repaired_jobs


def load_jobs() -> List[Dict[str, Any]]:
    """Load all jobs from storage."""
    jobs_file = _current_cron_store().jobs_file
    ensure_dirs()
    try:
        snapshot, jobs = _read_validated_jobs_snapshot(jobs_file)
    except RuntimeError as exc:
        logger.error("Failed to load jobs.json: %s", exc)
        raise

    _record_load_snapshot(jobs)
    if _snapshot_repair_reason(snapshot) is not None:
        # A public load may not hold _jobs_lock. Reacquire it and re-read: the
        # generation that requested repair can already have been replaced by
        # a sibling, and blindly saving the stale decoded list would erase it.
        return _repair_jobs_snapshot(jobs_file)
    return jobs


_MISSING_JOB = object()


def _valid_job_id(job: Any) -> Optional[str]:
    """Return an identity usable for record-level merge, else ``None``."""
    if not isinstance(job, dict):
        return None
    job_id = job.get("id")
    return job_id if isinstance(job_id, str) and job_id else None


def _poisoned_job_ids(*snapshots: List[Dict[str, Any]]) -> Set[str]:
    """Return IDs duplicated within any one B/D/C snapshot.

    A duplicate cannot safely identify either record. Poisoning that ID across
    all three views keeps the entire duplicate group in one conflict domain
    instead of silently coalescing it in a dictionary.
    """
    poisoned = set()
    for snapshot in snapshots:
        seen = set()
        for job in snapshot:
            job_id = _valid_job_id(job)
            if job_id is None:
                continue
            if job_id in seen:
                poisoned.add(job_id)
            seen.add(job_id)
    return poisoned


def _partition_jobs_for_merge(
    jobs: List[Dict[str, Any]], poisoned_ids: Set[str]
) -> Tuple[List[str], Dict[str, Dict[str, Any]], List[Any]]:
    """Partition jobs into uniquely identified records and an opaque bag."""
    order = []
    identified = {}
    unidentifiable = []
    for job in jobs:
        job_id = _valid_job_id(job)
        if job_id is None or job_id in poisoned_ids:
            unidentifiable.append(job)
            continue
        order.append(job_id)
        identified[job_id] = job
    return order, identified, unidentifiable


def _job_order_signature(
    jobs: List[Dict[str, Any]], poisoned_ids: Set[str], bag_namespace: str
) -> List[Any]:
    """Represent identified IDs and ordered opaque occurrences for comparison."""
    signature = []
    bag_index = 0
    for job in jobs:
        job_id = _valid_job_id(job)
        if job_id is None or job_id in poisoned_ids:
            signature.append(("bag", bag_namespace, bag_index))
            bag_index += 1
        else:
            signature.append(("id", job_id))
    return signature


def _relative_job_order_changed(base_order: List[Any], side_order: List[Any]) -> bool:
    """Return whether one side reordered tokens that exist in both snapshots."""
    common_tokens = set(base_order) & set(side_order)
    return [token for token in base_order if token in common_tokens] != [
        token for token in side_order if token in common_tokens
    ]


def _sequence_can_overlay_on_other_skeleton(
    base_order: List[Any], side_order: List[Any]
) -> bool:
    """Allow only deletions and suffix additions on another side's skeleton."""
    if _relative_job_order_changed(base_order, side_order):
        return False
    base_tokens = set(base_order)
    saw_addition = False
    for token in side_order:
        if token not in base_tokens:
            saw_addition = True
        elif saw_addition:
            # Reassembly appends IDs missing from the chosen skeleton. An
            # insertion before a surviving base token would lose its position.
            return False
    return True


def _merge_jobs_three_way(
    base: List[Dict[str, Any]],
    desired: List[Dict[str, Any]],
    current: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge caller D and latest C against loaded B without mutating inputs."""
    # Preserve the exact winning snapshot—including its source order—when one
    # side did not change or both sides made the same change. Besides avoiding
    # needless partitioning, this retains the legacy tolerance for malformed
    # records in the uncontested cases.
    if desired == base:
        return copy.deepcopy(current)
    if current == base:
        return copy.deepcopy(desired)
    if desired == current:
        return copy.deepcopy(desired)

    # Missing/invalid IDs, non-object legacy rows, and every member of a
    # duplicate-ID group cannot be matched record-by-record. Treat them as one
    # opaque ordered bag and apply the same B/D/C rules to that whole bag. This
    # lets an in-place legacy repair coexist with a sibling change elsewhere,
    # while divergent edits inside the ambiguous bag still fail closed.
    poisoned_ids = _poisoned_job_ids(base, desired, current)
    base_order, base_by_id, base_bag = _partition_jobs_for_merge(
        base, poisoned_ids
    )
    desired_order, desired_by_id, desired_bag = _partition_jobs_for_merge(
        desired, poisoned_ids
    )
    current_order, current_by_id, current_bag = _partition_jobs_for_merge(
        current, poisoned_ids
    )

    chosen = {}
    all_ids = dict.fromkeys(base_order + desired_order + current_order)
    for job_id in all_ids:
        base_job = base_by_id.get(job_id, _MISSING_JOB)
        desired_job = desired_by_id.get(job_id, _MISSING_JOB)
        current_job = current_by_id.get(job_id, _MISSING_JOB)
        if desired_job == base_job:
            merged_job = current_job
        elif current_job == base_job:
            merged_job = desired_job
        elif desired_job == current_job:
            merged_job = desired_job
        else:
            raise RuntimeError(
                "Refusing to save conflicting concurrent cron job change for "
                f"id {job_id!r}"
            )
        chosen[job_id] = merged_job

    # Order is part of the persisted snapshot contract here. An order-only
    # change inside the opaque bag cannot be attributed to individual rows, so
    # use ordinary list equality and retain the selected source bag verbatim.
    if desired_bag == base_bag:
        merged_bag = current_bag
        bag_order_source = current if current_bag != base_bag else None
    elif current_bag == base_bag:
        merged_bag = desired_bag
        bag_order_source = desired
    else:
        raise RuntimeError(
            "Refusing to save conflicting concurrent unidentifiable cron job "
            "changes"
        )

    base_signature = _job_order_signature(base, poisoned_ids, "base")
    desired_signature = _job_order_signature(
        desired,
        poisoned_ids,
        "base" if desired_bag == base_bag else "desired",
    )
    current_signature = _job_order_signature(
        current,
        poisoned_ids,
        "base" if current_bag == base_bag else "current",
    )
    desired_sequence_changed = desired_signature != base_signature
    current_sequence_changed = current_signature != base_signature
    desired_overlay_safe = _sequence_can_overlay_on_other_skeleton(
        base_signature, desired_signature
    )
    current_overlay_safe = _sequence_can_overlay_on_other_skeleton(
        base_signature, current_signature
    )

    # Reassembly needs one full-list ordering skeleton. Deletions and suffix
    # additions from the other side can be overlaid by skipping/appending the
    # chosen IDs. A mid-list insertion or reorder cannot be moved to the end
    # silently. A changed opaque bag forces its owning side's skeleton; without
    # one, prefer the side on which the other sequence can be overlaid safely.
    if bag_order_source is current:
        if desired_sequence_changed and not desired_overlay_safe:
            raise RuntimeError(
                "Refusing to save concurrent cron job ordering and "
                "unidentifiable-record changes"
            )
        order_source = current
    elif bag_order_source is desired:
        if current_sequence_changed and not current_overlay_safe:
            raise RuntimeError(
                "Refusing to save concurrent cron job ordering and "
                "unidentifiable-record changes"
            )
        order_source = desired
    elif desired_signature == current_signature:
        order_source = desired
    elif desired_sequence_changed and current_sequence_changed:
        if current_overlay_safe:
            order_source = desired
        elif desired_overlay_safe:
            order_source = current
        else:
            raise RuntimeError(
                "Refusing to save conflicting concurrent cron job ordering changes"
            )
    elif desired_sequence_changed and not current_sequence_changed:
        order_source = desired
    elif current_sequence_changed and not desired_sequence_changed:
        order_source = current
    else:
        order_source = desired

    _, _, order_source_bag = _partition_jobs_for_merge(
        order_source, poisoned_ids
    )
    if order_source_bag != merged_bag:
        raise RuntimeError(
            "Refusing to save cron jobs without a coherent ordering for "
            "unidentifiable records"
        )

    result = []
    emitted = set()
    for job in order_source:
        job_id = _valid_job_id(job)
        if job_id is None or job_id in poisoned_ids:
            result.append(copy.deepcopy(job))
            continue
        emitted.add(job_id)
        merged_job = chosen[job_id]
        if merged_job is not _MISSING_JOB:
            result.append(copy.deepcopy(merged_job))
    for job_id in desired_order + current_order + base_order:
        if job_id in emitted:
            continue
        emitted.add(job_id)
        merged_job = chosen[job_id]
        if merged_job is not _MISSING_JOB:
            result.append(copy.deepcopy(merged_job))
    if merged_bag:
        logger.warning(
            "Reconciled jobs.json while preserving %d unidentifiable or "
            "duplicate-ID record(s) as one conflict domain",
            len(merged_bag),
        )
    return result


def _save_jobs_unlocked(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Save jobs while preserving a loaded B, or explicitly replacing without B.

    A call with no preceding load in this lock is an intentional whole-snapshot
    recovery operation. It captures only the current path generation (so
    corrupt or unreadable content can be replaced), then performs the same
    staged generation recheck.
    """
    jobs_file = _current_cron_store().jobs_file
    ensure_dirs()
    desired_jobs = copy.deepcopy(jobs)
    loaded_jobs = getattr(_jobs_lock_state, "load_jobs", None)
    base_jobs = copy.deepcopy(loaded_jobs) if loaded_jobs is not None else None
    with _jobs_commit_lock() as validate_commit_lock:
        for attempt in range(1, _JOBS_GENERATION_MAX_ATTEMPTS + 1):
            if base_jobs is None:
                try:
                    observed_stamp = _jobs_file_stamp(jobs_file)
                except OSError as exc:
                    raise RuntimeError(f"Failed to read cron database: {exc}") from exc
                reconciled_jobs = copy.deepcopy(desired_jobs)
            else:
                current_snapshot, current_jobs = _read_validated_jobs_snapshot(
                    jobs_file
                )
                reconciled_jobs = _merge_jobs_three_way(
                    base_jobs, desired_jobs, current_jobs
                )
                observed_stamp = current_snapshot.stamp
            # Snapshot the current owner BEFORE the atomic replace so a
            # privileged writer (root CLI in Docker) can hand ownership back
            # to the gateway user afterwards instead of locking its ticker out
            # (#68483). When creating the file, inherit the cron dir's owner.
            try:
                stat_before = os.stat(jobs_file)
            except OSError:
                try:
                    stat_before = os.stat(jobs_file.parent)
                except OSError:
                    stat_before = None
            fd, tmp_path = tempfile.mkstemp(
                dir=str(jobs_file.parent), suffix=".tmp", prefix=".jobs_"
            )
            published = False
            raw_fd = fd
            try:
                if os.name == "posix":
                    os.fchmod(raw_fd, 0o600)
                file_obj = os.fdopen(raw_fd, "w", encoding="utf-8")
                raw_fd = -1
                with file_obj as f:
                    json.dump(
                        {
                            "jobs": reconciled_jobs,
                            "updated_at": _hermes_now().isoformat(),
                        },
                        f,
                        indent=2,
                    )
                    f.flush()
                    os.fsync(f.fileno())

                if _jobs_file_stamp(jobs_file) != observed_stamp:
                    logger.warning(
                        "jobs.json changed while this process staged its save; "
                        "re-reading and reconciling before publish (attempt %d/%d)",
                        attempt,
                        _JOBS_GENERATION_MAX_ATTEMPTS,
                    )
                    continue

                # The commit-lock pathname can be renamed/recreated after its
                # initial validation, splitting cooperating writers across two
                # lock inodes. Revalidate at the publication boundary.
                validate_commit_lock()
                published_path = Path(atomic_replace(tmp_path, jobs_file))
                published = True
                # mkstemp/fchmod secures the normal rename path. Re-apply mode
                # to cover atomic_replace's EXDEV/EBUSY copy fallback too.
                _secure_file(published_path)
                _preserve_file_ownership(published_path, stat_before)
                if getattr(_jobs_lock_state, "depth", 0):
                    # Preserve the caller's view D, not merged C'. A recovered
                    # C-only create/delete must remain external provenance on
                    # a second save within the same outer mutation lock.
                    _jobs_lock_state.load_jobs = copy.deepcopy(desired_jobs)
                return copy.deepcopy(reconciled_jobs)
            finally:
                if raw_fd >= 0:
                    try:
                        os.close(raw_fd)
                    except OSError:
                        pass
                if not published:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    raise RuntimeError(
        "jobs.json kept changing while this process prepared a save; "
        "refusing to overwrite concurrent updates"
    )


def save_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Save all jobs and return the exact reconciled snapshot published."""
    with _jobs_lock():
        return _save_jobs_unlocked(jobs)


def _normalize_workdir(workdir: Optional[str]) -> Optional[str]:
    """Normalize and validate a cron job workdir.

    Rules:
      - Empty / None → None (feature off, preserves old behaviour).
      - ``~`` is expanded.  Relative paths are rejected — cron jobs run detached
        from any shell cwd, so relative paths have no stable meaning.
      - The path must exist and be a directory at create/update time.  We do
        NOT re-check at run time (a user might briefly unmount the dir; the
        scheduler will just fall back to old behaviour with a logged warning).

    Returns the absolute path string, or None when disabled.
    Raises ValueError on invalid input.
    """
    if workdir is None:
        return None
    raw = str(workdir).strip()
    if not raw:
        return None
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        raise ValueError(
            f"Cron workdir must be an absolute path (got {raw!r}). "
            f"Cron jobs run detached from any shell cwd, so relative paths are ambiguous."
        )
    resolved = expanded.resolve()
    if not resolved.exists():
        raise ValueError(f"Cron workdir does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Cron workdir is not a directory: {resolved}")
    return str(resolved)


def _resolve_default_model_snapshot() -> Optional[str]:
    """Resolve the global default model the same way the cron ticker does.

    Mirrors the unpinned-model resolution in ``cron/scheduler.py`` ``run_job``:
    read ``config.yaml`` ``model.default`` (or the ``model`` alias / bare string
    form), applying the managed-scope overlay and env expansion. Used by
    ``create_job`` to snapshot the default model for unpinned jobs so a later
    swap of the global default is detected at fire time (#44585).

    Returns the resolved model string, or ``None`` if config is missing/empty
    or resolution fails (fail-open — caller treats ``None`` as "no snapshot").
    """
    try:
        from hermes_cli.config import _expand_env_vars, read_user_config_raw

        cfg_path = get_hermes_home() / "config.yaml"
        if not cfg_path.exists():
            return None
        cfg = read_user_config_raw(cfg_path)
        try:
            from hermes_cli import managed_scope
            cfg = managed_scope.apply_managed_overlay(cfg)
        except Exception:
            pass
        cfg = _expand_env_vars(cfg)
        # Mirror run_job's precedence: the explicit cron-fleet default
        # (cron.model) beats the global chat model for unpinned cron jobs.
        cron_cfg = cfg.get("cron") or {}
        if isinstance(cron_cfg, dict):
            cron_model = cron_cfg.get("model")
            if isinstance(cron_model, str) and cron_model.strip():
                return cron_model.strip()
        model_cfg = cfg.get("model") or {}
        if isinstance(model_cfg, str):
            return model_cfg.strip() or None
        if isinstance(model_cfg, dict):
            default = model_cfg.get("default") or model_cfg.get("model")
            if isinstance(default, str):
                return default.strip() or None
        return None
    except Exception:
        return None


def _normalize_job_optional_text(value: Any, *, strip_trailing_slash: bool = False) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if strip_trailing_slash:
        text = text.rstrip("/")
    return text or None


def _compute_provider_model_snapshots(
    *,
    provider: Any,
    model: Any,
    base_url: Any,
    no_agent: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Snapshot unpinned inference axes for the provider/model drift guard.

    Agent cron jobs with unpinned provider/model follow global config at fire
    time. Capture the current resolution for each unpinned axis so a later
    global switch fails closed instead of silently changing spend. Pinned axes
    and no-agent script jobs intentionally carry no snapshot.
    """
    normalized_provider = _normalize_job_optional_text(provider)
    normalized_model = _normalize_job_optional_text(model)
    normalized_base_url = _normalize_job_optional_text(
        base_url,
        strip_trailing_slash=True,
    )
    if bool(no_agent):
        return None, None

    provider_snapshot: Optional[str] = None
    model_snapshot: Optional[str] = None
    if normalized_provider is None:
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider

            runtime_kwargs = {"requested": None}
            if normalized_base_url:
                runtime_kwargs["explicit_base_url"] = normalized_base_url
            snap = resolve_runtime_provider(**runtime_kwargs)
            snap_provider = str(snap.get("provider") or "").strip().lower()
            provider_snapshot = snap_provider or None
        except Exception:
            provider_snapshot = None
    if normalized_model is None:
        try:
            model_snapshot = _resolve_default_model_snapshot() or None
        except Exception:
            model_snapshot = None
    return provider_snapshot, model_snapshot


def _normalized_inference_axes(job: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """Return the stored inference-routing fields in their semantic form."""
    return (
        _normalize_job_optional_text(job.get("provider")),
        _normalize_job_optional_text(job.get("model")),
        _normalize_job_optional_text(job.get("base_url"), strip_trailing_slash=True),
        bool(job.get("no_agent")),
    )


def create_job(
    prompt: Optional[str],
    schedule: str,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    deliver: Optional[str] = None,
    origin: Optional[Dict[str, Any]] = None,
    skill: Optional[str] = None,
    skills: Optional[List[str]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    script: Optional[str] = None,
    context_from: Optional[Union[str, List[str]]] = None,
    enabled_toolsets: Optional[List[str]] = None,
    workdir: Optional[str] = None,
    no_agent: bool = False,
    attach_to_session: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Create a new cron job.

    Args:
        prompt: The prompt to run (must be self-contained, or a task instruction when skill is set).
                Ignored when ``no_agent=True`` except as an optional name hint.
        schedule: Schedule string (see parse_schedule)
        name: Optional friendly name
        repeat: How many times to run (None = forever, 1 = once)
        deliver: Where to deliver output ("origin", "local", "telegram", etc.)
        origin: Source info where job was created (for "origin" delivery)
        skill: Optional legacy single skill name to load before running the prompt
        skills: Optional ordered list of skills to load before running the prompt
        model: Optional per-job model override
        provider: Optional per-job provider override
        base_url: Optional per-job base URL override
        script: Optional path to a script whose stdout feeds the job. With
                ``no_agent=True`` the script IS the job — its stdout is
                delivered verbatim. Without ``no_agent``, its stdout is
                injected into the agent's prompt as context (data-collection /
                change-detection pattern). Paths resolve under
                ~/.hermes/scripts/; ``.sh`` / ``.bash`` files run via bash,
                anything else via Python.
        context_from: Optional job ID (or list of job IDs) whose most recent output
                      is injected into the prompt as context before each run.
                      Useful for chaining cron jobs: job A finds data, job B processes it.
        enabled_toolsets: Optional list of toolset names to restrict the agent to.
                          When set, only tools from these toolsets are loaded, reducing
                          token overhead. When omitted, all default tools are loaded.
                          Ignored when ``no_agent=True``.
        workdir: Optional absolute path.  When set, the job runs as if launched
                from that directory: AGENTS.md / CLAUDE.md / .cursorrules from
                that directory are injected into the system prompt, and the
                terminal/file/code_exec tools use it as their working directory
                (via TERMINAL_CWD).  When unset, the old behaviour is preserved
                (no context files injected, tools use the scheduler's cwd).
                With ``no_agent=True``, ``workdir`` is still applied as the
                script's cwd so relative paths inside the script behave
                predictably.
        no_agent: When True, skip the agent entirely — run ``script`` on schedule
                and deliver its stdout directly. Empty stdout = silent (no
                delivery). Requires ``script`` to be set. Ideal for classic
                watchdogs and periodic alerts that don't need LLM reasoning.

    Returns:
        The created job dict
    """
    parsed_schedule = parse_schedule(schedule)

    # Normalize repeat: treat 0 or negative values as None (infinite)
    if repeat is not None and repeat <= 0:
        repeat = None

    # Auto-set repeat=1 for one-shot schedules if not specified
    if parsed_schedule["kind"] == "once" and repeat is None:
        repeat = 1

    # Default delivery to origin if available, otherwise local
    if deliver is None:
        deliver = "origin" if origin else "local"

    job_id = uuid.uuid4().hex[:12]
    now = _hermes_now().isoformat()

    normalized_skills = _normalize_skill_list(skill, skills)
    normalized_model = _normalize_job_optional_text(model)
    normalized_provider = _normalize_job_optional_text(provider)
    normalized_base_url = _normalize_job_optional_text(base_url, strip_trailing_slash=True)
    normalized_script = str(script).strip() if isinstance(script, str) else None
    normalized_script = normalized_script or None
    normalized_toolsets = [str(t).strip() for t in enabled_toolsets if str(t).strip()] if enabled_toolsets else None
    normalized_toolsets = normalized_toolsets or None
    normalized_workdir = _normalize_workdir(workdir)
    normalized_no_agent = bool(no_agent)
    normalized_attach = attach_to_session if isinstance(attach_to_session, bool) else None

    # no_agent jobs are meaningless without a script — the script IS the job.
    # Surface this as a clear ValueError at create time so bad configs never
    # reach the scheduler.
    if normalized_no_agent and not normalized_script:
        raise ValueError(
            "no_agent=True requires a script — with no agent and no script "
            "there is nothing for the job to run."
        )

    # Normalize context_from: accept str or list of str, store as list or None
    if isinstance(context_from, str):
        context_from = [context_from.strip()] if context_from.strip() else None
    elif isinstance(context_from, list):
        context_from = [str(j).strip() for j in context_from if str(j).strip()] or None
    else:
        context_from = None

    prompt_text = _coerce_job_text(prompt)

    # Reject cron jobs that schedule gateway-lifecycle commands. Prevents
    # agent-driven SIGTERM-respawn loops under launchd/systemd KeepAlive
    # (#30719). Enforced here (not only in the CLI layer) so the agent's
    # `cronjob` model tool — which calls create_job directly — is also
    # covered, not just `hermes cron create`.
    from cron.lifecycle_guard import check_gateway_lifecycle
    check_gateway_lifecycle(prompt_text, normalized_script)

    label_source = (prompt_text or (normalized_skills[0] if normalized_skills else None) or (normalized_script if normalized_no_agent else None)) or "cron job"

    provider_snapshot, model_snapshot = _compute_provider_model_snapshots(
        provider=normalized_provider,
        model=normalized_model,
        base_url=normalized_base_url,
        no_agent=normalized_no_agent,
    )

    next_run_at = compute_next_run(parsed_schedule)
    if parsed_schedule.get("kind") == "once" and next_run_at is None:
        run_at = parsed_schedule.get("run_at") or schedule
        logger.warning(
            "Rejecting one-shot cron job '%s': run_at %s is outside the %ss grace window",
            name or label_source[:50].strip(),
            run_at,
            ONESHOT_GRACE_SECONDS,
        )
        raise ValueError(
            f"Requested one-shot time {run_at} is more than "
            f"{ONESHOT_GRACE_SECONDS}s in the past and cannot be scheduled."
        )

    job = {
        "id": job_id,
        "name": name or label_source[:50].strip(),
        "prompt": prompt_text,
        "skills": normalized_skills,
        "skill": normalized_skills[0] if normalized_skills else None,
        "model": normalized_model,
        "provider": normalized_provider,
        # Provider/model resolution captured at creation for unpinned jobs
        # (#44585). None for pinned axes, no_agent jobs, resolution failures, and
        # any pre-existing job written before these fields existed (back-compat).
        "provider_snapshot": provider_snapshot,
        "model_snapshot": model_snapshot,
        "base_url": normalized_base_url,
        "script": normalized_script,
        "no_agent": normalized_no_agent,
        "context_from": context_from,
        "schedule": parsed_schedule,
        "schedule_display": parsed_schedule.get("display", schedule),
        "repeat": {
            "times": repeat,  # None = forever
            "completed": 0
        },
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "created_at": now,
        "next_run_at": next_run_at,
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "last_delivery_error": None,
        # Delivery configuration
        "deliver": deliver,
        "origin": origin,  # Tracks where job was created for "origin" delivery
        "enabled_toolsets": normalized_toolsets,
        "workdir": normalized_workdir,
    }
    # Only persist attach_to_session when explicitly set, so existing jobs and
    # the common case stay byte-identical (absent key => fall back to the
    # global cron.mirror_delivery config, default off).
    if normalized_attach is not None:
        job["attach_to_session"] = normalized_attach

    with _jobs_lock():
        jobs = load_jobs()
        jobs.append(job)
        save_jobs(jobs)

    return job


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a job by ID."""
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            return _normalize_job_record(job)
    return None


class AmbiguousJobReference(LookupError):
    """Raised when a job name matches more than one job."""

    def __init__(self, ref: str, matches: List[Dict[str, Any]]):
        self.ref = ref
        self.matches = matches
        ids = ", ".join(m["id"] for m in matches)
        super().__init__(
            f"Job name '{ref}' is ambiguous — matches {len(matches)} jobs: {ids}. "
            f"Use the job ID instead."
        )


def resolve_job_ref(ref: str) -> Optional[Dict[str, Any]]:
    """Resolve a job reference (ID or name) to a job record.

    - Exact ID match wins (works even if a different job's name equals this ID).
    - Otherwise, case-insensitive name match.
    - If a name matches more than one job, raises AmbiguousJobReference so the
      caller can surface the matching IDs rather than silently picking one.
    """
    if not ref:
        return None
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == ref:
            return _normalize_job_record(job)
    ref_lower = ref.lower()
    name_matches = [j for j in jobs if (j.get("name") or "").lower() == ref_lower]
    if not name_matches:
        return None
    if len(name_matches) > 1:
        raise AmbiguousJobReference(
            ref, [_normalize_job_record(j) for j in name_matches]
        )
    return _normalize_job_record(name_matches[0])


def list_jobs(include_disabled: bool = False) -> List[Dict[str, Any]]:
    """List all jobs, optionally including disabled ones."""
    jobs = [_normalize_job_record(j) for j in load_jobs()]
    if not include_disabled:
        jobs = [j for j in jobs if j.get("enabled", True)]
    try:
        from cron.executions import latest_executions

        latest = latest_executions([job.get("id", "") for job in jobs])
    except Exception:
        latest = {}
    for job in jobs:
        job["latest_execution"] = latest.get(job.get("id", ""))
    return jobs


def update_job(job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a job by ID, refreshing derived schedule fields when needed."""
    # Block mutation of immutable fields. ``id`` in particular is a filesystem
    # path component under OUTPUT_DIR — letting an update change it leaks
    # path-escape values into output writes/deletes.
    bad_fields = _IMMUTABLE_JOB_FIELDS.intersection(updates or {})
    if bad_fields:
        raise ValueError(
            f"Cron job field(s) cannot be updated: {', '.join(sorted(bad_fields))}"
        )

    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] != job_id:
                continue

            # Validate / normalize workdir if present in updates.  Empty string
            # or None both mean "clear the field" (restore old behaviour).
            if "workdir" in updates:
                _wd = updates["workdir"]
                if _wd in {None, "", False}:
                    updates["workdir"] = None
                else:
                    updates["workdir"] = _normalize_workdir(_wd)

            previous_inference_axes = _normalized_inference_axes(job)
            updated = _apply_skill_fields({**job, **updates})
            schedule_changed = "schedule" in updates
            inference_fields_changed = bool(
                {"provider", "model", "base_url", "no_agent"}.intersection(updates)
            ) and _normalized_inference_axes(updated) != previous_inference_axes

            if "skills" in updates or "skill" in updates:
                normalized_skills = _normalize_skill_list(updated.get("skill"), updated.get("skills"))
                updated["skills"] = normalized_skills
                updated["skill"] = normalized_skills[0] if normalized_skills else None

            if schedule_changed:
                updated_schedule = updated["schedule"]
                # The API may pass schedule as a raw string (e.g. "every 10m")
                # instead of a pre-parsed dict.  Normalize it the same way
                # create_job() does so downstream code can call .get() safely.
                if isinstance(updated_schedule, str):
                    updated_schedule = parse_schedule(updated_schedule)
                    updated["schedule"] = updated_schedule
                updated["schedule_display"] = updates.get(
                    "schedule_display",
                    updated_schedule.get("display", updated.get("schedule_display")),
                )
                if updated.get("state") != "paused":
                    updated_next_run = compute_next_run(updated_schedule)
                    # Same guard as create_job: an UPDATE that sets a one-shot
                    # to a time >ONESHOT_GRACE_SECONDS in the past would store
                    # next_run_at=None with state="scheduled", re-creating the
                    # ghost job that never fires (#59395). Reject it here too so
                    # the bug can't re-enter through the update door.
                    if (
                        updated_next_run is None
                        and updated_schedule.get("kind") == "once"
                    ):
                        run_at = updated_schedule.get("run_at") or updated_schedule
                        logger.warning(
                            "Rejecting one-shot cron job update '%s': run_at %s "
                            "is outside the %ss grace window",
                            updated.get("name", job_id),
                            run_at,
                            ONESHOT_GRACE_SECONDS,
                        )
                        raise ValueError(
                            f"Requested one-shot time {run_at} is more than "
                            f"{ONESHOT_GRACE_SECONDS}s in the past and cannot be scheduled."
                        )
                    updated["next_run_at"] = updated_next_run

            if inference_fields_changed:
                provider_snapshot, model_snapshot = _compute_provider_model_snapshots(
                    provider=updated.get("provider"),
                    model=updated.get("model"),
                    base_url=updated.get("base_url"),
                    no_agent=updated.get("no_agent"),
                )
                updated["provider_snapshot"] = provider_snapshot
                updated["model_snapshot"] = model_snapshot

            if updated.get("enabled", True) and updated.get("state") != "paused" and not updated.get("next_run_at"):
                next_run = compute_next_run(updated["schedule"])
                if next_run is None and updated["schedule"].get("kind") == "once":
                    run_at = updated["schedule"].get("run_at", "unknown")
                    raise ValueError(
                        f"Requested one-shot time {run_at} is in the past "
                        f"(grace window: {ONESHOT_GRACE_SECONDS}s) and cannot be scheduled."
                    )
                updated["next_run_at"] = next_run

            jobs[i] = updated
            save_jobs(jobs)
            return _normalize_job_record(jobs[i])
    return None


def pause_job(job_id: str, reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Pause a job without deleting it. Accepts a job ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return None
    return update_job(
        job["id"],
        {
            "enabled": False,
            "state": "paused",
            "paused_at": _hermes_now().isoformat(),
            "paused_reason": reason,
        },
    )


def resume_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Resume a paused job and compute the next future run from now. Accepts a job ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return None

    next_run_at = compute_next_run(job["schedule"])
    if next_run_at is None and job["schedule"].get("kind") == "once":
        run_at = job["schedule"].get("run_at", "unknown")
        raise ValueError(
            f"Cannot resume: one-shot time {run_at} is in the past "
            f"(grace window: {ONESHOT_GRACE_SECONDS}s) and will never fire."
        )
    return update_job(
        job["id"],
        {
            "enabled": True,
            "state": "scheduled",
            "paused_at": None,
            "paused_reason": None,
            "next_run_at": next_run_at,
        },
    )


def trigger_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Schedule a job to run on the next scheduler tick. Accepts a job ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return None
    return update_job(
        job["id"],
        {
            "enabled": True,
            "state": "scheduled",
            "paused_at": None,
            "paused_reason": None,
            "next_run_at": _hermes_now().isoformat(),
        },
    )


def remove_job(job_id: str) -> bool:
    """Remove a job by ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return False
    canonical_id = job["id"]
    with _jobs_lock():
        jobs = load_jobs()
        original_len = len(jobs)
        jobs = [j for j in jobs if j["id"] != canonical_id]
        if len(jobs) < original_len:
            # Resolve the output dir BEFORE saving so a legacy unsafe ID (e.g.
            # left over from before the create-time guard) fails closed without
            # half-applying the removal.
            job_output_dir = _job_output_dir(canonical_id)
            save_jobs(jobs)
            # Clean up output directory to prevent orphaned dirs accumulating
            if job_output_dir.exists():
                shutil.rmtree(job_output_dir)
            return True
    return False


def mark_job_run(job_id: str, success: bool, error: Optional[str] = None,
                 delivery_error: Optional[str] = None):
    """
    Mark a job as having been run.
    
    Updates last_run_at, last_status, increments completed count,
    computes next_run_at, and auto-deletes if repeat limit reached.

    ``delivery_error`` is tracked separately from the agent error — a job
    can succeed (agent produced output) but fail delivery (platform down).
    """
    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] == job_id:
                now = _hermes_now().isoformat()
                job["last_run_at"] = now
                job["last_status"] = "ok" if success else "error"
                job["last_error"] = error if not success else None
                # Track delivery failures separately — cleared on successful delivery
                job["last_delivery_error"] = delivery_error
                # Clear any external-fire claim so a re-armed recurring job can
                # be claimed again on its next fire (Phase 4C CAS).
                job["fire_claim"] = None
                # Clear the one-shot running-claim (#59229): the run is over, so
                # a re-armed recurring job or a re-dispatched one-shot recovery
                # is claimable again. No-op if the job never carried a claim.
                if job.get("run_claim") is not None:
                    job["run_claim"] = None
                
                # Increment completed count.  Finite one-shot jobs are
                # pre-claimed by claim_dispatch() BEFORE the side effect runs
                # (issue #38758), which already incremented completed — do not
                # double-count them here.  Recurring jobs and direct callers
                # with no pre-run claim still get the legacy increment.
                if job.get("repeat"):
                    repeat = job["repeat"]
                    times = repeat.get("times")
                    completed = repeat.get("completed", 0)
                    kind = job.get("schedule", {}).get("kind")
                    preclaimed_oneshot = (
                        kind == "once"
                        and times is not None
                        and times > 0
                        and completed > 0
                    )
                    if not preclaimed_oneshot:
                        completed += 1
                        repeat["completed"] = completed

                    # Check if we've hit the repeat limit
                    if times is not None and times > 0 and completed >= times:
                        # Limit reached: retain the record as a terminal
                        # completion instead of popping it. Deleting the job
                        # here discarded the last_status / last_error /
                        # last_delivery_error written above — a finished
                        # one-shot vanished from `cronjob list` with no
                        # inspectable outcome, and a failed delivery was
                        # invisible. Mirror the terminal shape of the
                        # next_run_at-is-None branch below; the retention
                        # sweep prunes these after
                        # COMPLETED_ONESHOT_RETENTION_DAYS.
                        job["enabled"] = False
                        job["state"] = "completed"
                        job["next_run_at"] = None
                        save_jobs(jobs)
                        return
                
                # Compute next run
                job["next_run_at"] = compute_next_run(job["schedule"], now)

                # If no next run, decide whether this is terminal completion
                # (one-shot) or a transient failure (recurring schedule couldn't
                # compute — e.g. 'croniter' missing from the runtime env).
                # Recurring jobs must NEVER be silently disabled: that turns a
                # missing runtime dep into "job completed" and the user's
                # schedule quietly goes off. See issue #16265.
                if job["next_run_at"] is None:
                    kind = job.get("schedule", {}).get("kind")
                    if kind in {"cron", "interval"}:
                        job["state"] = "error"
                        if not job.get("last_error"):
                            job["last_error"] = (
                                "Failed to compute next run for recurring "
                                "schedule (is the 'croniter' package "
                                "installed in the gateway's Python env?)"
                            )
                        logger.error(
                            "Job '%s' (%s) could not compute next_run_at; "
                            "leaving enabled and marking state=error so the "
                            "job is not silently disabled.",
                            job.get("name", job.get("id", "?")),
                            kind,
                        )
                    else:
                        job["enabled"] = False
                        job["state"] = "completed"
                elif job.get("state") != "paused":
                    job["state"] = "scheduled"

                save_jobs(jobs)
                return

        logger.warning("mark_job_run: job_id %s not found, skipping save", job_id)


def _write_wedged_oneshot_diagnostic(job: Dict[str, Any]) -> None:
    """Leave an operator-visible trace when a wedged one-shot is removed.

    A finite one-shot whose dispatch was claimed (``repeat.completed`` >=
    ``repeat.times``) but which never reached ``mark_job_run`` (``last_run_at``
    is null) was interrupted mid-run — scheduler restart, gateway kill, or a
    non-Exception escape (#73973). The recovery guards remove such jobs so
    they stop appearing due, but a silent removal leaves the user with no
    output, no error, and no job record. Write a small diagnostic file into
    the job's output directory so the removal is observable and debuggable.

    Best-effort: diagnostics must never break the removal itself.
    """
    if job.get("last_run_at") is not None:
        return  # a prior run was recorded — normal completion race, not a wedge
    try:
        repeat = job.get("repeat") or {}
        claim = job.get("run_claim") or {}
        text = (
            "# Cron job removed without producing output\n\n"
            f"- job id: {job.get('id')}\n"
            f"- name: {job.get('name')}\n"
            f"- dispatch claimed: {repeat.get('completed', '?')}/{repeat.get('times', '?')}\n"
            f"- run claimed at: {claim.get('at', 'unknown')} by {claim.get('by', 'unknown')}\n"
            f"- removed at: {_hermes_now().isoformat()}\n\n"
            "This one-shot job's dispatch was claimed, but the run never "
            "completed (`last_run_at` was never written) — the scheduler "
            "process was most likely killed or restarted mid-execution. The "
            "job has been removed to stop it re-firing; recreate it to run "
            "again.\n"
        )
        save_job_output(job.get("id", ""), text)
        logger.warning(
            "Job '%s': removed without a completed run — diagnostic written to "
            "its output directory",
            job.get("name", job.get("id", "?")),
        )
    except Exception as e:
        logger.debug(
            "Failed to write wedged-oneshot diagnostic for job %r: %s",
            job.get("id"), e,
        )


def claim_dispatch(job_id: str) -> bool:
    """Atomically claim a finite one-shot job dispatch BEFORE execution.

    Increments ``repeat.completed`` under the cross-process jobs lock and
    persists the claim immediately, so that if the tick dies mid-execution
    (gateway kill, OOM, segfault, hard-timeout) the dispatch is not lost.
    This converts finite one-shot jobs from *at-least-once* to *at-most-times*
    semantics — a job that self-destructs fires at most ``repeat.times`` times
    instead of infinitely (issue #38758).

    Returns ``True`` if the caller may proceed to run the job, ``False`` if the
    dispatch limit is already reached (in which case the stale job is removed).

    Only claims jobs with ``schedule.kind == "once"`` and ``repeat.times > 0``.
    Recurring jobs (they use ``advance_next_run``) and infinite-repeat / no-repeat
    jobs are left unchanged and always allowed to proceed.
    """
    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] != job_id:
                continue
            if job.get("schedule", {}).get("kind") != "once":
                return True  # recurring jobs use advance_next_run(), not dispatch claims
            repeat = job.get("repeat")
            if not repeat:
                return True  # no repeat limit — always dispatch
            times = repeat.get("times")
            if times is None or times <= 0:
                return True  # infinite — always dispatch
            completed = repeat.get("completed", 0)
            if completed >= times:
                # Already dispatched the max number of times.
                if job.get("last_run_at") is not None:
                    # A prior run completed normally (e.g. mark_job_run raced
                    # with this tick). Retain the terminal record — same shape
                    # as mark_job_run's repeat-limit branch — instead of
                    # deleting the job and its final status/delivery error.
                    job["enabled"] = False
                    job["state"] = "completed"
                    job["next_run_at"] = None
                    save_jobs(jobs)
                    logger.info(
                        "Job '%s': dispatch limit reached (%d/%d) — marking completed",
                        job.get("name", job.get("id", "?")),
                        completed,
                        times,
                    )
                    return False
                # A prior tick claimed the dispatch then died before the run
                # completed (#73973) — a genuinely wedged claim. Remove it so
                # it stops appearing as due, and leave an operator-visible
                # diagnostic instead of vanishing silently.
                jobs.pop(i)
                save_jobs(jobs)
                _write_wedged_oneshot_diagnostic(job)
                logger.info(
                    "Job '%s': dispatch limit reached (%d/%d) — removing",
                    job.get("name", job.get("id", "?")),
                    completed,
                    times,
                )
                return False
            # Claim this dispatch before the side effect runs.
            repeat["completed"] = completed + 1
            save_jobs(jobs)
            logger.debug(
                "Job '%s': claimed dispatch %d/%d",
                job.get("name", job.get("id", "?")),
                repeat["completed"],
                times,
            )
            return True

        logger.debug(
            "claim_dispatch: job_id %s not in store — proceeding without claim "
            "(handed-in job dict; nothing to persist a claim against)",
            job_id,
        )
        return True


def heartbeat_run_claim(job_id: str, *, expected_owner: str) -> bool:
    """Refresh a one-shot's ``run_claim`` timestamp while its run is alive.

    Called periodically from the scheduler's run monitor (#62002) so a
    legitimately long run keeps its claim fresh: an expired claim then really
    does mean "the claiming process died", and neither another process's tick
    nor this process's own next tick will re-dispatch or stale-remove the job
    while the run is in flight. mark_job_run() clears the claim on completion.

    ``expected_owner`` is the stable owner copied from the dispatched job. The
    compare-and-refresh prevents a stale runner that resumes after a long sleep
    from extending a claim another scheduler process has since taken over.

    Returns True if this owner's one-shot claim was refreshed; False when the
    job, claim, or ownership no longer matches.
    """
    with _jobs_lock():
        jobs = load_jobs()
        for job in jobs:
            if job.get("id") != job_id:
                continue
            if job.get("schedule", {}).get("kind") != "once":
                return False
            claim = job.get("run_claim")
            if not isinstance(claim, dict) or claim.get("by") != expected_owner:
                return False
            claim["at"] = _hermes_now().isoformat()
            save_jobs(jobs)
            return True
    return False


def advance_next_runs(job_ids) -> int:
    """Batch form of :func:`advance_next_run` for the due-dispatch loop.

    One ``load_jobs()`` + at most one ``save_jobs()`` for the whole due
    set, instead of one of each per job — the per-job form costs
    O(N loads + N saves) for N due jobs (~110 ms at N=50, measured), the
    batch form O(1 + 1) (~2 ms). ``job_ids`` may contain ids of one-shot
    or unknown jobs; they are skipped exactly as the per-job form skips
    them. Returns the number of jobs whose ``next_run_at`` was advanced.

    Crash semantics: the batch persists once at the end, so a crash
    mid-batch re-fires the whole set on restart (at-least-once burst)
    rather than advancing a prefix — acceptable given the sub-10ms window,
    and identical to the per-job form once the batch completes.
    """
    ids = set(job_ids)
    if not ids:
        return 0
    with _jobs_lock():
        jobs = load_jobs()
        now = _hermes_now().isoformat()
        advanced = 0
        for job in jobs:
            if job["id"] not in ids:
                continue
            kind = job.get("schedule", {}).get("kind")
            if kind not in {"cron", "interval"}:
                continue
            new_next = compute_next_run(job["schedule"], now)
            if new_next and new_next != job.get("next_run_at"):
                job["next_run_at"] = new_next
                advanced += 1
        if advanced:
            save_jobs(jobs)
        return advanced


def advance_next_run(job_id: str) -> bool:
    """Preemptively advance next_run_at for a recurring job before execution.

    Call this BEFORE run_job() so that if the process crashes mid-execution,
    the job won't re-fire on the next gateway restart.  This converts the
    scheduler from at-least-once to at-most-once for recurring jobs — missing
    one run is far better than firing dozens of times in a crash loop.

    One-shot jobs are left unchanged so they can still retry on restart.

    Returns True if next_run_at was advanced, False otherwise.
    """
    # >= 1 (not == 1): a corrupted jobs file with duplicate ids advances
    # every matching record; the wrapper still reports the advance.
    return advance_next_runs([job_id]) >= 1


def _machine_id() -> str:
    """Stable-ish identifier for claim attribution/debugging (NOT correctness).

    Uses ``HERMES_MACHINE_ID`` if set, else hostname + pid. The CAS correctness
    comes from the file lock + the fresh-claim check, not from this value.
    """
    explicit = os.getenv("HERMES_MACHINE_ID", "").strip()
    if explicit:
        return explicit
    try:
        import socket
        host = socket.gethostname()
    except Exception:
        host = "unknown"
    return f"{host}:{os.getpid()}"


def claim_job_for_fire(job_id: str, *, claim_ttl_seconds: int = 300) -> bool:
    """Atomically claim a job for a single external 'fire' (multi-machine
    at-most-once). Returns True iff THIS caller won the claim.

    Used by the external-provider fire path (``CronScheduler.fire_due``) when an
    external scheduler (Chronos) signals a job is due across N gateway replicas:
    exactly one wins. Single-machine deployments always win.

    Under the file lock: reject if the job is missing/disabled/paused. If a
    fresh claim (younger than ``claim_ttl_seconds``) already exists, lose.
    Otherwise stamp a ``fire_claim`` and, for recurring jobs, advance
    ``next_run_at`` (mirrors ``advance_next_run``'s at-most-once bump so a stale
    re-delivery for the old time can't re-fire). One-shots keep ``next_run_at``
    but the fresh ``fire_claim`` blocks a duplicate retry for the same fire.
    ``mark_job_run`` clears the claim on completion so a re-armed recurring job
    is claimable again next fire.

    The stale-claim TTL means a machine that crashed after claiming but before
    completing doesn't wedge the job forever — after the TTL another fire can
    reclaim it.
    """
    with _jobs_lock():
        jobs = load_jobs()
        for job in jobs:
            if job["id"] != job_id:
                continue
            if not job.get("enabled", True) or job.get("state") == "paused":
                return False
            now = _hermes_now()
            existing = job.get("fire_claim")
            if existing:
                try:
                    claimed_at = _ensure_aware(datetime.fromisoformat(existing["at"]))
                    # Bounded on BOTH sides (#60703): a claim stamped in the
                    # future (clock/TZ skew across a restart, or a corrupted
                    # timestamp) would otherwise have a negative age and stay
                    # "fresh" forever — the job becomes permanently unfireable
                    # and every manual `cron run` reports "already being
                    # fired". Treat future-dated claims as stale/overwritable.
                    _age = (now - claimed_at).total_seconds()
                    if 0 <= _age < claim_ttl_seconds:
                        return False  # someone holds a fresh claim
                except Exception:
                    pass  # malformed claim → overwrite
            job["fire_claim"] = {"at": now.isoformat(), "by": _machine_id()}
            kind = job.get("schedule", {}).get("kind")
            if kind in {"cron", "interval"}:
                nxt = compute_next_run(job["schedule"], now.isoformat())
                if nxt:
                    job["next_run_at"] = nxt
            save_jobs(jobs)
            return True
        return False


# Completed one-shot job records are retained in jobs.json (final status +
# delivery error stay inspectable via `cronjob list`) instead of being deleted
# at completion, then pruned by _sweep_completed_oneshots once they age out.
COMPLETED_ONESHOT_RETENTION_DAYS = 7


def _completed_oneshot_retention_days() -> float:
    """Resolve the completed one-shot retention window from config.

    ``cron.completed_retention_days`` (number, default
    ``COMPLETED_ONESHOT_RETENTION_DAYS``). A non-positive value disables the
    sweep, retaining completed one-shot records indefinitely.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        return float(
            cron_cfg.get(
                "completed_retention_days", COMPLETED_ONESHOT_RETENTION_DAYS
            )
        )
    except Exception:
        return float(COMPLETED_ONESHOT_RETENTION_DAYS)


def _sweep_completed_oneshots(raw_jobs: List[Dict[str, Any]], now: datetime) -> bool:
    """Prune terminal ``state == "completed"`` one-shot records past retention.

    Mutates *raw_jobs* in place; returns True when anything was removed (the
    caller persists). Only one-shot (``schedule.kind == "once"``) records in
    the terminal completed state are candidates; recurring jobs and non-
    terminal one-shots are never touched. Age is measured from
    ``last_run_at`` — a completed record without a parseable ``last_run_at``
    is kept (never guess a record into deletion).
    """
    retention_days = _completed_oneshot_retention_days()
    if retention_days <= 0:
        return False
    cutoff = now - timedelta(days=retention_days)
    removed = False
    for rj in list(raw_jobs):
        if not isinstance(rj, dict):
            continue
        try:
            if rj.get("state") != "completed":
                continue
            schedule = rj.get("schedule")
            kind = schedule.get("kind") if isinstance(schedule, dict) else None
            if kind != "once":
                continue
            last_run = rj.get("last_run_at")
            if not isinstance(last_run, str):
                continue
            try:
                last_run_dt = _ensure_aware(datetime.fromisoformat(last_run))
            except Exception:
                continue
            if last_run_dt >= cutoff:
                continue
            raw_jobs.remove(rj)
            removed = True
            logger.info(
                "Job '%s': pruning completed one-shot record "
                "(finished %s, retention %.1f days)",
                rj.get("name", rj.get("id", "?")),
                last_run,
                retention_days,
            )
        except Exception:
            logger.debug(
                "Retention sweep skipped malformed job record %r",
                rj.get("id", "?"),
                exc_info=True,
            )
    return removed


def _unique_jobs_by_id(
    raw_jobs: List[Any], wanted_ids: Set[str]
) -> Dict[str, Dict[str, Any]]:
    """Index only wanted IDs that occur exactly once in a raw snapshot."""
    indexed = {}
    ambiguous_ids = set()
    for job in raw_jobs:
        job_id = _valid_job_id(job)
        if job_id is None or job_id not in wanted_ids:
            continue
        if job_id in indexed:
            ambiguous_ids.add(job_id)
        else:
            indexed[job_id] = job
    for job_id in ambiguous_ids:
        indexed.pop(job_id, None)
    return indexed


def _revalidate_due_jobs(
    due: List[Dict[str, Any]],
    expected_jobs: List[Any],
    authoritative_jobs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Fail closed when a due record changed before the dispatch boundary.

    A successful save supplies the exact reconciled snapshot it published, so
    its one-shot run_claim cannot be stranded by an unnecessary second lock or
    read failure. Without a save, the commit lock makes the final read
    authoritative against cooperating publishers. Either boundary necessarily
    ends before the caller can dispatch, so a later pause remains an
    API-boundary race; closing that gap would require a durable claim for
    recurring jobs, not a broader file lock.
    """
    wanted_ids = {
        job_id for job in due if (job_id := _valid_job_id(job)) is not None
    }
    expected_by_id = _unique_jobs_by_id(expected_jobs, wanted_ids)
    if authoritative_jobs is None:
        jobs_file = _current_cron_store().jobs_file
        with _jobs_commit_lock() as validate_commit_lock:
            _snapshot, authoritative_jobs = _read_validated_jobs_snapshot(jobs_file)
            validate_commit_lock()
    authoritative_by_id = _unique_jobs_by_id(authoritative_jobs, wanted_ids)

    validated_due = []
    emitted_ids = set()
    for job in due:
        job_id = _valid_job_id(job)
        expected_job = expected_by_id.get(job_id) if job_id is not None else None
        authoritative_job = (
            authoritative_by_id.get(job_id) if job_id is not None else None
        )
        if (
            job_id is None
            or job_id in emitted_ids
            or expected_job is None
            or authoritative_job is None
            or authoritative_job != expected_job
            or not authoritative_job.get("enabled", True)
        ):
            logger.warning(
                "Skipping due cron job %r because its authoritative record "
                "changed before dispatch",
                job_id or "?",
            )
            continue
        emitted_ids.add(job_id)
        validated_due.append(job)
    return validated_due


def get_due_jobs() -> List[Dict[str, Any]]:
    """Get all jobs that are due to run now.

    For recurring jobs (cron/interval), if the scheduled time is stale (more
    than one period in the past, e.g. because the gateway was down OR because a
    long-running previous execution overran the interval), the accumulated
    missed runs are collapsed — ``next_run_at`` is fast-forwarded to the next
    future occurrence so a backlog does NOT burst-fire on restart — but the job
    still fires ONCE now. This prevents the perpetual-defer loop (#33315) where
    a job whose runtime exceeds ``interval + grace`` would be skipped forever.

    Note: firing once on catch-up flows through ``mark_job_run``, so a job with
    a ``repeat.times`` limit consumes one of its runs on that catch-up fire.
    """
    with _jobs_lock():
        return _get_due_jobs_locked()


def _get_due_jobs_locked() -> List[Dict[str, Any]]:
    """Inner implementation of get_due_jobs(); must be called with _jobs_lock held."""
    now = _hermes_now()
    raw_jobs = load_jobs()
    needs_save = False

    opaque_count = sum(not isinstance(rj, dict) for rj in raw_jobs)
    if opaque_count:
        logger.warning(
            "Skipping %d non-object cron job record(s) during due scan; "
            "preserving them unchanged in jobs.json",
            opaque_count,
        )

    # Repair missing/invalid/duplicate identities BEFORE anything keys off
    # ``job["id"]``. Direct jobs.json edits and older writers can leave an
    # id-less ``job_id`` record; duplicate IDs are equally unsafe because every
    # persistence loop updates the first matching row. Keep the first valid ID,
    # give every later collision a fresh one, and persist the unambiguous view.
    reserved_ids = {
        job_id
        for rj in raw_jobs
        if (job_id := _valid_job_id(rj)) is not None
    }
    seen_ids = set()
    for rj in raw_jobs:
        if not isinstance(rj, dict):
            continue
        job_id = _valid_job_id(rj)
        if job_id is not None and job_id not in seen_ids:
            seen_ids.add(job_id)
            continue

        legacy_id = rj.get("job_id") if job_id is None else None
        if (
            isinstance(legacy_id, str)
            and legacy_id
            and legacy_id not in reserved_ids
        ):
            replacement_id = legacy_id
        else:
            replacement_id = None
            for _attempt in range(_JOB_ID_REPAIR_MAX_ATTEMPTS):
                candidate_id = uuid.uuid4().hex[:12]
                if candidate_id not in reserved_ids:
                    replacement_id = candidate_id
                    break
            if replacement_id is None:
                raise RuntimeError(
                    "Could not generate a unique ID while repairing cron jobs; "
                    "refusing to persist ambiguous records"
                )
        if job_id is None:
            rj.pop("job_id", None)
        rj["id"] = replacement_id
        reserved_ids.add(replacement_id)
        seen_ids.add(replacement_id)
        needs_save = True

    jobs = [
        _apply_skill_fields(j)
        for j in copy.deepcopy(raw_jobs)
        if isinstance(j, dict)
    ]
    due = []

    # Normalize malformed "schedule" records (direct jobs.json edit, old writers,
    # corruption, etc.). "schedule" must be a dict; a null/string/etc. value
    # makes `schedule.get("kind")` or direct `schedule["kind"]` / ["expr"] /
    # ["minutes"] later raise and abort the entire scan *before* save_jobs().
    # Healthy jobs then lose their fast-forwarded next_run_at (exactly the
    # failure mode of the id-less job bug fixed above). Repair early at the
    # source so the rest of the tick can proceed and persist progress for
    # siblings.
    for j in jobs:
        if not isinstance(j.get("schedule"), dict):
            j["schedule"] = {}
            needs_save = True
    for rj in raw_jobs:
        if not isinstance(rj, dict):
            continue
        if not isinstance(rj.get("schedule"), dict):
            rj["schedule"] = {}
            needs_save = True

    # Normalize malformed "next_run_at" records (direct jobs.json edit,
    # corruption, migration, or buggy writer). If present but not a valid
    # ISO string, datetime.fromisoformat(next_run) later raises and aborts
    # the entire scan *before* save_jobs(). Healthy siblings then lose any
    # fast-forwarded next_run_at (same class of bug as bad "id" or "schedule").
    # Strip the bad value so the existing "no next_run_at" recovery path
    # recomputes a sane value and persists it for this job.
    for j in jobs:
        nr = j.get("next_run_at")
        if nr is not None:
            if not isinstance(nr, str):
                j.pop("next_run_at", None)
                needs_save = True
            else:
                try:
                    datetime.fromisoformat(nr)
                except Exception:
                    j.pop("next_run_at", None)
                    needs_save = True
    for rj in raw_jobs:
        if not isinstance(rj, dict):
            continue
        nr = rj.get("next_run_at")
        if nr is not None:
            if not isinstance(nr, str):
                rj.pop("next_run_at", None)
                needs_save = True
            else:
                try:
                    datetime.fromisoformat(nr)
                except Exception:
                    rj.pop("next_run_at", None)
                    needs_save = True

    # Same treatment for last_run_at (used as base in recovery / compute_next_run).
    for j in jobs:
        lr = j.get("last_run_at")
        if lr is not None and not isinstance(lr, str):
            j.pop("last_run_at", None)
            needs_save = True
        elif isinstance(lr, str):
            try:
                datetime.fromisoformat(lr)
            except Exception:
                j.pop("last_run_at", None)
                needs_save = True
    for rj in raw_jobs:
        if not isinstance(rj, dict):
            continue
        lr = rj.get("last_run_at")
        if lr is not None and not isinstance(lr, str):
            rj.pop("last_run_at", None)
            needs_save = True
        elif isinstance(lr, str):
            try:
                datetime.fromisoformat(lr)
            except Exception:
                rj.pop("last_run_at", None)
                needs_save = True

    # Resolve the one-shot running-claim stale-recovery TTL once per scan
    # (derived from HERMES_CRON_TIMEOUT). See _oneshot_run_claim_ttl_seconds.
    _run_claim_ttl = _oneshot_run_claim_ttl_seconds()

    # Retention sweep: completed one-shots are retained (so their final
    # status / delivery error stay inspectable via `cronjob list`) instead of
    # being deleted on completion, but they must not accumulate in jobs.json
    # forever. Prune terminal one-shot records older than the retention
    # window each scan.
    if _sweep_completed_oneshots(raw_jobs, now):
        needs_save = True
        jobs = [
            j
            for j in jobs
            if any(
                isinstance(rj, dict) and rj.get("id") == j.get("id")
                for rj in raw_jobs
            )
        ]

    for job in jobs:
        # Per-job containment (structural guard): one malformed or
        # unexpected job record must never abort the whole scan. The id /
        # schedule / timestamp normalizations above repair the known shapes;
        # this guard catches every FUTURE variant, degrading to "skip this
        # job this tick" so healthy siblings still run and their recovered
        # state still reaches save_jobs() below.
        try:
            if not job.get("enabled", True):
                continue

            # Cross-process running-claim guard (#59229): if another scheduler
            # process already claimed this one-shot and its run is still in flight
            # (claim younger than the TTL), skip it — do NOT re-dispatch. The
            # claim is stamped just before we return the job as due (below) and
            # cleared by mark_job_run() on completion. A claim older than the TTL
            # is treated as stale (the claiming tick died mid-run) and allowed
            # through so the job is recovered rather than wedged forever.
            existing_claim = job.get("run_claim")
            if existing_claim and job.get("schedule", {}).get("kind") == "once":
                try:
                    claimed_at = _ensure_aware(
                        datetime.fromisoformat(existing_claim["at"])
                    )
                    # 0 <= age: a future-dated claim (clock/TZ skew across a
                    # restart) must be treated as stale, not eternally fresh,
                    # or the one-shot is skipped forever (#60703).
                    _age = (now - claimed_at).total_seconds()
                    if 0 <= _age < _run_claim_ttl:
                        continue  # a fresh claim is held by an in-flight run
                except (KeyError, ValueError, TypeError):
                    pass  # malformed claim → fall through and (re)claim

            next_run = job.get("next_run_at")
            if not next_run:
                schedule = job.get("schedule", {})
                kind = schedule.get("kind")

                # One-shot jobs use a small grace window via the dedicated helper.
                recovered_next = _recoverable_oneshot_run_at(
                    schedule,
                    now,
                    last_run_at=job.get("last_run_at"),
                )
                recovery_kind = "one-shot" if recovered_next else None

                # Recurring jobs reach here only when something — typically a
                # direct jobs.json edit that bypassed add_job() — left
                # next_run_at unset.  Without this branch, such jobs are
                # silently skipped forever; recompute next_run_at from the
                # schedule so they pick up at their next scheduled tick.
                if not recovered_next and kind in {"cron", "interval"}:
                    recovered_next = compute_next_run(schedule, now.isoformat())
                    if recovered_next:
                        recovery_kind = kind

                if not recovered_next:
                    continue

                job["next_run_at"] = recovered_next
                next_run = recovered_next
                logger.info(
                    "Job '%s' had no next_run_at; recovering %s run at %s",
                    job.get("name", job.get("id", "?")),
                    recovery_kind,
                    recovered_next,
                )
                for rj in raw_jobs:
                    if isinstance(rj, dict) and rj.get("id") == job["id"]:
                        rj["next_run_at"] = recovered_next
                        needs_save = True
                        break

            raw_next_run_dt = datetime.fromisoformat(next_run)
            schedule = job.get("schedule", {})
            kind = schedule.get("kind")

            next_run_dt = _ensure_aware(raw_next_run_dt)
            # Migration repair: a cron job persists next_run_at as an absolute
            # instant, but the cron expr describes local wall-clock intent. If the
            # configured/system timezone changed after persistence, the stored
            # instant's offset no longer matches now's, and its converted time can
            # look due hours early (21:00+10 -> 13:00+02). When the stored *wall
            # clock* is still in the future, recompute from the schedule so we fire
            # at the intended local time instead of early-then-again.
            #
            # TRADE-OFF: this cannot distinguish a config/host TZ migration from a
            # legitimate DST offset change. A DST boundary that satisfies all four
            # conditions will recompute (and thus SKIP the pending occurrence, no
            # catch-up) rather than fire it. Accepted: in the pure-migration case
            # the recompute lands on the same wall-clock time later the same period,
            # and DST-boundary collisions with a still-future stored wall clock are
            # rare relative to the double-fire bug this prevents (#28934).
            if (
                kind == "cron"
                and next_run_dt <= now
                and _timezone_offset_mismatch(raw_next_run_dt, now)
                and _stored_wall_clock_is_future(raw_next_run_dt, now)
            ):
                new_next = compute_next_run(schedule, now.isoformat())
                if new_next:
                    logger.info(
                        "Job '%s' next_run_at offset changed (%s -> %s). "
                        "Recomputing cron run to preserve local wall-clock intent: %s",
                        job.get("name", job.get("id", "?")),
                        raw_next_run_dt.utcoffset(),
                        now.utcoffset(),
                        new_next,
                    )
                    for rj in raw_jobs:
                        if isinstance(rj, dict) and rj.get("id") == job["id"]:
                            rj["next_run_at"] = new_next
                            needs_save = True
                            break
                    continue

            if next_run_dt <= now:

                # For recurring jobs, check if the scheduled time is stale
                # (gateway was down and missed the window). Fast-forward to
                # the next future occurrence instead of firing a stale run.
                grace = _compute_grace_seconds(schedule)
                if kind in {"cron", "interval"} and (now - next_run_dt).total_seconds() > grace:
                    # Job is past its catch-up grace window — skip accumulated
                    # missed runs but still execute once now to avoid deferring
                    # indefinitely (e.g. a long-running job just finished).
                    new_next = compute_next_run(schedule, now.isoformat())
                    if new_next:
                        logger.info(
                            "Job '%s' missed its scheduled time (%s, grace=%ds). "
                            "Running now; next run provisionally set to: %s "
                            "(re-anchored on completion)",
                            job.get("name", job.get("id", "?")),
                            next_run,
                            grace,
                            new_next,
                        )
                        # Persist the fast-forward to storage now (skip accumulated
                        # slots). In the built-in ticker path this is shortly
                        # overwritten by advance_next_run + mark_job_run, but it is
                        # NOT redundant: it (a) protects the crash window between
                        # here and mark_job_run, and (b) covers the external
                        # fire_due provider path, which does not call
                        # advance_next_run. mark_job_run re-anchors next_run_at off
                        # the actual completion time, so this value is provisional.
                        for rj in raw_jobs:
                            if (
                                isinstance(rj, dict)
                                and rj.get("id") == job["id"]
                            ):
                                rj["next_run_at"] = new_next
                                needs_save = True
                                break
                        record_catch_up_occurrence()
                        # Fall through to due.append(job) — execute once now

                # One-shot dispatch-limit guard (issue #38758): a finite one-shot
                # claimed via claim_dispatch() but whose tick died before
                # mark_job_run could remove it will have completed >= times while
                # still looking due (last_run_at was never written, so the
                # recovery helper re-armed it). Remove it instead of re-firing.
                if kind == "once":
                    repeat = job.get("repeat")
                    if repeat:
                        times = repeat.get("times")
                        completed = repeat.get("completed", 0)
                        if times is not None and times > 0 and completed >= times:
                            # A live run must never have its job record deleted
                            # underneath it (#62002): a run that outlives the
                            # run_claim TTL (stream stall, laptop asleep
                            # mid-run) satisfies the same completed >= times +
                            # expired-claim condition as a dead tick, but
                            # mark_job_run() still needs the record to land
                            # last_run_at / last_status / last_delivery_error.
                            # If this process is still running the job, it is
                            # slow, not stale — keep the entry and skip.
                            if _job_running_in_this_process(job.get("id", "")):
                                logger.info(
                                    "Job '%s': dispatch limit reached (%d/%d) "
                                    "but its run is still in flight in this "
                                    "process — keeping entry",
                                    job.get("name", job.get("id", "?")),
                                    completed,
                                    times,
                                )
                                continue
                            logger.info(
                                "Job '%s': one-shot dispatch limit reached (%d/%d) "
                                "— removing stale due entry",
                                job.get("name", job.get("id", "?")),
                                completed,
                                times,
                            )
                            for rj in raw_jobs:
                                if (
                                    isinstance(rj, dict)
                                    and rj.get("id") == job["id"]
                                ):
                                    raw_jobs.remove(rj)
                                    needs_save = True
                                    break
                            # The claimed run never completed here by
                            # definition (last_run_at unwritten is what made
                            # the entry look due) — leave an operator-visible
                            # diagnostic instead of vanishing silently (#73973).
                            _write_wedged_oneshot_diagnostic(job)
                            continue

                # Durably claim a one-shot for the DURATION of its run before
                # returning it as due, so a second scheduler process (gateway +
                # desktop both run in-process 60s tickers on one HERMES_HOME)
                # cannot re-dispatch it while the first run is still in flight
                # (#59229). A plain one-shot's due-state is not resolved until
                # mark_job_run() completes it minutes later, so advancing
                # next_run_at by a fixed window is not enough — a job that outlives
                # one tick (e.g. a 2.5-min research prompt) would simply re-fire on
                # the next tick after the window. Instead we stamp a run_claim under
                # the same lock get_due_jobs already holds; the other process reads
                # a fresh claim on its next tick and skips (handled at the top of
                # this loop). mark_job_run() clears the claim on completion. The TTL
                # is only a safety valve: a claiming tick that DIES mid-run leaves a
                # stale claim that expires after the resolved run-claim TTL
                # (_oneshot_run_claim_ttl_seconds, derived from HERMES_CRON_TIMEOUT),
                # so the job is re-dispatched rather than wedged forever.
                if kind == "once":
                    claim = {"at": now.isoformat(), "by": _machine_id()}
                    job["run_claim"] = claim
                    for rj in raw_jobs:
                        if isinstance(rj, dict) and rj.get("id") == job["id"]:
                            rj["run_claim"] = claim
                            needs_save = True
                            break

                due.append(job)
        except Exception:
            logger.exception(
                "Skipping malformed cron job %r during due scan",
                job.get("name") or job.get("id") or "?",
            )
            continue

    authoritative_jobs = save_jobs(raw_jobs) if needs_save else None

    return (
        _revalidate_due_jobs(due, raw_jobs, authoritative_jobs)
        if due
        else due
    )


# Per-run cron output (`cron/output/<job>/<timestamp>.md`) is written once per
# execution. Unlike the quick-snapshot store (`hermes_cli.backup`, capped at 20)
# it had no retention, so a frequently-scheduled job on a long-running deploy
# accumulated one file per run forever and could fill the disk (#52383). Keep the
# most recent N files per job; a non-positive value disables pruning (opt-out).
_CRON_OUTPUT_DEFAULT_KEEP = 50


def _cron_output_keep() -> int:
    """Resolve the per-job output-file retention cap from config (``cron.output_retention``)."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        return int(cron_cfg.get("output_retention", _CRON_OUTPUT_DEFAULT_KEEP))
    except Exception:
        return _CRON_OUTPUT_DEFAULT_KEEP


def _prune_job_output(job_output_dir: Path, keep: int) -> int:
    """Remove the oldest ``*.md`` run-output files beyond *keep*. Returns count deleted.

    Mirrors the quick-snapshot retention in ``hermes_cli.backup._prune_quick_snapshots``:
    output filenames are timestamp-based (``%Y-%m-%d_%H-%M-%S.md``) so a reverse
    lexical sort orders newest-first, and everything past *keep* is the tail to
    drop. A non-positive *keep* disables pruning. Pruning failures are swallowed
    so they can never break output saving.
    """
    if keep <= 0:
        return 0
    try:
        files = sorted(
            (f for f in job_output_dir.glob("*.md") if f.is_file()),
            key=lambda f: f.name,
            reverse=True,
        )
    except OSError:
        return 0
    deleted = 0
    for stale in files[keep:]:
        try:
            stale.unlink()
            deleted += 1
        except OSError as exc:
            logger.debug("Failed to prune cron output %s: %s", stale.name, exc)
    return deleted


def save_job_output(job_id: str, output: str):
    """Save job output to file."""
    ensure_dirs()
    job_output_dir = _job_output_dir(job_id)
    job_output_dir.mkdir(parents=True, exist_ok=True)
    _secure_dir(job_output_dir)

    timestamp = _hermes_now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = job_output_dir / f"{timestamp}.md"

    fd, tmp_path = tempfile.mkstemp(dir=str(job_output_dir), suffix='.tmp', prefix='.output_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(output)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, output_file)
        _secure_file(output_file)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Bound per-job output growth so long-running deploys don't fill the disk (#52383).
    _prune_job_output(job_output_dir, _cron_output_keep())

    return output_file


# =============================================================================
# Skill reference rewriting (curator integration)
# =============================================================================

def referenced_skill_names() -> Set[str]:
    """Return the set of skill names referenced by ANY cron job.

    Includes paused and disabled jobs deliberately: a paused job never
    fires, so its skills never get a ``bump_use`` from the scheduler, yet
    resuming it must still find its skills present. The curator uses this
    set to protect referenced skills from inactivity archival — a skill a
    live job depends on is "in use" regardless of when it was last loaded.

    Best-effort: a corrupt/unreadable jobs store returns an empty set
    rather than raising, so a cron issue can never break the curator.
    """
    try:
        jobs = load_jobs()
    except Exception:
        logger.debug("referenced_skill_names: failed to load cron jobs", exc_info=True)
        return set()

    names: Set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            continue
        for name in _normalize_skill_list(job.get("skill"), job.get("skills")):
            cleaned = str(name).strip().lstrip("/")
            if cleaned:
                names.add(cleaned)
    return names


def rewrite_skill_refs(
    consolidated: Optional[Dict[str, str]] = None,
    pruned: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Rewrite cron job skill references after a curator consolidation pass.

    When the curator consolidates a skill X into umbrella Y (or archives X
    as pruned), any cron job that lists ``X`` in its ``skills`` field will
    fail to load ``X`` at run time — the scheduler logs a warning and
    skips the skill, so the job runs without the instructions it was
    scheduled to follow. See cron/scheduler.py where ``skill_view`` is
    called per skill name.

    This function repairs cron jobs in-place:

    - A skill listed in ``consolidated`` is replaced with its umbrella
      target (the ``into`` value). If the umbrella is already in the
      job's skill list, the stale name is dropped without duplication.
    - A skill listed in ``pruned`` is dropped outright — there is no
      forwarding target.
    - Ordering and other skills in the list are preserved.
    - The legacy ``skill`` field is realigned via ``_apply_skill_fields``.

    Args:
        consolidated: mapping of ``old_skill_name -> umbrella_skill_name``.
        pruned: list of skill names that were archived with no forwarding
            target.

    Returns a report dict::

        {
            "rewrites": [
                {
                    "job_id": ...,
                    "job_name": ...,
                    "before": [...],
                    "after": [...],
                    "mapped": {"old": "new", ...},
                    "dropped": ["old", ...],
                },
                ...
            ],
            "jobs_updated": N,
            "jobs_scanned": M,
        }

    Best-effort: exceptions from loading/saving propagate to the caller so
    tests can assert behaviour; the curator invocation site wraps this
    call in a try/except so a failure here never breaks the curator.
    """
    consolidated = dict(consolidated or {})
    pruned_set = set(pruned or [])
    # A skill listed in both wins as "consolidated" — it has a target,
    # which is the more useful of the two outcomes.
    pruned_set -= set(consolidated.keys())

    if not consolidated and not pruned_set:
        return {"rewrites": [], "jobs_updated": 0, "jobs_scanned": 0}

    with _jobs_lock():
        jobs = load_jobs()
        rewrites: List[Dict[str, Any]] = []
        changed = False

        for job in jobs:
            skills_before = _normalize_skill_list(job.get("skill"), job.get("skills"))
            if not skills_before:
                continue

            mapped: Dict[str, str] = {}
            dropped: List[str] = []
            new_skills: List[str] = []

            for name in skills_before:
                if name in consolidated:
                    target = consolidated[name]
                    mapped[name] = target
                    if target and target not in new_skills:
                        new_skills.append(target)
                elif name in pruned_set:
                    dropped.append(name)
                elif name not in new_skills:
                    new_skills.append(name)

            if not mapped and not dropped:
                continue

            job["skills"] = new_skills
            job["skill"] = new_skills[0] if new_skills else None
            changed = True

            rewrites.append({
                "job_id": job.get("id"),
                "job_name": job.get("name") or job.get("id"),
                "before": list(skills_before),
                "after": list(new_skills),
                "mapped": mapped,
                "dropped": dropped,
            })

        if changed:
            save_jobs(jobs)
            logger.info(
                "Curator rewrote skill references in %d cron job(s)", len(rewrites)
            )

        return {
            "rewrites": rewrites,
            "jobs_updated": len(rewrites),
            "jobs_scanned": len(jobs),
        }
