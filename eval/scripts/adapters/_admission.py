#!/usr/bin/env python3
"""Admission control by KV footprint.

vLLM admits a request on the size it has now, not the size it will reach. A
GPQA item arrives at 275 prompt tokens -- trivially admissible -- and grows to
52k; 280 of them are all admitted, then grow together into a full cache and the
scheduler starts preempting. The eventual size is knowable only from a previous
run's distribution, which the client has and the server does not, so the
reservation is made here.
"""

import hashlib
import json
import os
import socket
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Matches _common.reasoning_tokens, which infers token counts the same way. The
# estimate only has to be right to within the margin the controller corrects.
CHARS_PER_TOKEN = 4

# One request at the served context length. Backing off below this would make
# the longest items serialize against each other, which costs more wall clock
# than the preemption the backoff is avoiding.
FLOOR_TOKENS = 262_144
# One 128k item's worth of headroom per adjustment window: the increment is
# additive so that recovery is slow next to the multiplicative backoff, which
# is what keeps the loop from oscillating around the preemption threshold.
GROWTH_TOKENS = 131_072
BACKOFF = 0.8
# Grow only against a real queue. A few waiting requests are the scheduler
# working normally; a budget raised while nothing waits admits nothing new and
# only removes the margin that absorbs the next underestimate.
WAITING_TARGET = 8


def next_capacity(
    capacity: int,
    *,
    preempted: int,
    waiting: float,
    ceiling: int,
    floor: int = FLOOR_TOKENS,
    held: int = 0,
) -> int:
    """AIMD on the one signal that means the reservation was wrong.

    Preemption is the only observable that distinguishes "the cache is working"
    from "the cache is thrashing"; queue depth alone cannot, because a deep
    queue is the intended state when items outnumber capacity.

    `held` decides whether a cut is worth making, not how deep it goes. A
    resize is not a revocation, so once held is above capacity admission is
    already shut and a further cut un-admits nobody: it only deepens an
    overdraft that completions alone can clear. Backing off from 5.9M to the
    floor while 4.6M was held bought nothing and cost three hours.

    Bounding every cut by `held` instead of only the overdrawn ones put the
    brake out of service at full load, which is the one load that needs it. A
    budget that is merely full has held a hair under capacity, so the bound
    returned capacity unchanged and the preemption counter climbed against a
    budget that never moved -- 5,990 preemptions at 5,882,813 held of 5,883,095
    before the engine stopped answering. Cut when the budget is what admits
    requests; hold still only when a cut cannot reach them.
    """
    if preempted > 0:
        if held >= capacity:
            return capacity
        return max(min(floor, ceiling), int(capacity * BACKOFF))
    if waiting > WAITING_TARGET:
        return min(ceiling, capacity + GROWTH_TOKENS)
    return capacity


def _entry(value: Any, where: str) -> dict[str, int]:
    if not isinstance(value, dict) or "prompt" not in value or "output" not in value:
        raise ValueError(f"{where}: each prior needs a 'prompt' and an 'output'")
    return {"prompt": int(value["prompt"]), "output": int(value["output"])}


def load_priors(path: Path) -> dict[str, Any]:
    """Per-suite prompt and output length, measured rather than guessed."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    suites = data.get("suites")
    if not isinstance(suites, dict) or "default" not in data:
        raise ValueError(f"{path}: needs a 'suites' object and a 'default'")
    return {
        "suites": {str(k): _entry(v, f"{path}:{k}") for k, v in suites.items()},
        "default": _entry(data["default"], f"{path}:default"),
    }


def reservation(text: str, suite: str, priors: dict[str, Any], *, max_tokens: int) -> int:
    """KV tokens this request will occupy at its peak.

    The prompt is the larger of what the text implies and what the suite
    measured last time. Character counting is exact enough for a text prompt
    and blind to an image one: a multimodal item whose text is a one-line
    caption carries a prompt of 869 tokens at p50 and 4078 at p90, all of it in
    pixels. Taking the maximum keeps RULER priced per item -- its lengths span
    4k to 128k, so a median would misprice two thirds of it -- while giving the
    image suites a floor that character counting cannot see.

    The output half is the suite's median. Reserving p90 would idle five sixths
    of the pool on GPQA, whose output spans 8k at p50 and 52k at p90; the
    controller corrects the tail from the preemption counter instead.
    """
    entry = priors["suites"].get(suite) or priors["default"]
    prompt = max(len(text) // CHARS_PER_TOKEN, int(entry["prompt"]))
    expected = int(entry["output"])
    # A prior above the cap describes a request the server will not produce.
    if max_tokens > 0:
        expected = min(expected, max_tokens)
    return prompt + expected


class TokenBudget:
    """A semaphore counted in KV tokens rather than in requests."""

    def __init__(
        self,
        capacity: int,
        *,
        max_hold_seconds: float | None = None,
        poll_seconds: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.capacity = int(capacity)
        self.max_hold_seconds = max_hold_seconds
        self._held: dict[str, tuple[int, float]] = {}
        self._waiters = 0
        self._poll = poll_seconds
        self._clock = clock
        self._condition = threading.Condition()

    def _total(self) -> int:
        return sum(tokens for tokens, _ in self._held.values())

    def _expire(self) -> None:
        """Return the tokens of holds that have outlived the cap.

        A request keeps its reservation across retries, so a server that stops
        answering turns the tokens in flight into a permanent deduction from a
        budget nothing will ever give back. The cap is a safety valve set past
        the longest request the protocol allows, not a routine path -- when it
        fires, the budget is briefly optimistic about a request the server may
        still be running, which is the lesser of the two failures.
        """
        if not self.max_hold_seconds:
            return
        cutoff = self._clock() - self.max_hold_seconds
        stale = [key for key, (_, since) in self._held.items() if since <= cutoff]
        for key in stale:
            del self._held[key]
        if stale:
            self._condition.notify_all()

    def outstanding(self) -> int:
        with self._condition:
            self._expire()
            return self._total()

    def waiting(self) -> int:
        """How many lanes are blocked here.

        This is the queue the controller cannot see anywhere else: a request
        held at the budget has not been sent, so it is absent from the server's
        own queue depth exactly when the budget is what is holding it up.
        """
        with self._condition:
            return self._waiters

    def resize(self, capacity: int) -> None:
        """Change what may be admitted next; never revoke what is already held.

        Revoking would mean cancelling a request mid-generation, which discards
        exactly the work the backoff is trying to stop wasting.
        """
        with self._condition:
            self.capacity = max(1, int(capacity))
            self._condition.notify_all()

    def acquire(self, key: str, tokens: int) -> int:
        """Block until `tokens` fit, then hold them under `key`.

        A reservation larger than the whole budget is clamped to it rather than
        refused: it then waits for an empty budget and runs alone. Refusing it
        would hang the suite on capacity that cannot appear, and admitting it
        unreserved would let the cache overfill by exactly the amount the
        estimate was wrong by.

        The clamp is applied against the capacity that admits the request, not
        the one in force when it arrived. Clamping once and then waiting left a
        request that arrived under a large budget asking for more than the
        whole budget it was now waiting on -- unsatisfiable however empty the
        budget got, while the same request made a second later would run.
        """
        want = max(1, int(tokens))
        with self._condition:
            self._waiters += 1
            try:
                while True:
                    self._expire()
                    grant = min(want, self.capacity)
                    if self._total() + grant <= self.capacity:
                        self._held[key] = (grant, self._clock())
                        return grant
                    # Timed, so an expiry that nobody notifies us about still
                    # gets noticed by whoever is waiting on it.
                    self._condition.wait(self._poll)
            finally:
                self._waiters -= 1

    def release(self, key: str) -> None:
        with self._condition:
            self._held.pop(key, None)
            self._condition.notify_all()


class AdmissionError(RuntimeError):
    pass


# sockaddr_un.sun_path is 104 bytes on macOS and 108 on Linux, including the
# terminator. The natural home for the socket is the run directory, and those
# are already close to it: the current RUN_BASE puts a per-variant socket at 88
# characters, so a longer campaign name would fail at bind with an OSError that
# names no path.
SUN_PATH_LIMIT = 100


def socket_path(path: Any) -> Path:
    """The path to actually bind, given the one the caller wants.

    Substitutes a short path under the system temp directory when the requested
    one will not fit. Derived from the full path by hash, so the broker and every
    lane arrive at the same answer without having to agree on it.
    """
    requested = Path(path)
    if len(str(requested).encode()) <= SUN_PATH_LIMIT:
        return requested
    digest = hashlib.sha256(str(requested).encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"admission-{digest}.sock"


class Broker:
    """One budget for every lane process sharing a server.

    Lanes are separate processes, so a shared budget needs somewhere to live.
    Handing each lane a fixed share instead was the previous design and it
    cannot recover: the share is set when the lane starts, so a lane that
    finishes leaves its capacity stranded while its siblings queue.

    Each connection gets its own thread, so one lane blocking on capacity never
    blocks another's release. A connection's reservations are released when it
    closes, which is what keeps a killed lane from ratcheting the budget down.
    """

    def __init__(self, path: Path, capacity: int, max_hold_seconds: float | None = None):
        self.path = socket_path(path)
        self.budget = TokenBudget(capacity, max_hold_seconds=max_hold_seconds)
        self._connections = 0
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.path))
        self._server.listen(128)
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def outstanding(self) -> int:
        return self.budget.outstanding()

    def waiting(self) -> int:
        return self.budget.waiting()

    def resize(self, capacity: int) -> None:
        self.budget.resize(capacity)

    def _accept_loop(self) -> None:
        while True:
            try:
                connection, _ = self._server.accept()
            except OSError:
                return
            with self._lock:
                self._connections += 1
                connection_id = self._connections
            thread = threading.Thread(
                target=self._serve_connection, args=(connection, connection_id), daemon=True
            )
            thread.start()

    def _serve_connection(self, connection: socket.socket, connection_id: int) -> None:
        # Reservations are namespaced per connection: two lanes scoring
        # different suites can hand us the same item id.
        held: set[str] = set()
        try:
            with connection.makefile("rwb") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    reply = self._dispatch(json.loads(line), connection_id, held)
                    stream.write((json.dumps(reply) + "\n").encode())
                    stream.flush()
        except (OSError, ValueError):
            pass
        finally:
            for key in held:
                self.budget.release(key)
            connection.close()

    def _dispatch(self, request: dict[str, Any], connection_id: int, held: set[str]) -> dict:
        op = request.get("op")
        key = f"{connection_id}:{request.get('key')}"
        if op == "acquire":
            granted = self.budget.acquire(key, int(request.get("tokens", 1)))
            held.add(key)
            return {"granted": granted}
        if op == "release":
            self.budget.release(key)
            held.discard(key)
            return {"ok": True}
        if op == "stat":
            return {
                "capacity": self.budget.capacity,
                "outstanding": self.budget.outstanding(),
                "waiting": self.budget.waiting(),
            }
        return {"error": f"unknown op {op!r}"}

    def stop(self) -> None:
        try:
            self._server.close()
        finally:
            try:
                os.unlink(self.path)
            except OSError:
                pass


def serve(path: Path, capacity: int, max_hold_seconds: float | None = None) -> Broker:
    return Broker(Path(path), capacity, max_hold_seconds)


class RemoteBudget:
    """A lane's view of the shared budget.

    One connection per thread, deliberately. A single shared connection would
    deadlock the moment one worker blocked waiting for capacity, because the
    peer that could free it would be queued behind that same socket.
    """

    def __init__(self, path: Path):
        self.path = socket_path(path)
        self._local = threading.local()
        self._sockets: list[socket.socket] = []
        self._lock = threading.Lock()

    def _stream(self):
        stream = getattr(self._local, "stream", None)
        if stream is None:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.connect(str(self.path))
            stream = connection.makefile("rwb")
            self._local.stream = stream
            with self._lock:
                self._sockets.append(connection)
        return stream

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        stream = self._stream()
        stream.write((json.dumps(request) + "\n").encode())
        stream.flush()
        line = stream.readline()
        if not line:
            raise AdmissionError(f"admission broker at {self.path} closed the connection")
        return json.loads(line)

    def acquire(self, key: str, tokens: int) -> int:
        return int(self._call({"op": "acquire", "key": key, "tokens": int(tokens)})["granted"])

    def release(self, key: str) -> None:
        self._call({"op": "release", "key": key})

    def close(self) -> None:
        with self._lock:
            for connection in self._sockets:
                try:
                    connection.close()
                except OSError:
                    pass
            self._sockets.clear()
        self._local = threading.local()


def from_environment(env: dict[str, str]) -> Any:
    """The budget this process should reserve against, or None for no control.

    Absent configuration means an adapter run by hand against a dev server,
    where there is nothing to share and nothing to protect.
    """
    socket_path = (env.get("EVAL_ADMISSION_SOCKET") or "").strip()
    if socket_path:
        return RemoteBudget(Path(socket_path))
    tokens = (env.get("EVAL_ADMISSION_TOKENS") or "").strip()
    if tokens:
        return TokenBudget(int(tokens))
    return None
