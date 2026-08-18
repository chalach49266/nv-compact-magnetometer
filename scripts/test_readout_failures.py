"""The three ways this readout failed at the rig, reproduced and fixed.

STATUS: this documents an OPEN board behaviour, not a fix. The acquisition path
was reverted on 2026-08-18 to revision 8fd303f -- the one the second machine takes
good data with -- and that revision, like the one before it, does not clear the
counter. The clear_tproc_counter() call that removes the effect was reverted along
with everything else on that path. Keep this test as the evidence for what happens
when the counter is dirty, and as the ready-made check if the fix is revisited.


The 2026-08-18 failure was

    ValueError: could not broadcast input array from shape (4000,2) into shape (200,2)

raised from _drain_readout on a 100-rep priming acquire. It is not a poll or a
buffer problem: the tProc shot counter still held 2000 from the previous run, so
the board-side streamer worker computed `newshots = 2000 - 0` on its very first
look and transferred a whole previous run's buffer into a 100-rep array.

qick's own AcquireProgram.start_round() clears that counter one line after
reload_mem(); the vendored qickdawg acquire() reproduced every other step of the
sequence and dropped that one.

This drives qick's REAL DataStreamer._run_readout against a fake soc whose
counter persists across runs, because the arithmetic under test lives in that
worker, not in our code. A mock of the worker would prove nothing.

Run:  python scripts/test_shot_counter.py      (exit 0 = the mechanism reproduces)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Ahead of site-packages, so `import qickdawg` is the vendored copy the notebooks
# run and not the editable install elsewhere on this machine.
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "notebook_modules"))

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(condition)


# --------------------------------------------------------------------------- #
# A board whose shot counter survives between runs, which is the real behaviour:
# the counter is tProc data memory, the program only ever increments it, and
# stop_tproc(lazy=True) is a documented no-op on tProc v1.
# --------------------------------------------------------------------------- #

class FakeSoc:
    """Counter persists between runs; the program ramps it rather than jumping.

    The ramp matters. A fixture whose program completes the instant start_tproc()
    is called cannot tell "the worker transferred a stale buffer" from "the worker
    transferred a finished run", because both look complete on the first poll. Here
    get_tproc_counter() returns the value the board would have RIGHT NOW and only
    then advances, so the worker's first look sees the program still at zero
    progress -- which is the situation the missing clear leaves it in.
    """

    AVG_MAXLEN = 8192

    def __init__(self, counter=0):
        self.counter = counter          # survives runs unless cleared
        self.produced = 0               # shots this run's program has actually taken
        self.reads = []                 # (address, length, produced_at_the_time)
        self.clear_calls = 0
        self._program_shots = 0
        self._remaining = 0
        self._step = 1

    # -- the calls qick's worker makes on the soc -------------------------- #
    def __getitem__(self, key):
        assert key == "readouts"
        return {0: {"avg_maxlen": self.AVG_MAXLEN}, 1: {"avg_maxlen": self.AVG_MAXLEN}}

    def start_tproc(self):
        self.produced = 0
        self._remaining = self._program_shots
        self._step = max(1, self._program_shots // 10)

    def get_tproc_counter(self, addr):
        now = self.counter
        if self._remaining > 0:                      # the program advances it
            step = min(self._step, self._remaining)
            self.counter += step
            self.produced += step
            self._remaining -= step
        return now

    def get_accumulated(self, ch, address, length):
        self.reads.append((address, length, self.produced))
        return np.full((length, 2), 1, dtype=np.int64)

    def start_src(self, src):
        pass

    # -- what acquire() is supposed to call before arming ------------------ #
    def clear_tproc_counter(self, addr):
        self.clear_calls += 1
        self.counter = 0


def run_one_job(soc, total_shots, clear_first, timeout_s=3.0):
    """Submit one job to qick's real DataStreamer and collect its packets."""
    from qick.streamer import DataStreamer

    soc._program_shots = total_shots
    if clear_first:
        soc.clear_tproc_counter(addr=1)
    soc.reads.clear()

    streamer = DataStreamer(soc)
    streamer.total_count = total_shots
    streamer.count = 0
    streamer.done_flag.clear()
    streamer.job_queue.put((total_shots, 1, [0], [2], None))

    packets, deadline = [], time.time() + timeout_s
    while time.time() < deadline:
        if not streamer.data_queue.empty():
            n, (buf, _stats) = streamer.data_queue.get()
            packets.append((n, buf))
            if sum(p[0] for p in packets) >= total_shots:
                break
        elif streamer.done_flag.is_set() and streamer.data_queue.empty():
            break
        else:
            time.sleep(0.005)
    streamer.stop_readout()
    return packets


# --------------------------------------------------------------------------- #
# 1. Reproduce the failure, then show the fix removes it
# --------------------------------------------------------------------------- #

def test_oversized_packet():
    print("\n1. stale counter LARGER than the run  (the 2026-08-18 ValueError)")

    # 2000 left over from the previous auto-zero; this run primes with 100.
    soc = FakeSoc(counter=2000)
    packets = run_one_job(soc, total_shots=100, clear_first=False)
    biggest = max((n for n, _ in packets), default=0)
    check("without the clear, the worker emits a packet bigger than the run",
          biggest > 100, f"first packet = {biggest} shots into a 100-shot run")
    check("its size is exactly the stale counter value",
          biggest == 2000, f"got {biggest}")
    rows = biggest * 2          # reads_per_shot = 2
    check("which is the reported (4000,2) into (200,2)",
          (rows, 100 * 2) == (4000, 200), f"({rows},2) into ({100*2},2)")

    soc = FakeSoc(counter=2000)
    packets = run_one_job(soc, total_shots=100, clear_first=True)
    total = sum(n for n, _ in packets)
    check("with the clear, no packet exceeds the run",
          all(n <= 100 for n, _ in packets), f"sizes {[n for n, _ in packets]}")
    check("with the clear, exactly the right number of shots arrive",
          total == 100, f"got {total}")


def test_silent_replay():
    print("\n2. stale counter EQUAL to the run  (silent replay, no exception)")

    soc = FakeSoc(counter=500)
    packets = run_one_job(soc, total_shots=500, clear_first=False)
    total = sum(n for n, _ in packets)
    check("without the clear, the run 'completes' with the right shot count",
          total == 500, f"got {total} -- no error is raised at all")
    # Third field of each read is how many shots the program had actually taken
    # at that moment. The first transfer claims 500 shots the program has not run.
    addr, length, produced = soc.reads[0]
    check("but the data was read before the program produced it",
          produced < 500 and length == 1000,
          f"first read: {length} values at address {addr}, program had produced "
          f"{produced} of 500 shots")
    check("in a single instant transfer, not a stride at a time",
          len(soc.reads) == 1, f"{len(soc.reads)} transfers")

    soc = FakeSoc(counter=500)
    packets = run_one_job(soc, total_shots=500, clear_first=True)
    check("with the clear, the transfer is strided as designed",
          len(soc.reads) > 1, f"{len(soc.reads)} transfers")
    check("with the clear, the shot count is still exactly right",
          sum(n for n, _ in packets) == 500)


def test_counter_accumulates():
    print("\n3. why it compounds: nothing else ever resets the counter")

    # What each run INHERITS is the thing that matters, so record it on the way in.
    soc, inherited = FakeSoc(counter=0), []
    for _ in range(3):
        inherited.append(soc.counter)
        run_one_job(soc, total_shots=200, clear_first=False)
    check("without the clear, every run after the first inherits a dirty counter",
          inherited[0] == 0 and all(v > 0 for v in inherited[1:]),
          f"counter entering each run: {inherited}")
    check("and the value is not even predictable",
          len(set(inherited[1:])) == len(inherited[1:]) or inherited[1] != 200,
          f"{inherited} -- a run that exits early on stale data does not finish "
          "counting, so the next value is not a multiple of the rep count")

    soc, inherited = FakeSoc(counter=0), []
    for _ in range(3):
        soc.clear_tproc_counter(addr=1)
        inherited.append(soc.counter)
        run_one_job(soc, total_shots=200, clear_first=True)
    check("with the clear, every run starts from zero",
          inherited == [0, 0, 0], f"counter entering each run: {inherited}")


def main():
    print(__doc__.splitlines()[0])
    test_oversized_packet()
    test_silent_replay()
    test_counter_accumulates()

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
