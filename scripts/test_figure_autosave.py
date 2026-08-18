"""Exercise figure_autosave inside a real IPython shell with the inline backend.

Run:  python scripts/test_figure_autosave.py     (exit 0 = all checks passed)


The whole point of the module is hook ordering against matplotlib-inline's
flush_figures, so a mock would prove nothing. This drives the actual shell.
"""
import shutil, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebook_modules"))

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.clear_instance()
ip = InteractiveShell.instance()
# A bare InteractiveShell has no GUI event loop, so %matplotlib raises after it
# has already configured the inline backend -- which is the only part this test
# needs. Silence the traceback rather than let it drown the results.
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    ip.run_cell("%matplotlib inline")

OUT = Path(tempfile.mkdtemp(prefix="fa_"))
fails = []
def check(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond: fails.append(name)

ip.run_cell(f"""
import figure_autosave as fa
fa.enable({str(OUT)!r}, verbose=False)
""")
check("enable() ran", "Traceback" not in (ip.user_ns.get("_", "") or ""))

# --- 1. a figure that is never shown: caught by the post_execute hook ---------
ip.run_cell("""
import matplotlib.pyplot as plt
fa.set_run("run_alpha.csv", directory=%r)
fig, ax = plt.subplots(); ax.plot([0,1],[0,1]); ax.set_ylabel("peak shift (kHz)")
""" % str(OUT))
p = OUT / "run_alpha_result.png"
check("unshown figure saved by post_execute hook", p.exists(), p.name)

# --- 2. plt.show() mid-cell: caught by the show wrapper ----------------------
ip.run_cell("""
fa.set_run("run_beta.csv", directory=%r)
fig, ax = plt.subplots(); ax.plot([0,1],[1,0]); ax.set_ylabel("nT")
plt.show()
fig2, ax2 = plt.subplots(); ax2.loglog([1,10],[1,0.1]); ax2.set_ylabel("nT / sqrt(Hz)")
""" % str(OUT))
check("figure closed by plt.show() still saved", (OUT / "run_beta_result.png").exists())
check("pure log-log spectrum classified as _fft", (OUT / "run_beta_fft.png").exists())

# --- 3. mixed figure (time series + ASD panel) must NOT claim _fft ------------
ip.run_cell("""
fa.set_run("run_gamma.csv", directory=%r)
fig, axes = plt.subplots(3,1)
axes[0].plot([0,1],[0,1]); axes[0].set_ylabel("peak shift (kHz)")
axes[1].plot([0,1],[1,0]); axes[1].set_ylabel("normalised PL (z)")
axes[2].loglog([1,10],[1,0.1]); axes[2].set_ylabel("nT / sqrt(Hz)")
plt.show()
""" % str(OUT))
check("3-panel figure with one ASD panel -> _result", (OUT / "run_gamma_result.png").exists())
check("...and does NOT take the _fft name", not (OUT / "run_gamma_fft.png").exists())

# --- 4. numbering, tags, idempotence -----------------------------------------
ip.run_cell("""
fa.set_run("run_delta.csv", tag="run", directory=%r)
for _ in range(3):
    f, a = plt.subplots(); a.plot([0,1],[0,1]); a.set_ylabel("kHz")
plt.show()
""" % str(OUT))
check("tag lands in the filename", (OUT / "run_delta_run_result.png").exists())
check("second figure numbered", (OUT / "run_delta_run_result2.png").exists())
check("third figure numbered", (OUT / "run_delta_run_result3.png").exists())

before = sorted(q.name for q in OUT.glob("run_delta*"))
ip.run_cell("""
fa.set_run("run_delta.csv", tag="run", directory=%r)
for _ in range(3):
    f, a = plt.subplots(); a.plot([0,1],[0,1]); a.set_ylabel("kHz")
plt.show()
""" % str(OUT))
after = sorted(q.name for q in OUT.glob("run_delta*"))
check("re-running a cell overwrites, does not accumulate", before == after,
      f"{len(before)} -> {len(after)}")

# --- 5. no double-write when both hooks see the same figure ------------------
ip.run_cell("""
fa.set_run("run_eps.csv", directory=%r)
f, a = plt.subplots(); a.plot([0,1],[0,1]); a.set_ylabel("kHz")
plt.show(); plt.show()
""" % str(OUT))
check("a figure is written exactly once",
      sorted(q.name for q in OUT.glob("run_eps*")) == ["run_eps_result.png"])

# --- 6. inline display still works (flush_figures was not lost) --------------
res = ip.run_cell("""
fa.set_run("run_zeta.csv", directory=%r)
f, a = plt.subplots(); a.plot([0,1],[0,1]); a.set_ylabel("kHz")
""" % str(OUT))
from matplotlib_inline.backend_inline import flush_figures
import figure_autosave as fa_local
cbs = ip.events.callbacks["post_execute"]
check("flush_figures still registered", flush_figures in cbs)
check("flush_figures runs LAST", cbs[-1] is flush_figures,
      " -> ".join(getattr(c, "__name__", str(c)) for c in cbs))
check("our hook runs before flush_figures",
      cbs.index(fa_local._on_post_execute) < cbs.index(flush_figures))

# --- 7. empty figures are skipped, disable() stops writing -------------------
ip.run_cell("""
fa.set_run("run_eta.csv", directory=%r)
plt.figure()
""" % str(OUT))
check("empty figure not written", not (OUT / "run_eta_result.png").exists())

ip.run_cell("""
fa.disable()
fa.set_run("run_theta.csv", directory=%r)
f, a = plt.subplots(); a.plot([0,1],[0,1])
plt.show()
""" % str(OUT))
check("disable() stops writing", not any(OUT.glob("run_theta*")))

ip.run_cell("fa.enable(verbose=False)")
ip.run_cell("""
fa.set_run("run_iota.csv", directory=%r)
f, a = plt.subplots(); a.plot([0,1],[0,1])
plt.show()
""" % str(OUT))
check("re-enable() works", (OUT / "run_iota_result.png").exists())

# --- 8. explicit label overrides the classifier ------------------------------
ip.run_cell("""
fa.set_run("run_kappa.csv", directory=%r)
f, a = plt.subplots(); a.plot([0,1],[0,1]); f.set_label("PL droop")
plt.show()
""" % str(OUT))
check("fig.set_label() names the file", (OUT / "run_kappa_pl_droop.png").exists())

print()
print(f"{len(fails)} failed" if fails else "all checks passed")
print("files:", sorted(q.name for q in OUT.iterdir()))
shutil.rmtree(OUT, ignore_errors=True)
sys.exit(1 if fails else 0)
