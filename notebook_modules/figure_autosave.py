"""Write every notebook figure to disk, next to the run it describes.

Why this exists
---------------
Every acquisition and analysis cell in ``Twopoint_Lockin_module.ipynb`` and
``Lockin_module.ipynb`` draws figures, and none of them were written to disk: the
plots lived only in the notebook's cell output, which is lost on "Restart & Clear
Outputs" and cannot be opened side by side with the CSV it came from. The PNG
pairs already sitting in ``data/results/`` (``*_result.png`` / ``*_fft.png``) were
produced by a separate tool, not by these notebooks.

How it works
------------
``enable()`` installs two hooks, because there are two different moments at which
the inline backend destroys a figure:

* an IPython ``post_execute`` callback, registered **ahead of** matplotlib-inline's
  ``flush_figures`` -- that is the callback which displays and then closes every
  open figure at the end of a cell. IPython fires ``post_execute`` callbacks in
  registration order, so ``enable()`` registers its own and then moves
  ``flush_figures`` to the back of the queue.
* a wrapper around ``pyplot.show``, because ``show()`` on the inline backend also
  closes what it displays, and several helpers call it mid-cell
  (``twopoint_runner.plot_run``, ``TwoPointResult.figure(show=True)``, the live
  loops in ``Lockin_module``).

Both funnel into :func:`capture`. A figure is written at most once -- the marker
lives on the figure object, so neither hook can duplicate the other's work.

Naming
------
::

    <stem>_result.png        first figure of the run
    <stem>_result2.png       second, third, ...
    <stem>_fft.png           a figure whose axes are ALL log-log spectra
    <stem>_<tag>_result.png  when set_run(..., tag="run") names the stage

``<stem>`` is the run's CSV stem, so the PNG sorts next to its data. Each
``set_run()`` restarts the numbering, which means re-running a cell overwrites
that cell's own PNGs instead of accumulating copies.

A figure can override the classification with ``fig.set_label("droop")``, giving
``<stem>_droop.png``.

Usage
-----
::

    import figure_autosave
    figure_autosave.enable(TWOPOINT_DIR)       # once, in the imports cell
    ...
    figure_autosave.set_run(AVG_CSV, tag="run")   # acquisition cell
    figure_autosave.set_run(A5_CSV)               # analysis cell

Outside IPython (scripts, pytest) the hooks are not installed and nothing is
patched; call :func:`capture` directly instead.
"""

from __future__ import annotations

import builtins
import functools
import re
from pathlib import Path

__all__ = ["enable", "disable", "set_run", "set_directory", "capture",
           "status", "is_enabled"]


# --------------------------------------------------------------------------- #
# State
#
# Held on `builtins` rather than at module level because these notebooks run
# `%autoreload 2`, which re-executes this module's body whenever the file is
# touched. Module globals would be reset by that -- and the hooks registered
# with IPython would then be pointing at a state dict nothing else can see.
# --------------------------------------------------------------------------- #

_STATE = getattr(builtins, "_NV_FIGURE_AUTOSAVE", None)
if _STATE is None:
    _STATE = {
        "enabled": False,
        "directory": None,   # Path -- where PNGs go
        "stem": None,        # str  -- basename shared with the run's CSV
        "tag": None,         # str  -- optional stage marker ("run", "4S", ...)
        "counts": {},        # kind -> how many of that kind this run has written
        "dpi": 150,
        "verbose": True,
        "written": [],       # paths written since the last set_run(), for status()
        "warned": False,     # a savefig failure is reported once, not per figure
    }
    builtins._NV_FIGURE_AUTOSAVE = _STATE


def _get_ipython():
    try:
        from IPython import get_ipython
    except Exception:
        return None
    return get_ipython()


# --------------------------------------------------------------------------- #
# Classification and naming
# --------------------------------------------------------------------------- #

_SPECTRUM_TEXT = re.compile(
    r"sqrt\(\s*hz\s*\)|/\s*rthz|spectral density|\basd\b|\bpsd\b|\bfft\b|nT\s*/\s*√",
    re.IGNORECASE)
_DEFAULT_LABEL = re.compile(r"^(figure\s*\d+)?$", re.IGNORECASE)


def _axis_is_spectrum(ax) -> bool:
    """A spectrum panel: log frequency axis, or spectral-density wording."""
    if ax.get_xscale() == "log" and ax.get_yscale() == "log":
        return True
    text = " ".join(filter(None, (ax.get_ylabel(), ax.get_xlabel(), ax.get_title())))
    return bool(_SPECTRUM_TEXT.search(text))


def _classify(fig) -> str:
    """``fft`` only when the WHOLE figure is spectra.

    The three-panel figure from ``TwoPointResult.figure()`` carries an ASD as its
    last panel but is a time-series plot overall, so it must not claim the
    ``_fft`` name from the standalone spectrum figure of the same run.
    """
    label = (fig.get_label() or "").strip()
    if label and not _DEFAULT_LABEL.match(label):
        return re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower() or "result"
    axes = [a for a in fig.get_axes() if not getattr(a, "_nv_colorbar", False)]
    if axes and all(_axis_is_spectrum(a) for a in axes):
        return "fft"
    return "result"


def _next_path(fig) -> Path:
    st = _STATE
    kind = _classify(fig)
    n = st["counts"].get(kind, 0) + 1
    st["counts"][kind] = n
    suffix = kind if n == 1 else f"{kind}{n}"
    parts = [st["stem"] or "figure"]
    if st["tag"]:
        parts.append(st["tag"])
    parts.append(suffix)
    return Path(st["directory"]) / ("_".join(parts) + ".png")


# --------------------------------------------------------------------------- #
# The worker
# --------------------------------------------------------------------------- #

def capture(force: bool = False) -> list[Path]:
    """Write every open, un-saved, non-empty figure. Returns the paths written."""
    st = _STATE
    if not (st["enabled"] or force) or st["directory"] is None:
        return []
    try:
        from matplotlib._pylab_helpers import Gcf
    except Exception:
        return []

    written = []
    # Gcf rather than plt.get_fignums()+plt.figure(n): reading the manager list
    # does not disturb which figure is current, and a cell that goes on to draw
    # after a plt.show() must not have gcf() changed under it.
    for manager in list(Gcf.get_all_fig_managers()):
        fig = manager.canvas.figure
        if getattr(fig, "_nv_autosave_path", None) is not None:
            continue
        if not fig.get_axes():
            continue
        try:
            path = _next_path(fig)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=st["dpi"], bbox_inches="tight", facecolor="white")
        except Exception as exc:                       # never break a live run
            if not st["warned"]:
                print(f"  [figures] could not save: {exc.__class__.__name__}: {exc}")
                st["warned"] = True
            continue
        fig._nv_autosave_path = path
        written.append(path)

    if written:
        st["written"].extend(written)
        if st["verbose"]:
            names = ", ".join(p.name for p in written)
            print(f"  [figures] {len(written)} PNG -> {names}")
    return written


# --------------------------------------------------------------------------- #
# Hooks
# --------------------------------------------------------------------------- #

def _on_post_execute():
    try:
        capture()
    except Exception:
        pass


def _patch_show() -> None:
    import matplotlib.pyplot as plt
    if getattr(plt.show, "_nv_autosave", False):
        return
    original = plt.show

    @functools.wraps(original)
    def show(*args, **kwargs):
        capture()                      # before the inline backend closes them
        return original(*args, **kwargs)

    show._nv_autosave = True
    show._nv_autosave_original = original
    plt.show = show


def _unpatch_show() -> None:
    import matplotlib.pyplot as plt
    original = getattr(plt.show, "_nv_autosave_original", None)
    if original is not None:
        plt.show = original


def _register_event() -> bool:
    ip = _get_ipython()
    if ip is None:
        return False
    try:
        ip.events.unregister("post_execute", _on_post_execute)
    except Exception:
        pass
    ip.events.register("post_execute", _on_post_execute)

    # matplotlib-inline's flush_figures closes every figure it displays, so it has
    # to run last. IPython dispatches in registration order; re-registering moves
    # it to the back.
    try:
        from matplotlib_inline.backend_inline import flush_figures
    except Exception:
        return True
    callbacks = getattr(ip.events, "callbacks", {}).get("post_execute", [])
    if flush_figures in callbacks:
        try:
            ip.events.unregister("post_execute", flush_figures)
            ip.events.register("post_execute", flush_figures)
        except Exception:
            pass
    return True


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def enable(directory=None, dpi: int = 150, verbose: bool = True) -> None:
    """Start saving figures into ``directory`` (created if missing)."""
    st = _STATE
    if directory is not None:
        st["directory"] = Path(directory)
        st["directory"].mkdir(parents=True, exist_ok=True)
    if st["directory"] is None:
        raise ValueError("figure_autosave.enable() needs a directory the first time.")
    st.update(enabled=True, dpi=int(dpi), verbose=bool(verbose), warned=False)

    hooked = _register_event()
    _patch_show()
    where = "on cell end and on plt.show()" if hooked else "on plt.show() only (no IPython)"
    print(f"Figure autosave ON -> {st['directory']}  ({where}, {st['dpi']} dpi)")


def disable() -> None:
    """Stop saving. Leaves already-written files alone."""
    _STATE["enabled"] = False
    ip = _get_ipython()
    if ip is not None:
        try:
            ip.events.unregister("post_execute", _on_post_execute)
        except Exception:
            pass
    _unpatch_show()
    print("Figure autosave OFF")


def set_directory(directory) -> None:
    _STATE["directory"] = Path(directory)
    _STATE["directory"].mkdir(parents=True, exist_ok=True)


def set_run(target=None, tag: str | None = None, directory=None) -> None:
    """Name the figures that follow after ``target``, and restart the numbering.

    ``target`` is normally the run's CSV path, in which case the PNGs land in the
    same folder under the same stem. A plain string is used as the stem as-is.
    ``tag`` marks the stage ("run", "4S", ...) so an acquisition cell and the
    analysis cell that follows it do not fight over ``<stem>_result.png``.
    """
    st = _STATE
    if directory is not None:
        set_directory(directory)
    if target is not None:
        if isinstance(target, (str, Path)) and str(target).lower().endswith(
                (".csv", ".json", ".png", ".txt")):
            path = Path(target)
            st["stem"] = path.stem
            if directory is None and path.parent != Path("."):
                st["directory"] = path.parent
                st["directory"].mkdir(parents=True, exist_ok=True)
        else:
            st["stem"] = str(target)
    st["tag"] = str(tag) if tag else None
    st["counts"] = {}
    st["written"] = []


def is_enabled() -> bool:
    return bool(_STATE["enabled"])


def status() -> dict:
    st = _STATE
    return {"enabled": st["enabled"], "directory": st["directory"],
            "stem": st["stem"], "tag": st["tag"], "dpi": st["dpi"],
            "written_this_run": list(st["written"])}
