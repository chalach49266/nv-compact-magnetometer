
from .nvqicksweep import NVQickSweep
from .nvconfiguration import NVConfiguration
from .nvaverageprogram import NVAveragerProgram

from .laseron import LaserOn, laser_on
from .laseroff import LaserOff, laser_off
from .plintensity import PLIntensity
from .dark_counts import DarkCounts
from .lockinodmr import LockinODMR
from .fixedfrequencyodmr import FixedFrequencyODMR
from .readoutwindow import ReadoutWindow
from .getreadoutwindow import get_readout_window
from .check_readout import check_readout

from .integrated_readout_window import IntegratedReadoutWindow
from .rabisweep import RabiSweep
from .hahnechodelaysweep import HahnEchoDelaySweep
from .t1delaysweep import T1DelaySweep
from .ramsey import Ramsey
from .cpmgxy8ndelaysweep import CPMGXY8nDelaySweep
from .cpmgxy8nsweep import CPMGXY8nSweep

try:
    from .nv_live_helpers import PLIntensityDual, run_fast_pl_no_display, run_live_odmr
except ModuleNotFoundError as exc:
    _nv_live_helpers_import_error = exc

    class PLIntensityDual:  # pragma: no cover - optional dependency guard
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "PLIntensityDual requires optional notebook dependencies. "
                f"Missing module: {_nv_live_helpers_import_error.name}"
            ) from _nv_live_helpers_import_error

    def run_fast_pl_no_display(*args, **kwargs):  # pragma: no cover - optional dependency guard
        raise ModuleNotFoundError(
            "run_fast_pl_no_display requires optional notebook dependencies. "
            f"Missing module: {_nv_live_helpers_import_error.name}"
        ) from _nv_live_helpers_import_error

    def run_live_odmr(*args, **kwargs):  # pragma: no cover - optional dependency guard
        raise ModuleNotFoundError(
            "run_live_odmr requires optional notebook dependencies. "
            f"Missing module: {_nv_live_helpers_import_error.name}"
        ) from _nv_live_helpers_import_error
