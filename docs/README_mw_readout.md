# Simple Microwave Modulation and Readout with QICK-DAWG

This repository contains example code for performing microwave modulation and readout using QICK-DAWG with RFSoC boards.

## Overview

The code demonstrates three main functionalities:

1. **Simple PL Readout**: Basic photoluminescence intensity measurement without microwave
2. **ODMR (Optically Detected Magnetic Resonance)**: Microwave frequency sweep with lock-in detection
3. **Simple MW Pulse**: Basic microwave pulse followed by readout

## Requirements

- QICK-DAWG installed (see [installation guide](https://github.com/sandialabs/qick-dawg))
- RFSoC4x2 or ZCU111 board
- Python packages: `numpy`, `matplotlib`, `qickdawg`

## Installation

1. Clone the QICK-DAWG repository:
```bash
git clone https://github.com/sandialabs/qick-dawg.git
cd qick-dawg
pip install -e .
```

2. Ensure your RFSoC board is properly configured and connected (see QICK-DAWG installation guide)

## Usage

### Basic Usage

```python
import qickdawg as qd
from simple_mw_readout import setup_configuration, simple_pl_readout, microwave_modulation_readout

# Connect to RFSoC (update IP address)
qd.start_client('172.16.26.5')  # Replace with your RFSoC IP

# Set up configuration
config = setup_configuration()

# Perform simple PL readout
pl_signal = simple_pl_readout(config)

# Perform ODMR measurement
odmr_results = microwave_modulation_readout(config)
```

### Configuration Parameters

Key parameters you may need to adjust:

- `adc_channel`: ADC channel for readout (0 or 1)
- `mw_channel`: Microwave generator channel (0 or 1 for RFSoC4x2)
- `mw_nqz`: Nyquist zone (1 for f < fdss/2, 2 for f > fdss/2)
- `mw_gain`: Microwave amplitude (0 to 32767)
- `mw_fMHz`: Microwave frequency in MHz
- `laser_gate_pmod`: PMOD channel for laser gating (0-4)
- `readout_integration_tus`: Readout integration time in microseconds
- `reps`: Number of repetitions for averaging

### Example: Custom Frequency Sweep

```python
from simple_mw_readout import setup_configuration, microwave_modulation_readout, plot_odmr_results

config = setup_configuration()

# Custom frequency sweep
config.mw_start_fMHz = 2850
config.mw_end_fMHz = 2890
config.nsweep_points = 200
config.reps = 500  # More repetitions for better signal-to-noise

# Run measurement
results = microwave_modulation_readout(config)

# Plot results
plot_odmr_results(results)
```

## Understanding the Code Structure

### Configuration (`setup_configuration()`)

The `NVConfiguration` class handles unit conversions automatically:
- Time: `_tns`, `_tus`, `_treg` (nanoseconds, microseconds, register values)
- Frequency: `_fMHz`, `_fGHz`, `_freg` (MHz, GHz, register values)
- Phase: `_pdegrees`, `_preg` (degrees, register values)

### Programs

QICK-DAWG uses program classes that inherit from `NVAveragerProgram`:
- `PLIntensity`: Simple laser on/readout
- `LockinODMR`: Microwave frequency sweep with lock-in detection
- `RabiSweep`: Rabi oscillation measurement
- Custom programs can be created by subclassing `NVAveragerProgram`

### Data Acquisition

Programs have an `acquire()` method that:
1. Configures the FPGA
2. Runs the pulse sequence
3. Collects and averages data
4. Returns processed results

## Hardware Setup

Typical setup includes:
- RFSoC board (RFSoC4x2 or ZCU111)
- Microwave amplifier
- Differential amplifier (e.g., LMH5401EVM) for ADC input
- Laser with AOM (Acousto-Optic Modulator) gated by PMOD
- Photodiode for PL detection

## Troubleshooting

1. **Connection Issues**: Ensure RFSoC IP address is correct and board is powered on
2. **No Signal**: Check ADC connections and differential amplifier settings
3. **Microwave Issues**: Verify microwave channel, gain, and frequency settings
4. **Timing Errors**: Increase `relax_delay_tns` if synchronization issues occur

## References

- [QICK-DAWG Repository](https://github.com/sandialabs/qick-dawg)
- [QICK Documentation](https://github.com/openquantumhardware/qick)
- QICK-DAWG installation guide: `qick-dawg/installation/Readme.md`

## License

This example code follows the QICK-DAWG MIT License.
