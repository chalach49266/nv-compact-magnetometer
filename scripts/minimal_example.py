"""
Minimal Example: Microwave Modulation and Readout
=================================================

This is the simplest possible example for performing microwave modulation
and readout with QICK-DAWG.

Usage:
    1. Update the RFSoC IP address below
    2. Ensure QICK-DAWG is installed
    3. Run: python minimal_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
import qickdawg as qd
from qickdawg.nvpulsing import PLIntensity, LockinODMR

# Step 1: Connect to RFSoC board
# Replace '172.16.26.5' with your RFSoC IP address
qd.start_client('172.16.26.5')

# Step 2: Create configuration
config = qd.NVConfiguration()

# Essential settings
config.adc_channel = 0          # ADC channel for readout
config.mw_channel = 0            # Microwave generator channel
config.mw_nqz = 1                # Nyquist zone (1 or 2)
config.mw_gain = 5000            # Microwave amplitude (0-32767)
config.laser_gate_pmod = 0       # PMOD channel for laser gating
config.relax_delay_tns = 500     # Delay between repetitions (ns)
config.readout_integration_tus = 1.0  # Readout time (us)
config.reps = 100                # Number of repetitions

# Step 3: Simple PL readout (no microwave)
print("Measuring PL intensity...")
pl_program = PLIntensity(config)
pl_signal = pl_program.acquire()
print(f"PL Intensity: {pl_signal:.2f} ADC units")

# Step 4: ODMR measurement (microwave frequency sweep)
print("\nPerforming ODMR measurement...")
config.mw_start_fMHz = 2860      # Start frequency (MHz)
config.mw_end_fMHz = 2880       # End frequency (MHz)
config.nsweep_points = 100      # Number of frequency points
config.pre_init = True          # Pre-initialize with laser

odmr_program = LockinODMR(config)
odmr_results = odmr_program.acquire()

# Step 5: Display results
print(f"\nODMR Results:")
print(f"Frequency range: {odmr_results.frequencies[0]:.2f} - {odmr_results.frequencies[-1]:.2f} MHz")
print(f"Max contrast: {np.max(np.abs(odmr_results.contrast)):.2f} ADC units")

# Optional: Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(odmr_results.frequencies, odmr_results.contrast, 'b-', linewidth=2)
plt.xlabel('Frequency (MHz)')
plt.ylabel('ODMR Contrast (ADC units)')
plt.title('ODMR Spectrum')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nDone!")
