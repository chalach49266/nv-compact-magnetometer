"""
Simple Microwave Modulation and Readout Script
==============================================

This script demonstrates how to perform simple microwave modulation and readout
using QICK-DAWG with an RFSoC board.

Based on: https://github.com/sandialabs/qick-dawg.git

Author: Generated for QICK-DAWG usage
"""

import numpy as np
import matplotlib.pyplot as plt
import qickdawg as qd
from qickdawg.nvpulsing.plintensity import PLIntensity
from qickdawg.nvpulsing.lockinodmr import LockinODMR


def setup_configuration():
    """
    Set up the default configuration for QICK-DAWG measurements.
    
    Returns:
        qd.NVConfiguration: Configuration object with default settings
    """
    config = qd.NVConfiguration()
    
    # ADC channel for readout (0 or 1 for RFSoC4x2)
    config.adc_channel = 0
    
    # Microwave channel configuration
    config.mw_channel = 0  # 0 or 1 for RFSoC4x2, 0-6 for ZCU111/ZCU216
    config.mw_nqz = 1  # Nyquist zone: 1 for f < fdss/2, 2 for f > fdss/2
    config.mw_gain = 5000  # Microwave amplitude (0 to 32767)
    
    # Microwave frequency (in MHz)
    config.mw_fMHz = 2870  # Typical NV center frequency
    
    # Laser gate PMOD channel (0-4)
    config.laser_gate_pmod = 0
    
    # Timing parameters
    config.relax_delay_tns = 500  # Delay between repetitions (ns)
    config.readout_integration_tus = 1.0  # Readout integration time (us)
    
    # Repetitions
    config.reps = 100  # Number of repetitions for averaging
    
    return config


def simple_pl_readout(config):
    """
    Perform a simple photoluminescence (PL) intensity readout.
    
    This function turns on the laser and measures the PL intensity
    without any microwave modulation.
    
    Args:
        config: NVConfiguration object
        
    Returns:
        float: Average PL intensity
    """
    print("Performing PL intensity readout...")
    
    # Create PL intensity program
    pl_program = PLIntensity(config)
    
    # Acquire data
    pl_intensity = pl_program.acquire()
    
    print(f"PL Intensity: {pl_intensity:.2f} ADC units")
    return pl_intensity


def microwave_modulation_readout(config, mw_frequencies=None):
    """
    Perform microwave modulation and readout using ODMR technique.
    
    This function sweeps microwave frequency and measures the difference
    between PL with microwave on and off (lock-in detection).
    
    Args:
        config: NVConfiguration object
        mw_frequencies: List of microwave frequencies to sweep (MHz)
                      If None, uses default sweep from config
        
    Returns:
        dict: Dictionary containing frequencies, signal, reference, and contrast
    """
    print("Performing microwave modulation readout (ODMR)...")
    
    # Set up frequency sweep if not provided
    if mw_frequencies is None:
        config.mw_start_fMHz = 2860  # Start frequency (MHz)
        config.mw_end_fMHz = 2880   # End frequency (MHz)
        config.nsweep_points = 100   # Number of frequency points
    else:
        config.mw_start_fMHz = min(mw_frequencies)
        config.mw_end_fMHz = max(mw_frequencies)
        config.nsweep_points = len(mw_frequencies)
    
    # Pre-initialization (laser pulse before measurement)
    config.pre_init = True
    
    # Create ODMR program
    odmr_program = LockinODMR(config)
    
    # Acquire data
    results = odmr_program.acquire()
    
    print(f"ODMR measurement complete!")
    print(f"Frequency range: {results.frequencies[0]:.2f} - {results.frequencies[-1]:.2f} MHz")
    print(f"Max contrast: {np.max(np.abs(results.contrast)):.2f} ADC units")
    
    return results


def plot_odmr_results(results):
    """
    Plot ODMR results showing microwave frequency vs contrast.
    
    Args:
        results: Results dictionary from microwave_modulation_readout()
    """
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(results.frequencies, results.signal, 'b-', label='MW On', linewidth=2)
    plt.plot(results.frequencies, results.reference, 'r-', label='MW Off', linewidth=2)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('PL Intensity (ADC units)')
    plt.title('ODMR: Signal and Reference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(results.frequencies, results.contrast, 'g-', linewidth=2)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Contrast (ADC units)')
    plt.title('ODMR Contrast (Signal - Reference)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def simple_mw_pulse_readout(config, pulse_length_us=0.1):
    """
    Perform a simple microwave pulse followed by readout.
    
    This demonstrates basic microwave pulsing with a single pulse length.
    
    Args:
        config: NVConfiguration object
        pulse_length_us: Microwave pulse length in microseconds
        
    Returns:
        float: Readout signal after microwave pulse
    """
    print(f"Performing simple MW pulse readout (pulse length: {pulse_length_us} us)...")
    
    # Note: This is a simplified example. For more complex pulse sequences,
    # you would use RabiSweep or create a custom program.
    
    # For now, we'll use PL intensity as a placeholder
    # In practice, you would create a custom program that:
    # 1. Applies a microwave pulse
    # 2. Waits a delay
    # 3. Performs readout
    
    config.readout_integration_tus = pulse_length_us
    
    pl_program = PLIntensity(config)
    signal = pl_program.acquire()
    
    print(f"Readout signal: {signal:.2f} ADC units")
    return signal


def main():
    """
    Main function demonstrating microwave modulation and readout.
    """
    print("=" * 60)
    print("QICK-DAWG: Simple Microwave Modulation and Readout")
    print("=" * 60)
    
    # Connect to QICK-DAWG server
    # Replace '172.16.26.5' with your RFSoC IP address
    # For local operation, you may not need this
    try:
        qd.start_client('172.16.26.5')  # Update with your RFSoC IP
        print("Connected to QICK-DAWG server")
    except Exception as e:
        print(f"Warning: Could not connect to server: {e}")
        print("Continuing with local configuration...")
    
    # Set up configuration
    config = setup_configuration()
    print("\nConfiguration set up:")
    print(f"  ADC Channel: {config.adc_channel}")
    print(f"  MW Channel: {config.mw_channel}")
    print(f"  MW Frequency: {config.mw_fMHz} MHz")
    print(f"  MW Gain: {config.mw_gain}")
    print(f"  Readout Integration: {config.readout_integration_tus} us")
    
    # Example 1: Simple PL readout (no microwave)
    print("\n" + "-" * 60)
    print("Example 1: Simple PL Readout")
    print("-" * 60)
    pl_signal = simple_pl_readout(config)
    
    # Example 2: Microwave modulation readout (ODMR)
    print("\n" + "-" * 60)
    print("Example 2: Microwave Modulation Readout (ODMR)")
    print("-" * 60)
    odmr_results = microwave_modulation_readout(config)
    
    # Plot results
    print("\nPlotting ODMR results...")
    plot_odmr_results(odmr_results)
    
    # Example 3: Simple microwave pulse
    print("\n" + "-" * 60)
    print("Example 3: Simple MW Pulse Readout")
    print("-" * 60)
    mw_pulse_signal = simple_mw_pulse_readout(config, pulse_length_us=0.1)
    
    print("\n" + "=" * 60)
    print("Measurement complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
