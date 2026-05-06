"""
Advanced Microwave Modulation and Readout Examples
==================================================

This script demonstrates more advanced microwave pulse sequences including:
- Custom pulse sequences
- Multiple readout windows
- Phase modulation
- Time-domain measurements

Based on: https://github.com/sandialabs/qick-dawg.git
"""

import numpy as np
import matplotlib.pyplot as plt
import qickdawg as qd
from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram
from qickdawg.nvpulsing import NVQickSweep


class SimpleMWReadout(NVAveragerProgram):
    """
    Custom program for simple microwave pulse followed by readout.
    
    This demonstrates how to create a custom pulse sequence:
    1. Initialize spin state (optional laser pulse)
    2. Apply microwave pulse
    3. Perform readout
    """
    
    required_cfg = [
        "adc_channel",
        "mw_channel",
        "mw_nqz",
        "mw_gain",
        "mw_freg",
        "mw_pulse_length_treg",
        "readout_integration_treg",
        "laser_gate_pmod",
        "relax_delay_treg",
        "reps",
        "mw_readout_delay_treg"
    ]
    
    def initialize(self):
        """Initialize the pulse sequence."""
        self.check_cfg()
        
        # Setup readout
        self.setup_readout()
        
        # Configure microwave generator
        self.declare_gen(
            ch=self.cfg.mw_channel,
            nqz=self.cfg.mw_nqz
        )
        
        # Set pulse parameters
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            freq=self.cfg.mw_freg,
            gain=self.cfg.mw_gain,
            length=self.cfg.mw_pulse_length_treg,
            phase=0
        )
        
        self.synci(100)
        
        # Optional: pre-initialize with laser
        if hasattr(self.cfg, 'pre_init') and self.cfg.pre_init:
            self.trigger(
                pins=[self.cfg.laser_gate_pmod],
                width=self.cfg.readout_integration_treg,
                adc_trig_offset=0
            )
            self.sync_all(self.cfg.readout_integration_treg + self.cfg.relax_delay_treg)
    
    def body(self):
        """Body of the pulse sequence."""
        # Apply microwave pulse
        self.pulse(ch=self.cfg.mw_channel, t=0)
        
        # Wait for microwave pulse to finish + delay
        self.synci(self.cfg.mw_pulse_length_treg + self.cfg.mw_readout_delay_treg)
        
        # Perform readout
        self.trigger(
            adcs=[self.cfg.adc_channel],
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.readout_integration_treg,
            adc_trig_offset=0,
            t=0
        )
        
        self.wait_all()
        self.sync_all(self.cfg.relax_delay_treg)


class MWPulseSweep(NVAveragerProgram):
    """
    Sweep microwave pulse length and measure readout signal.
    
    This is similar to Rabi oscillations but simplified.
    """
    
    required_cfg = [
        "adc_channel",
        "mw_channel",
        "mw_nqz",
        "mw_gain",
        "mw_freg",
        "mw_start_treg",
        "mw_end_treg",
        "nsweep_points",
        "readout_integration_treg",
        "laser_gate_pmod",
        "relax_delay_treg",
        "reps",
        "mw_readout_delay_treg"
    ]
    
    def initialize(self):
        """Initialize with pulse length sweep."""
        self.check_cfg()
        
        self.setup_readout()
        
        # Configure microwave generator
        self.declare_gen(
            ch=self.cfg.mw_channel,
            nqz=self.cfg.mw_nqz
        )
        
        # Set default pulse parameters
        self.default_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            freq=self.cfg.mw_freg,
            gain=self.cfg.mw_gain,
            phase=0
        )
        
        # Set initial pulse length
        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            length=self.cfg.mw_start_treg
        )
        
        # Create sweep register for pulse length
        self.mw_length_reg = self.new_gen_reg(
            self.cfg.mw_channel,
            name='mw_length',
            init_val=self.cfg.mw_start_treg
        )
        
        # Add sweep
        self.add_sweep(
            NVQickSweep(
                self,
                reg=self.mw_length_reg,
                start=self.cfg.mw_start_treg,
                stop=self.cfg.mw_end_treg,
                expts=self.cfg.nsweep_points,
                label='pulse_length',
                mw_channel=self.cfg.mw_channel
            )
        )
        
        self.synci(100)
    
    def body(self):
        """Body with swept pulse length."""
        # Apply microwave pulse (length is swept)
        self.pulse(ch=self.cfg.mw_channel, t=0)
        
        # Sync to ensure pulse completes
        self.sync(self.mw_length_reg.page, self.mw_length_reg.addr)
        
        # Wait delay before readout
        self.synci(self.cfg.mw_readout_delay_treg)
        
        # Perform readout
        self.trigger(
            adcs=[self.cfg.adc_channel],
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.readout_integration_treg,
            adc_trig_offset=0,
            t=0
        )
        
        self.wait_all()
        self.sync_all(self.cfg.relax_delay_treg)
    
    def acquire(self, raw_data=False, *arg, **kwarg):
        """Acquire and process data."""
        data = super().acquire(readouts_per_experiment=1, *arg, **kwarg)
        
        if raw_data:
            return data
        
        # Process data
        data = np.reshape(data, self.data_shape)
        
        if self.cfg.edge_counting is False:
            data = data / self.cfg.readout_integration_treg
        
        # Average over reps
        if self.cfg.edge_counting:
            signal = np.sum(data, axis=0)
        else:
            signal = np.mean(data, axis=0)
        
        # Get sweep points
        pulse_lengths = self.qick_sweeps[0].get_sweep_pts()
        
        return {
            'pulse_lengths': pulse_lengths,
            'signal': signal
        }


def run_simple_mw_readout(config, pulse_length_us=0.1):
    """
    Run simple microwave pulse readout.
    
    Args:
        config: NVConfiguration object
        pulse_length_us: Microwave pulse length in microseconds
        
    Returns:
        float: Readout signal
    """
    print(f"Running simple MW readout (pulse length: {pulse_length_us} us)...")
    
    # Set pulse length
    config.mw_pulse_length_tus = pulse_length_us
    config.mw_readout_delay_tus = 0.1  # Delay between MW and readout
    
    # Create and run program
    program = SimpleMWReadout(config)
    signal = program.acquire()
    
    if config.edge_counting:
        signal = np.sum(signal) / config.reps
    else:
        signal = np.mean(signal) / config.readout_integration_treg
    
    print(f"Readout signal: {signal:.2f}")
    return signal


def run_pulse_length_sweep(config, start_us=0.01, end_us=1.0, n_points=50):
    """
    Sweep microwave pulse length and measure readout.
    
    Args:
        config: NVConfiguration object
        start_us: Starting pulse length (us)
        end_us: Ending pulse length (us)
        n_points: Number of points
        
    Returns:
        dict: Results with pulse_lengths and signal arrays
    """
    print(f"Running pulse length sweep ({start_us} - {end_us} us, {n_points} points)...")
    
    # Set sweep parameters
    config.mw_start_tus = start_us
    config.mw_end_tus = end_us
    config.nsweep_points = n_points
    config.mw_readout_delay_tus = 0.1
    
    # Create and run program
    program = MWPulseSweep(config)
    results = program.acquire()
    
    print(f"Measurement complete!")
    print(f"Signal range: {np.min(results['signal']):.2f} - {np.max(results['signal']):.2f}")
    
    return results


def plot_pulse_sweep(results):
    """Plot pulse length sweep results."""
    plt.figure(figsize=(10, 6))
    
    # Convert register values to microseconds if needed
    if np.max(results['pulse_lengths']) > 1000:
        # Likely in register units, convert
        pulse_lengths_us = results['pulse_lengths'] * 1e-6 * qd.soccfg.cycles2us(1)
    else:
        pulse_lengths_us = results['pulse_lengths']
    
    plt.plot(pulse_lengths_us, results['signal'], 'b-', linewidth=2)
    plt.xlabel('Microwave Pulse Length (μs)')
    plt.ylabel('Readout Signal (ADC units)')
    plt.title('Pulse Length Sweep')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating advanced features."""
    print("=" * 60)
    print("QICK-DAWG: Advanced Microwave Modulation Examples")
    print("=" * 60)
    
    # Connect to server (update IP)
    try:
        qd.start_client('172.16.26.5')
        print("Connected to QICK-DAWG server")
    except Exception as e:
        print(f"Warning: Could not connect: {e}")
    
    # Setup configuration
    config = qd.NVConfiguration()
    config.adc_channel = 0
    config.mw_channel = 0
    config.mw_nqz = 1
    config.mw_gain = 5000
    config.mw_fMHz = 2870
    config.laser_gate_pmod = 0
    config.relax_delay_tns = 500
    config.readout_integration_tus = 1.0
    config.reps = 100
    
    # Example 1: Simple MW pulse readout
    print("\n" + "-" * 60)
    print("Example 1: Simple MW Pulse Readout")
    print("-" * 60)
    signal = run_simple_mw_readout(config, pulse_length_us=0.1)
    
    # Example 2: Pulse length sweep
    print("\n" + "-" * 60)
    print("Example 2: Pulse Length Sweep")
    print("-" * 60)
    sweep_results = run_pulse_length_sweep(
        config,
        start_us=0.01,
        end_us=1.0,
        n_points=50
    )
    
    # Plot results
    print("\nPlotting results...")
    plot_pulse_sweep(sweep_results)
    
    print("\n" + "=" * 60)
    print("Advanced examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
