# External Microwave Control Mode

## Overview
This code has been modified to support external microwave source control. When enabled, all built-in microwave control operations are bypassed, allowing you to control the microwave source externally.

## Changes Made

### Experiment_PB_DAQ.m
1. **Added External Microwave Control Flag**: Set `handles.ExternalMicrowaveControl = true` in `Experiment_PB_DAQ_OpeningFcn`
2. **Modified Callbacks**: The following callbacks now check the flag and return immediately if external control is enabled:
   - `fixPow_Callback` - Microwave power control
   - `fixFreq_Callback` - Microwave frequency control
   - `FixSG2Pow_Callback` - Signal Generator 2 power control
   - `FixSG2Freq_Callback` - Signal Generator 2 frequency control

### ImageNVC.m
1. **Added External Microwave Control Flag**: Set `handles.ExternalMicrowaveControl = true` in `ImageNVC_OpeningFcn`
2. **Consistency**: Flag matches the setting in Experiment_PB_DAQ.m

## Important Notes

### ExperimentFunctionPool.m
If `ExperimentFunctionPool.m` contains microwave control code (which is likely since it's called during experiment execution), you may need to modify it to check for the `ExternalMicrowaveControl` flag. Look for:
- GPIB commands to signal generators
- Frequency/power setting operations
- Microwave sweep operations

Example modification pattern:
```matlab
% Check if external microwave control is enabled
if isfield(handles, 'ExternalMicrowaveControl') && handles.ExternalMicrowaveControl
    % Skip microwave control - controlled externally
    return;  % or skip the microwave-related code block
end
% ... original microwave control code ...
```

### ImageFunctionPool.m
Similarly, if `ImageFunctionPool.m` contains microwave control code, modify it to check the flag before executing microwave operations.

## Usage

1. **Enable External Control**: The flag is set to `true` by default in both files
2. **Control Microwave Externally**: Use your external microwave source control software/hardware to set frequencies and power levels
3. **Run Experiments**: The MATLAB code will run normally but will skip all microwave control operations

## Disabling External Control Mode

To revert to built-in microwave control:
1. Set `handles.ExternalMicrowaveControl = false` in both `Experiment_PB_DAQ_OpeningFcn` and `ImageNVC_OpeningFcn`
2. Ensure your signal generator hardware is properly connected via GPIB

## Testing

After modification:
1. Run `Experiment_PB_DAQ` - verify GUI loads without errors
2. Run `ImageNVC` - verify GUI loads without errors  
3. Check that microwave-related GUI elements don't cause errors when clicked
4. Verify that your external microwave control works independently
5. Run a test experiment to ensure data acquisition still works correctly

## Troubleshooting

- **If experiments fail**: Check if `ExperimentFunctionPool.m` needs modification
- **If GUI elements cause errors**: Ensure the flag is properly set in OpeningFcn
- **If microwave still controlled**: Check for other files that might control microwave (search for GPIB, signal generator, or frequency commands)
