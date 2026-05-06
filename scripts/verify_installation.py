"""
Quick verification script to check QICK-DAWG installation
"""

print("=" * 60)
print("QICK-DAWG Installation Verification")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    import qickdawg as qd
    print("   ✅ qickdawg imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import qickdawg: {e}")
    exit(1)

try:
    from qickdawg.nvpulsing import PLIntensity, LockinODMR, NVConfiguration
    print("   ✅ QICK-DAWG modules imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import QICK-DAWG modules: {e}")
    exit(1)

try:
    import numpy as np
    import matplotlib.pyplot as plt
    print("   ✅ Dependencies (numpy, matplotlib) imported successfully")
except ImportError as e:
    print(f"   ❌ Failed to import dependencies: {e}")
    exit(1)

# Check version
print("\n2. Checking version...")
try:
    import subprocess
    result = subprocess.run(['pip', 'show', 'qickdawg'], 
                          capture_output=True, text=True)
    for line in result.stdout.split('\n'):
        if 'Version:' in line:
            print(f"   ✅ {line.strip()}")
            break
except Exception as e:
    print(f"   ⚠️  Could not check version: {e}")

# Test configuration creation
print("\n3. Testing configuration creation...")
try:
    # Note: NVConfiguration requires soccfg which is only available when connected to RFSoC
    # This is expected behavior - configuration will work once connected
    from qickdawg.nvpulsing import NVConfiguration
    print("   ✅ NVConfiguration class imported successfully")
    print("   ⚠️  Note: Full configuration requires RFSoC connection (soccfg)")
except Exception as e:
    print(f"   ❌ Failed to import NVConfiguration: {e}")
    exit(1)

# Check if repository exists
print("\n4. Checking repository...")
import os
repo_path = "/Users/ckasemtantikul/Documents/PhD Potential Projects/qick-dawg"
if os.path.exists(repo_path):
    print(f"   ✅ Repository found at: {repo_path}")
else:
    print(f"   ⚠️  Repository not found at expected location")

print("\n" + "=" * 60)
print("✅ Installation verification complete!")
print("=" * 60)
print("\nNext steps:")
print("1. Update RFSoC IP address in your notebooks")
print("2. Ensure Pyro server is running on RFSoC board")
print("3. Open and run the Jupyter notebooks")
print("\nNotebooks available:")
print("  - minimal_example.ipynb")
print("  - simple_mw_readout.ipynb")
print("  - advanced_mw_readout.ipynb")
