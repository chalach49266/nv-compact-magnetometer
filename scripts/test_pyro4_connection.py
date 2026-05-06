#!/usr/bin/env python3
"""
Diagnostic script to test Pyro4 nameserver connection.
Run this from your client machine to diagnose connection issues.
"""

import socket
import sys
import subprocess
from urllib.parse import urlparse

def test_ping(host):
    """Test if host is reachable via ping."""
    try:
        result = subprocess.run(
            ['ping', '-c', '1', '-W', '1', host],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False

def test_port(host, port, timeout=3):
    """Test if port is open and reachable."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_pyro4_connection(host, port=8888):
    """Test Pyro4 nameserver connection."""
    try:
        import Pyro4
        Pyro4.config.SERIALIZER = "pickle"
        Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
        
        print(f"  Attempting to locate nameserver...")
        ns = Pyro4.locateNS(host=host, port=port, timeout=5)
        print(f"  ✅ Successfully connected to nameserver!")
        
        # List registered services
        services = ns.list()
        if services:
            print(f"  Found {len(services)} registered service(s):")
            for name, uri in services.items():
                print(f"    - {name}: {uri}")
        else:
            print(f"  ⚠️  No services registered yet")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pyro4_connection.py <server_ip> [port]")
        print("Example: python test_pyro4_connection.py 192.168.0.102")
        print("Example: python test_pyro4_connection.py 192.168.0.102 8888")
        sys.exit(1)
    
    host = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8888
    
    print("=" * 60)
    print(f"Pyro4 Connection Diagnostic")
    print("=" * 60)
    print(f"Target: {host}:{port}")
    print()
    
    # Test 1: Ping
    print("1. Testing network connectivity (ping)...")
    if test_ping(host):
        print(f"  ✅ Host {host} is reachable")
    else:
        print(f"  ❌ Host {host} is NOT reachable")
        print(f"     Check: Network connection, IP address, firewall")
        print()
        return
    print()
    
    # Test 2: Port connectivity
    print(f"2. Testing port {port} connectivity...")
    if test_port(host, port):
        print(f"  ✅ Port {port} is open and reachable")
    else:
        print(f"  ❌ Port {port} is NOT reachable")
        print(f"     Check:")
        print(f"     - Nameserver is running: pyro4-ns -n {host} -p {port}")
        print(f"     - Firewall allows port {port}")
        print(f"     - Nameserver is bound to {host} (not localhost)")
        print()
        return
    print()
    
    # Test 3: Pyro4 connection
    print("3. Testing Pyro4 nameserver connection...")
    if test_pyro4_connection(host, port):
        print()
        print("=" * 60)
        print("✅ All tests passed! Connection should work.")
        print("=" * 60)
        print()
        print("In your notebook, use:")
        print(f"  qd.start_client('{host}')")
    else:
        print()
        print("=" * 60)
        print("❌ Pyro4 connection failed")
        print("=" * 60)
        print()
        print("Troubleshooting steps:")
        print(f"1. On server, verify nameserver is running:")
        print(f"   PYRO_SERIALIZERS_ACCEPTED=pickle PYRO_PICKLE_PROTOCOL_VERSION=4 \\")
        print(f"   pyro4-ns -n {host} -p {port}")
        print()
        print(f"2. Check nameserver output shows:")
        print(f"   NS running on {host}:{port}")
        print()
        print(f"3. Verify firewall allows port {port}")
        print()
        print("See PYRO4_CONNECTION_TROUBLESHOOTING.md for more details")

if __name__ == "__main__":
    main()
