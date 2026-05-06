#!/usr/bin/env python3
"""
Quick script to test Jupyter server connection.
Run this on your remote server to verify Jupyter is accessible.
"""

import subprocess
import sys
import socket
from urllib.parse import urlparse

def get_jupyter_servers():
    """Get list of running Jupyter servers."""
    try:
        result = subprocess.run(
            ['jupyter', 'server', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running jupyter server list: {e}")
        return None
    except FileNotFoundError:
        print("Jupyter not found. Install with: pip install jupyter")
        return None

def get_local_ip():
    """Get local IP address."""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

def parse_jupyter_urls(output):
    """Parse Jupyter server list output."""
    urls = []
    for line in output.split('\n'):
        if 'http' in line:
            # Extract URL
            parts = line.strip().split()
            if parts:
                url = parts[0]
                urls.append(url)
    return urls

def main():
    print("=" * 60)
    print("Jupyter Server Connection Test")
    print("=" * 60)
    print()
    
    # Get local IP
    local_ip = get_local_ip()
    print(f"Local IP address: {local_ip}")
    print()
    
    # Get Jupyter servers
    print("Checking for running Jupyter servers...")
    output = get_jupyter_servers()
    
    if output:
        print("\nFound Jupyter servers:")
        print("-" * 60)
        print(output)
        print("-" * 60)
        
        urls = parse_jupyter_urls(output)
        if urls:
            print("\n📋 URLs to use in Cursor:")
            print()
            for url in urls:
                parsed = urlparse(url)
                # Replace localhost/0.0.0.0 with actual IP
                if 'localhost' in parsed.netloc or '0.0.0.0' in parsed.netloc:
                    port = parsed.port or 8888
                    # Extract token from query string
                    token = None
                    if parsed.query:
                        for param in parsed.query.split('&'):
                            if param.startswith('token='):
                                token = param.split('=')[1]
                                break
                    
                    if token:
                        new_url = f"http://{local_ip}:{port}/?token={token}"
                        print(f"  {new_url}")
                    else:
                        print(f"  http://{local_ip}:{port}/")
                else:
                    print(f"  {url}")
        else:
            print("\n⚠️  No URLs found in output")
    else:
        print("\n❌ No Jupyter servers found running")
        print("\nTo start Jupyter Lab:")
        print(f"  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser")
        print("\nTo start Jupyter Notebook:")
        print(f"  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser")
    
    print()
    print("=" * 60)
    print("Next steps:")
    print("1. Copy one of the URLs above")
    print("2. In Cursor, open a .ipynb file")
    print("3. Run a cell → Select 'Existing Jupyter Server'")
    print("4. Paste the URL")
    print("=" * 60)

if __name__ == "__main__":
    main()
