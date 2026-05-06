# Quick Fix for Pyro4 Connection Timeout

## Your Error
```
TimeoutError: [Errno 60] Operation timed out
cannot connect to ('192.168.0.102', 8888)
```

## Most Likely Causes

### 1. **Nameserver IP Mismatch** (Most Common)
The nameserver is running but bound to the wrong IP address.

**Fix on Server:**
```bash
# Stop existing nameserver
pkill -f pyro4-ns

# Start with correct IP (use the IP you see in your error)
PYRO_SERIALIZERS_ACCEPTED=pickle PYRO_PICKLE_PROTOCOL_VERSION=4 \
pyro4-ns -n 192.168.0.102 -p 8888
```

**Verify output shows:**
```
NS running on 192.168.0.102:8888 (192.168.0.102)
```

### 2. **Network Connectivity Issue**
Test from your client machine:
```bash
# Test ping
ping 192.168.0.102

# Test port
telnet 192.168.0.102 8888
# or
nc -zv 192.168.0.102 8888
```

If these fail, check:
- Are you on the same network?
- Is the server IP correct?
- Is there a firewall blocking?

### 3. **Firewall Blocking Port 8888**
On the server:
```bash
# Allow port 8888
sudo ufw allow 8888/tcp
# or
sudo iptables -A INPUT -p tcp --dport 8888 -j ACCEPT
```

## Quick Diagnostic

Run this from your client machine:
```bash
python test_pyro4_connection.py 192.168.0.102
```

This will test:
1. Network connectivity (ping)
2. Port accessibility
3. Pyro4 connection

## Updated Code

I've fixed a bug in `startclient.py` - now you can use custom ports:

```python
# Default port 8888
qd.start_client('192.168.0.102')

# Custom port (if nameserver uses different port)
qd.start_client('192.168.0.102', host_port=9090)
```

## Step-by-Step Solution

1. **On Server** (where RFSoC/Pyro4 runs):
   ```bash
   # Find server IP
   hostname -I
   
   # Start nameserver with that IP
   PYRO_SERIALIZERS_ACCEPTED=pickle PYRO_PICKLE_PROTOCOL_VERSION=4 \
   pyro4-ns -n <YOUR_SERVER_IP> -p 8888
   ```

2. **On Client** (your Jupyter notebook):
   ```python
   import qickdawg as qd
   qd.start_client('<YOUR_SERVER_IP>')  # Use same IP as nameserver
   ```

3. **Verify**:
   - Nameserver output shows correct IP
   - Network connectivity works
   - Port 8888 is accessible

## Still Not Working?

See `PYRO4_CONNECTION_TROUBLESHOOTING.md` for detailed troubleshooting.
