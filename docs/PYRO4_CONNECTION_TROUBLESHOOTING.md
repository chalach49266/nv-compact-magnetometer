# Pyro4 Connection Troubleshooting Guide

## Error: `TimeoutError: [Errno 60] Operation timed out`

This error occurs when the client cannot connect to the Pyro4 nameserver. Here's how to fix it:

## Common Issues and Solutions

### 1. **Verify Nameserver is Running**

On the **server machine** (where RFSoC/Pyro4 server is running), check:

```bash
# Check if nameserver process is running
ps aux | grep pyro4-ns

# Check if port 8888 is listening
netstat -an | grep 8888
# or
lsof -i :8888
```

### 2. **Check Nameserver IP Binding**

The nameserver must be bound to an IP address that's accessible from your client machine.

**On the server**, start the nameserver with the correct IP:

```bash
PYRO_SERIALIZERS_ACCEPTED=pickle PYRO_PICKLE_PROTOCOL_VERSION=4 pyro4-ns -n <SERVER_IP> -p 8888
```

**Important**: Replace `<SERVER_IP>` with:
- The actual IP address of the server (e.g., `192.168.0.102`)
- NOT `localhost` or `127.0.0.1` (these only work locally)
- `0.0.0.0` works but you need to connect using the actual server IP

### 3. **Verify Network Connectivity**

**From your client machine**, test connectivity:

```bash
# Ping the server
ping 192.168.0.102

# Test if port 8888 is reachable
telnet 192.168.0.102 8888
# or
nc -zv 192.168.0.102 8888
```

If these fail, you have a network connectivity issue.

### 4. **Check Firewall Settings**

**On the server**, ensure port 8888 is open:

```bash
# For Linux (ufw)
sudo ufw allow 8888/tcp

# For Linux (iptables)
sudo iptables -A INPUT -p tcp --dport 8888 -j ACCEPT

# Check firewall status
sudo ufw status
```

### 5. **Verify IP Address**

Make sure you're using the correct IP address:

```bash
# On the server, check IP addresses
ip addr show
# or
ifconfig
```

Look for the IP address on the network interface that connects to your client machine.

### 6. **Check Nameserver Output**

When you start the nameserver, it should show:

```
Broadcast server running on 0.0.0.0:9091
NS running on <IP>:8888 (<IP>)
Warning: HMAC key not set. Anyone can connect to this server!
URI = PYRO:Pyro.NameServer@<IP>:8888
```

**Important**: The "NS running on" line shows the IP and port. Make sure:
- The IP matches what you're using in `start_client()`
- The port is 8888 (or match it in your client code)

## Step-by-Step Fix

### On the Server Machine:

1. **Stop any existing nameserver**:
   ```bash
   pkill -f pyro4-ns
   ```

2. **Find your server's IP address**:
   ```bash
   hostname -I
   # or
   ip addr show | grep "inet "
   ```

3. **Start nameserver with explicit IP**:
   ```bash
   PYRO_SERIALIZERS_ACCEPTED=pickle PYRO_PICKLE_PROTOCOL_VERSION=4 pyro4-ns -n 192.168.0.102 -p 8888
   ```
   (Replace `192.168.0.102` with your actual server IP)

4. **Verify it's running**:
   ```bash
   netstat -an | grep 8888
   ```

### On the Client Machine (where you run Jupyter):

1. **Test connectivity**:
   ```bash
   ping 192.168.0.102
   telnet 192.168.0.102 8888
   ```

2. **In your notebook**, use the correct IP:
   ```python
   qd.start_client('192.168.0.102')  # Use the actual server IP
   ```

## Alternative: Use Custom Port

If port 8888 is blocked or in use, you can modify the client to use a different port:

```python
# Modify start_client call to use custom port
import qickdawg as qd
qd.start_client('192.168.0.102', host_port=9090)  # Use port 9090 instead
```

**Note**: You'll need to modify `startclient.py` to accept the port parameter, or start the nameserver on port 9090.

## Network Configuration Examples

### Same Network (Most Common)
- Server IP: `192.168.0.102`
- Client IP: `192.168.0.xxx` (same subnet)
- Nameserver: `pyro4-ns -n 192.168.0.102 -p 8888`
- Client: `qd.start_client('192.168.0.102')`

### Different Networks (VPN/Remote)
- Server IP: `172.16.26.5` (remote network)
- Client IP: `192.168.0.xxx` (local network)
- Nameserver: `pyro4-ns -n 172.16.26.5 -p 8888`
- Client: `qd.start_client('172.16.26.5')`
- **Requires**: VPN or port forwarding

### Localhost (Same Machine)
- Server IP: `localhost` or `127.0.0.1`
- Nameserver: `pyro4-ns -n localhost -p 8888`
- Client: `qd.start_client('localhost')`

## Diagnostic Script

Run `test_pyro4_connection.py` (see below) to diagnose connection issues automatically.

## Still Not Working?

1. **Check if you're on the same network**:
   - Same WiFi/Ethernet network?
   - Same VPN if using remote connection?

2. **Try using the server's hostname** instead of IP:
   ```python
   qd.start_client('server-hostname.local')
   ```

3. **Check Pyro4 version compatibility**:
   ```bash
   pip show Pyro4
   ```

4. **Enable Pyro4 logging** for more details:
   ```python
   import Pyro4
   Pyro4.config.LOGLEVEL = "DEBUG"
   ```
