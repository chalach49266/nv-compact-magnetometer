# Connect Cursor to Remote Jupyter Server

This guide shows you how to connect Cursor (VS Code-based) to a remote Jupyter notebook server using an IP address.

## Prerequisites

1. **Remote Server**: A machine running Jupyter with an accessible IP address
2. **SSH Access**: You need SSH access to the remote server
3. **Jupyter Server**: Jupyter must be running on the remote server (with token authentication)

## Step 1: Install Required Extensions in Cursor

Open Cursor and install these extensions:

1. **Remote - SSH** (by Microsoft)
   - Search for "Remote - SSH" in the Extensions panel
   - Install the extension

2. **Python** (by Microsoft)
   - Search for "Python" in the Extensions panel
   - Install the extension

3. **Jupyter** (by Microsoft)
   - Search for "Jupyter" in the Extensions panel
   - Install the extension

## Step 2: Connect to Remote Server via SSH

1. **Open Remote Connection**:
   - Click the green `><` icon in the bottom-left corner of Cursor
   - Or press `Cmd+Shift+P` (Mac) / `Ctrl+Shift+P` (Windows/Linux)
   - Type "Remote-SSH: Connect to Host..."
   - Select it

2. **Enter SSH Address**:
   - Enter your SSH connection string:
     ```
     ssh username@remote-ip-address
     ```
   - Example: `ssh user@172.16.26.5` 
   - If you have SSH config set up, you can select from saved hosts

3. **Authenticate**:
   - Enter your password or provide SSH key
   - Cursor will install VS Code Server on the remote machine (first time only)

4. **Open Remote Folder**:
   - Once connected, click "Open Folder" in the Explorer
   - Navigate to your project directory on the remote server
   - Example: `/home/username/projects/NV_RFSoC`

## Step 3: Get Jupyter Server URL with Token

On the remote server, you need to get the Jupyter server URL with token. You can do this in several ways:

### Option A: If Jupyter is running in a Docker container
```bash
docker exec -it <container_name_or_id> jupyter server list
```

### Option B: If Jupyter is running directly on the server
```bash
jupyter server list
```

### Option C: Check Jupyter logs
Look at the output when you started Jupyter - it should show something like:
```
http://0.0.0.0:8888/?token=abcd1234efgh5678...
```

The URL should look like:
```
http://remote-ip-address:8888/?token=your-token-here
```

**Important**: Replace `0.0.0.0` or `localhost` with the actual IP address of your remote server.

## Step 4: Connect to Jupyter Server in Cursor

1. **Open a Jupyter Notebook**:
   - Open an existing `.ipynb` file (like `vector_field_sensing.ipynb`)
   - Or create a new notebook

2. **Run a Cell**:
   - Click on any cell and press `Shift + Enter`
   - Or click the "Run Cell" button

3. **Select Jupyter Server**:
   - A prompt will appear at the top asking "How would you like to connect to Jupyter?"
   - Select **"Existing Jupyter Server"**

4. **Enter Server URL**:
   - Select **"Enter the URL of the running Jupyter Server"**
   - Paste the full URL with token:
     ```
     http://172.16.26.5:8888/?token=your-token-here
     ```
   - Press Enter

5. **Select Kernel**:
   - Choose the appropriate Python kernel from the list
   - Usually "Python 3" or the name of your conda/virtual environment

## Step 5: Verify Connection

- You should see the kernel name in the top-right corner of the notebook
- Try running a cell - it should execute on the remote server
- Check that you can access remote resources (like your RFSoC at `172.16.26.5`)

## Troubleshooting

### Connection Issues

1. **Can't connect via SSH**:
   - Verify SSH is enabled on remote server
   - Check firewall settings
   - Ensure you have correct username/IP

2. **Jupyter URL not working**:
   - Make sure Jupyter is bound to `0.0.0.0` not just `localhost`
   - Check firewall allows connections on Jupyter port (usually 8888)
   - Verify the token is correct

3. **Kernel not found**:
   - Make sure Python is installed on remote server
   - Install Jupyter in the remote environment: `pip install jupyter`

### Jupyter Server Binding

If Jupyter is only accessible via `localhost`, you need to bind it to all interfaces:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Or for Jupyter Notebook:
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Security Note

- Always use token authentication (default in Jupyter)
- Consider using SSH tunneling for additional security
- Don't expose Jupyter to public internet without proper security

## Alternative: SSH Tunneling (More Secure)

If you prefer not to expose Jupyter directly, use SSH tunneling:

1. **Create SSH Tunnel** (on local machine):
   ```bash
   ssh -L 8888:localhost:8888 username@remote-ip
   ```

2. **Connect to Jupyter**:
   - Use `http://localhost:8888/?token=your-token` in Cursor
   - The tunnel forwards local port 8888 to remote port 8888

## Quick Reference

- **SSH Connection**: Bottom-left `><` icon → "Connect to Host..."
- **Jupyter Server URL**: Get from `jupyter server list` or Docker logs
- **Connect to Jupyter**: Run cell → "Existing Jupyter Server" → Paste URL
- **Select Kernel**: Top-right kernel selector

## Your Current Setup

Based on your notebook, you're connecting to:
- **RFSoC IP**: `172.16.26.5`
- **Notebook**: `vector_field_sensing.ipynb`

Make sure your remote Jupyter server can also access `172.16.26.5` if that's on the same network!
