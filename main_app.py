#!/usr/bin/env python3
import threading
import time
import sys
import os
import socket
try:
    import webview
except ImportError:
    print("pywebview is not installed. Please install it using 'pip install pywebview'")
    sys.exit(1)

import uvicorn
from gui_server import app

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port

FREE_PORT = get_free_port()

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=FREE_PORT, log_level="error")

if __name__ == '__main__':
    # Start the FastAPI server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for the server to start
    time.sleep(1)
    
    # Create the webview window pointing to our local server
    webview.create_window(
        title='ClinicalWhisper v5.0', 
        url=f'http://127.0.0.1:{FREE_PORT}',
        width=1000,
        height=800,
        resizable=True
    )
    
    # Start the webview application (blocks until window is closed)
    webview.start()
