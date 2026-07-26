import os
import sys
import threading
import time
import socket
import uvicorn
import webview
from gui_server import app

def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port

# Get a free port globally so both thread and webview use the same one
FREE_PORT = get_free_port()

def run_server():
    # Run the FastAPI server on a background thread using the free port
    uvicorn.run(app, host="127.0.0.1", port=FREE_PORT, log_level="error")

if __name__ == '__main__':
    # An app launched from Finder inherits a minimal PATH that excludes Homebrew,
    # so ffmpeg would be invisible to the pipeline's subprocess calls.
    os.environ["PATH"] += os.pathsep + "/usr/local/bin" + os.pathsep + "/opt/homebrew/bin"

    # Start the server thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Give the server a moment to start
    time.sleep(1)

    # Create the native macOS window. Resizable: the results grid and console
    # need room on longer interviews.
    webview.create_window(
        "ClinicalWhisper",
        f"http://127.0.0.1:{FREE_PORT}",
        width=900,
        height=760,
        resizable=True,
        text_select=True,
        background_color="#FAFAFA"
    )
    
    # Start the webview GUI loop
    webview.start()
