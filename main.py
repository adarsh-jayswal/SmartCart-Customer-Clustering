import sys
import os
import time
import socket
import subprocess

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def main():
    print("========================================")
    print("      SmartCart Clustering System")
    print("========================================\n")
    print("Frontend:")
    print("http://127.0.0.1:5500/\n")
    print("Backend:")
    print("http://127.0.0.1:8000/\n")
    print("API Health:")
    print("http://127.0.0.1:8000/health\n")
    print("API Root:")
    print("http://127.0.0.1:8000/\n")
    print("----------------------------------------")
    print("Frontend and Backend are running.")
    print("Press Ctrl+C to stop.")
    print("----------------------------------------\n")

    processes = []
    root_dir = os.path.abspath(os.path.dirname(__file__))
    frontend_dir = os.path.join(root_dir, "frontend")

    # 1. Start FastAPI Backend if port 8000 is free
    if is_port_in_use(8000):
        print("-> Backend port 8000 is already in use. Using existing running process.")
    else:
        backend_cmd = [
            sys.executable, "-m", "uvicorn", "backend.main:app",
            "--host", "127.0.0.1", "--port", "8000"
        ]
        backend_proc = subprocess.Popen(backend_cmd, cwd=root_dir)
        processes.append(("Backend", backend_proc))
        print("-> Started FastAPI backend on http://127.0.0.1:8000/")

    # 2. Start Frontend static server if port 5500 is free
    if is_port_in_use(5500):
        print("-> Frontend port 5500 is already in use. Using existing running process.")
    else:
        frontend_cmd = [
            sys.executable, "-m", "http.server", "5500",
            "--directory", frontend_dir
        ]
        frontend_proc = subprocess.Popen(frontend_cmd, cwd=root_dir)
        processes.append(("Frontend", frontend_proc))
        print("-> Started Frontend server on http://127.0.0.1:5500/\n")

    try:
        while True:
            time.sleep(1)
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"Warning: {name} process exited with code {proc.returncode}")
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                print(f"-> Stopped {name} server.")
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
