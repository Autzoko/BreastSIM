#!/usr/bin/env python
"""Probe Lab as a native desktop window (no browser), rendered on a remote GPU host.

Starts the preview server on the host over SSH if needed, opens an SSH tunnel and
shows the UI in a native window (WebKit on macOS via pywebview). Closing the window
closes the tunnel; the server keeps running for the next session.

    python src/scripts/probe_lab_app.py                       # NYUAIRLAB-127
    python src/scripts/probe_lab_app.py --host JUBAIL-GPU --remote-dir /scratch/...
"""
import argparse
import socket
import subprocess
import sys
import time
import urllib.request


def free_port(start):
    port = start
    while True:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="NYUAIRLAB-127", help="ssh alias of the GPU host")
    ap.add_argument("--port", type=int, default=8765, help="server port on the host")
    ap.add_argument("--remote-dir", default="~/langtian/BreastSIM")
    ap.add_argument("--restart", action="store_true", help="pull code and restart the server first")
    args = ap.parse_args()

    import webview  # pywebview

    session = "breastsim_rt"
    ssh = ["ssh", "-o", "BatchMode=yes", args.host]
    if args.restart:
        subprocess.run(ssh + [f"tmux kill-session -t {session} 2>/dev/null || true"], check=False)
    print(f"[app] {args.host}: syncing code and starting the server if needed")
    subprocess.run(ssh + [
        f"cd {args.remote_dir} && (git pull --ff-only -q || true) && "
        f"(tmux has-session -t {session} 2>/dev/null || "
        f"tmux new-session -d -s {session} 'PORT={args.port} bash src/scripts/realtime_server.sh')"], check=True)
    ok = subprocess.run(ssh + [
        f"for i in $(seq 90); do curl -sf http://127.0.0.1:{args.port}/api/config >/dev/null && exit 0; sleep 1; done; exit 1"])
    if ok.returncode != 0:
        sys.exit(f"[app] server on {args.host} did not come up; see {args.remote_dir}/logs/realtime.log")

    local = free_port(args.port)
    tunnel = subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                               "-L", f"{local}:127.0.0.1:{args.port}", args.host])
    url = f"http://127.0.0.1:{local}/"
    try:
        for _ in range(50):                      # wait for the tunnel
            try:
                urllib.request.urlopen(url + "api/config", timeout=1)
                break
            except OSError:
                time.sleep(0.2)
        print(f"[app] tunnel {url} -> {args.host}:{args.port}; opening window")
        webview.create_window("BreastSIM Probe Lab", url, width=1600, height=1000, min_size=(1100, 700))
        webview.start()
    finally:
        tunnel.terminate()
        print("[app] window closed, tunnel closed (server still running on the host)")


if __name__ == "__main__":
    main()
