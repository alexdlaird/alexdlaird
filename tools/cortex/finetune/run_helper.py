__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import sys
import time
from pathlib import Path

_BANNER_WIDTH = 80


def banner(label):
    """Print a loud visual banner — use at the start/end of long-running scripts."""
    bar = "#" * _BANNER_WIDTH
    print(f"\n{bar}", flush=True)
    print(f"##  {label}", flush=True)
    print(f"{bar}\n", flush=True)


def follow(proc, log_path, label):
    """Stream a background process's log to stdout until it exits, then exit with its code.

    Replaces os.execlp("tail", ...) — fixes the race where tail's poll interval
    causes final buffered output to be missed when the process exits quickly.
    On KeyboardInterrupt, detaches without killing the background process.
    Never returns.
    """
    log_path = Path(log_path)
    banner(f"{label.upper()} — STARTING")
    try:
        with open(log_path) as f:
            while True:
                chunk = f.read()
                if chunk:
                    print(chunk, end="", flush=True)
                if proc.poll() is not None:
                    # Process exited — wait briefly for final OS buffer flush, then drain.
                    time.sleep(0.2)
                    remainder = f.read()
                    if remainder:
                        print(remainder, end="", flush=True)
                    break
                time.sleep(0.05)
    except KeyboardInterrupt:
        banner(f"{label.upper()} — DETACHED")
        print(f"    PID {proc.pid} is still running in background.", flush=True)
        print(f"    Reattach: tail -f {log_path}\n", flush=True)
        sys.exit(0)

    if proc.returncode == 0:
        banner(f"{label.upper()} — DONE")
    else:
        banner(f"{label.upper()} — FAILED  (exit {proc.returncode})")
    sys.exit(proc.returncode)
