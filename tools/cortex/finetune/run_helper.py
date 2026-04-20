__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import sys
import time
from pathlib import Path


def follow(proc, log_path, label):
    """Stream a background process's log to stdout until it exits, then exit with its code.

    Replaces os.execlp("tail", ...) — fixes the race where tail's poll interval
    causes final buffered output to be missed when the process exits quickly.
    On KeyboardInterrupt, detaches without killing the background process.
    Never returns.
    """
    log_path = Path(log_path)
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
        print(
            f"\n--> Detached (PID {proc.pid} still running in background)."
            f"\n    Reattach: tail -f {log_path}",
            flush=True,
        )
        sys.exit(0)

    if proc.returncode != 0:
        print(f"\n!!> {label} failed (exit {proc.returncode}). Log: {log_path}", file=sys.stderr, flush=True)
    sys.exit(proc.returncode)
