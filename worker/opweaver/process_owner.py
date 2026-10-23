"""Stop disposable inference work when its supervising Worker disappears."""

import os
import subprocess
import threading
import time


def terminate_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        # Windows venv python.exe can be a launcher with a real interpreter child.
        # Terminate its tree before the launcher disappears and loses that link.
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
        if process.poll() is None:
            process.kill()
    else:
        process.kill()


def watch_owner(pid, cancellation=None):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if not handle:
            os._exit(72)

        def alive():
            return kernel.WaitForSingleObject(handle, 0) == 258
    else:

        def alive():
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False

    def monitor():
        while alive() and not (cancellation and cancellation.exists()):
            time.sleep(0.2)
        os._exit(72)  # Even if ORT is inside native code, do not publish after owner loss.

    threading.Thread(target=monitor, daemon=True).start()
