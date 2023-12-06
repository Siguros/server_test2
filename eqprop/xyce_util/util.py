import subprocess

import numpy as np


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def partition(pred, prefix, iterable):
    "Use a predicate to partition entries into false entries and true entries"
    trues = []
    falses = []
    [
        (trues.append((key, val)) if pred(key, prefix) else falses.append((key, val)))
        for (key, val) in iterable
    ]
    return trues, falses


def startswith(s, *args):
    return s.startswith(args)


def _xyce_available() -> bool:
    """Check if a package is available in your environment."""
    try:
        # The 'which' command looks for the executable in the system's PATH.
        # This can be used on Unix-like systems including Linux and macOS.

        # you should check your Xyce install path

        result = subprocess.run(
            ["which", "Xyce"],
            capture_output=True,
            check=True,
        )
        if result.returncode == 0:
            return True
        else:
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


_XYCE_AVAILABLE = _xyce_available()
