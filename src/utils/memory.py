"""Memory management helpers."""


def malloc_trim() -> None:
    """Try to return free heap pages back to the OS (Linux/glibc)."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
