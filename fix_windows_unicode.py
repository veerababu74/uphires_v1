"""
Windows Unicode Logging Fix
===========================

This script helps fix Unicode encoding issues on Windows systems
by setting proper environment variables and encoding settings.
"""

import os
import sys
import locale


def fix_windows_unicode():
    """Fix Windows Unicode encoding issues for console output."""
    try:
        # Set UTF-8 encoding for Python
        os.environ["PYTHONIOENCODING"] = "utf-8"
        os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"

        # Set console code page to UTF-8 if on Windows
        if sys.platform.startswith("win"):
            try:
                # Try to set console to UTF-8
                os.system("chcp 65001 > nul 2>&1")
            except Exception:
                pass

        # Set locale to UTF-8 if possible
        try:
            locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
        except Exception:
            try:
                locale.setlocale(locale.LC_ALL, "")
            except Exception:
                pass

    except Exception as e:
        print(f"Warning: Could not fully configure Unicode support: {e}")


def configure_safe_logging():
    """Configure logging to be safe for Windows console."""
    import logging

    # Create a custom formatter that strips Unicode characters if needed
    class SafeFormatter(logging.Formatter):
        def format(self, record):
            # Get the formatted message
            msg = super().format(record)

            # If on Windows and there are Unicode issues, clean the message
            if sys.platform.startswith("win"):
                try:
                    # Try to encode as cp1252 (Windows default)
                    msg.encode("cp1252")
                except UnicodeEncodeError:
                    # Remove problematic Unicode characters
                    msg = msg.encode("ascii", errors="ignore").decode("ascii")

            return msg

    # Configure the root logger with safe formatter
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log", encoding="utf-8"),
        ],
    )

    # Apply safe formatter to all handlers
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(
                SafeFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )


if __name__ == "__main__":
    print("Fixing Windows Unicode issues...")
    fix_windows_unicode()
    configure_safe_logging()
    print("Unicode fixes applied!")
