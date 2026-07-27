"""Test package marker.

Making the suite a package lets mypy resolve it and lets modules import shared helpers
as ``tests.conftest`` rather than relying on ``sys.path`` manipulation.
"""
