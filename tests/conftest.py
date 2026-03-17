"""
Pytest configuration for MIRROR test suite.

Excludes standalone integration/smoke test scripts that are meant to be
run directly (python tests/test_api.py) rather than via pytest.
"""

collect_ignore = [
    "test_api.py",
    "test_gemini_quick.py",
    "list_gemini_models.py",
    "verify_nvidia_models.py",
]
