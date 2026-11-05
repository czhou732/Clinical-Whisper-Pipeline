"""The one place the ClinicalWhisper version is set.

Shown in the app window, printed by the batch command, written into every
analysis and summary row, and stamped into the app bundle, so a problem
report always says which build produced it. pyproject.toml must match
(tests/test_version.py checks).
"""

__version__ = "5.3.0"
