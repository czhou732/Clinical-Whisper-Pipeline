"""One version everywhere."""

import re
from pathlib import Path

from version import __version__


def test_pyproject_matches_version_py():
    toml = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', toml, re.M).group(1) == __version__


def test_provenance_reports_the_same_version():
    import provenance
    assert provenance.APP_VERSION == __version__
