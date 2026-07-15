import warnings
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "relative_path",
    [
        "commodutil/dates.py",
        "commodutil/forwards.py",
        "commodutil/stats.py",
        "commodutil/forward/util.py",
    ],
)
def test_regex_modules_compile_without_invalid_escape_warnings(relative_path):
    source_path = Path(__file__).parents[1] / relative_path
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        warnings.simplefilter("error", SyntaxWarning)
        compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
