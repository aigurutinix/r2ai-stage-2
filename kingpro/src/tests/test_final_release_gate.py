from pathlib import Path

import pytest

from scripts.final_release_gate import (
    MAX_UPLOAD_FILENAME_CHARACTERS,
    validate_upload_filename,
)


def test_upload_filename_accepts_exact_portal_limit():
    name = "a" * (MAX_UPLOAD_FILENAME_CHARACTERS - 4) + ".zip"

    report = validate_upload_filename(Path(name))

    assert report["upload_filename_characters"] == 64
    assert report["upload_filename_limit"] == 64
    assert report["upload_filename_within_limit"] is True


def test_upload_filename_rejects_one_character_over_portal_limit():
    name = "a" * (MAX_UPLOAD_FILENAME_CHARACTERS - 3) + ".zip"

    with pytest.raises(RuntimeError, match="portal limit is 64"):
        validate_upload_filename(Path(name))
