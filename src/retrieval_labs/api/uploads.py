"""Validation of uploaded contracts: UTF-8 ``.txt`` only, size-limited, text kept verbatim.

The text is decoded but never normalized (no newline, whitespace or Unicode changes), so the
offsets in reports refer to exactly what was uploaded. Filenames are labels only: nothing is
ever written to disk under an uploaded name.
"""

from __future__ import annotations

import os
import re
import unicodedata

from retrieval_labs.api.errors import ApiError

DEFAULT_MAX_UPLOAD_BYTES = 2_000_000
ALLOWED_CONTENT_TYPES = {"text/plain", "application/octet-stream", ""}
MAX_LABEL_LENGTH = 255
_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")


def max_upload_bytes() -> int:
    return int(os.environ.get("MAX_UPLOAD_BYTES", str(DEFAULT_MAX_UPLOAD_BYTES)))


def safe_filename(filename: str | None) -> str:
    """A display label: last path component, unsafe characters replaced, no leading dots."""
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = unicodedata.normalize("NFKC", name)
    name = _UNSAFE.sub("_", name).strip(" .")
    return name[:MAX_LABEL_LENGTH] or "upload.txt"


def check_file_type(filename: str | None, content_type: str | None) -> None:
    if not (filename or "").lower().endswith(".txt"):
        raise ApiError(415, "unsupported_file_type", "only .txt files are accepted")
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if media_type not in ALLOWED_CONTENT_TYPES:
        raise ApiError(415, "unsupported_file_type", "only text/plain uploads are accepted")


def decode_contract(data: bytes, limit: int) -> str:
    if len(data) > limit:
        raise ApiError(413, "file_too_large", f"the file exceeds the {limit}-byte limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ApiError(422, "invalid_encoding", "the file is not valid UTF-8 text") from None
    if "\x00" in text:
        raise ApiError(422, "invalid_text", "the file contains NUL characters")
    if not text.strip():
        raise ApiError(422, "empty_file", "the file contains no text")
    return text


def clean_title(title: str | None, filename: str) -> str:
    """The submitted title, or the filename without its extension."""
    value = (title or "").strip() or filename.rsplit(".", 1)[0] or filename
    if len(value) > MAX_LABEL_LENGTH:
        raise ApiError(422, "invalid_title", f"the title exceeds {MAX_LABEL_LENGTH} characters")
    if any(unicodedata.category(c) == "Cc" for c in value):
        raise ApiError(422, "invalid_title", "the title contains control characters")
    return value
