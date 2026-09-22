from __future__ import annotations

import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalized(value: Any, *, strip_punctuation: bool = False) -> str:
    text = unicodedata.normalize("NFKC", as_text(value)).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".。;；,，:：")
    if strip_punctuation:
        text = re.sub(r"[\s\.,;:!?(){}\[\]\\\"']+", "", text)
    return text


def decimal_from(value: Any) -> Decimal | None:
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", as_text(value))
    if not match:
        return None
    try:
        number = Decimal(match.group(0))
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def parse_choice_label(value: Any) -> str | None:
    text = as_text(value).strip()
    match = re.match(r"^(?:option\s*)?([A-D])(?:[\s\.:\)\-]|$)", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def parse_boolean(value: Any) -> bool | None:
    text = normalized(value, strip_punctuation=True)
    if text in {"true", "yes", "1", "correct"}:
        return True
    if text in {"false", "no", "0", "incorrect"}:
        return False
    return None

