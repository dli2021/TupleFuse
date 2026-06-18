"""Value canonicalization: Unicode NFKC, lowercase, punctuation strip, whitespace collapse."""
import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")


def norm_text(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    s = unicodedata.normalize("NFKC", s).lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s or None


def norm_number(v):
    """Numeric canonicalization (year, length, price): round to 2 decimals."""
    if v is None:
        return None
    s = str(v).strip().replace(",", ".")
    if not s or s.lower() in ("nan", "none", "null", ""):
        return None
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    try:
        x = float(m.group(0))
    except ValueError:
        return None
    return f"{x:.2f}"
