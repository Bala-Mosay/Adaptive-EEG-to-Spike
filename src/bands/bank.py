"""Seven-band filter bank used by the Phase 3 adaptive selector.

The paper describes a 7-band bank spanning 4--32 Hz.  The bank is
implemented as seven contiguous 4 Hz bands (a 4 Hz step, not the 2 Hz
step mentioned in an earlier slide).  A 2 Hz step would produce 13
overlapping bands, which is not what the "7-band" description means.

Frequency names are descriptive only; the selector ranks bands by their
KNN-MI score, not by name.
"""
from typing import List, Tuple

#: Seven non-overlapping 4 Hz bands covering 4--32 Hz.
SEVEN_BANDS: List[Tuple[str, float, float]] = [
    ("theta", 4.0, 8.0),
    ("mu", 8.0, 12.0),
    ("low_beta", 12.0, 16.0),
    ("beta", 16.0, 20.0),
    ("high_beta", 20.0, 24.0),
    ("low_gamma", 24.0, 28.0),
    ("gamma", 28.0, 32.0),
]


def normalize_bands(bands: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """Validate and normalize a band definition list.

    Parameters
    ----------
    bands : list of (name, low_hz, high_hz)
        Band definitions, typically loaded from YAML as nested lists.

    Returns
    -------
    list of (str, float, float)
        Normalized bands with float frequency bounds.

    Raises
    ------
    ValueError
        If a band is malformed or ``low_hz >= high_hz``.
    """
    out: List[Tuple[str, float, float]] = []
    seen = set()
    for item in bands:
        if len(item) != 3:
            raise ValueError(f"Band must have three entries (name, low, high), got {item!r}")
        name = str(item[0])
        low = float(item[1])
        high = float(item[2])
        if not low < high:
            raise ValueError(f"Band {name!r} has non-increasing bounds: {low} >= {high}")
        if name in seen:
            raise ValueError(f"Duplicate band name: {name!r}")
        seen.add(name)
        out.append((name, low, high))
    if not out:
        raise ValueError("At least one band is required")
    return out
