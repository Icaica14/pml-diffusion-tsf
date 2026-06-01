"""Pandas offset-alias compatibility for the heavy GluonTS-based stack (M2/M3).

The Colab heavy group pins **GluonTS 0.13 + pandas < 2.2** (gluonts 0.13's freq
handling predates pandas 2.2). Both that older pandas *and* gluonts 0.13 speak the
**legacy UPPERCASE sub-daily offset aliases** ("H", "S", ...), whereas our data
configs use the **modern lowercase** ones (pandas >= 2.2, e.g. Electricity freq="h").

Two call sites need the legacy spelling, so the translation lives here once:
  * the model wrappers, before handing freq to gluonts / pts; and
  * the loader's :func:`attach_calendar`, as a fallback when pandas < 2.2 rejects the
    modern alias in ``pd.date_range``.

Pure-stdlib and light-env safe, so every layer can import it without pulling heavy deps.
"""

from __future__ import annotations

# pandas >= 2.2 lowercased the sub-daily offset aliases (H->h, S->s, ms/us/ns); pandas
# < 2.2 and GluonTS 0.13 only know the legacy uppercase names and raise on the lowercase
# ones (this is exactly what Electricity's freq="h" trips). Notes:
#   * "D"/"W"/"B" (daily/weekly/business) were NOT lowercased -> pass through unchanged.
#   * "min" (minutely) is understood by both eras as-is -> not remapped.
#   * milli/micro/nano are spelled "L"/"U"/"N" in the legacy scheme.
_SUBDAILY_LEGACY = {"h": "H", "s": "S", "ms": "L", "us": "U", "ns": "N"}


def gluonts_freq(freq: str) -> str:
    """Map a modern pandas offset alias to the legacy spelling (gluonts 0.13 / pandas <2.2).

    Splits an optional leading integer multiplier from the unit, so both a bare alias
    (``"h"`` -> ``"H"``) and a multiplied one (``"2h"`` -> ``"2H"``) are handled; anything
    the legacy scheme already accepts (``"D"``, ``"min"``, ...) is returned unchanged.

    >>> gluonts_freq("h"), gluonts_freq("2h"), gluonts_freq("D")
    ('H', '2H', 'D')
    """
    if not freq:
        return freq
    i = 0
    while i < len(freq) and freq[i].isdigit():
        i += 1
    mult, unit = freq[:i], freq[i:]
    return mult + _SUBDAILY_LEGACY.get(unit, unit)
