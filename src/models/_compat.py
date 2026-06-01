"""Library-version shims shared by the heavy GluonTS-based models (M2/M3).

These keep the **modern data contract** (pandas >= 2.2 offset aliases) working against
the **pinned older GluonTS 0.13** that the Colab heavy group installs. Pure-stdlib and
light-env safe, so the wrappers import this at module top level — unlike the gluonts /
pts imports they defer into methods.
"""

from __future__ import annotations

# pandas >= 2.2 lowercased the sub-daily offset aliases (H->h, S->s, ms/us/ns), but
# GluonTS 0.13's ``get_lags_for_frequency`` / ``time_features_from_frequency_str``
# dispatch on the LEGACY uppercase names and raise "invalid frequency" on the lowercase
# ones (this is exactly what Electricity's freq="h" trips). We translate only at the
# GluonTS boundary so the config/contract can keep the modern alias. Notes:
#   * "D"/"W"/"B" (daily/weekly/business) were NOT lowercased -> pass through unchanged.
#   * "min" (minutely) is accepted by GluonTS 0.13 as-is -> not remapped.
#   * GluonTS spells milli/micro/nano "L"/"U"/"N" (its own legacy aliases).
_SUBDAILY_LEGACY = {"h": "H", "s": "S", "ms": "L", "us": "U", "ns": "N"}


def gluonts_freq(freq: str) -> str:
    """Map a modern pandas offset alias to the spelling GluonTS 0.13 understands.

    Splits an optional leading integer multiplier from the unit, so both a bare alias
    (``"h"`` -> ``"H"``) and a multiplied one (``"2h"`` -> ``"2H"``) are handled; anything
    GluonTS already accepts (``"D"``, ``"min"``, ...) is returned unchanged.

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
