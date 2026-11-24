"""US macro data with point-in-time availability (see docs/data/FUNDAMENTALS_MACRO.md).

Automated sources: Yahoo market macro (`ingest_market_macro`) and BLS API v1 (`ingest_bls`).
FRED is manual-CSV only (`ingest_fred(csv_dir=...)`) because of its terms of use.
"""

from nebula.data.macro.bls import BLS_SERIES, ingest_bls
from nebula.data.macro.features import derived_series, load_macro_tables, macro_features
from nebula.data.macro.fred import LAGS, FredTermsNotAcceptedError, available_from, ingest_fred, load_fred
from nebula.data.macro.fred import macro_features as fred_macro_features
from nebula.data.macro.market import MARKET_SERIES, ingest_market_macro

__all__ = [
    "BLS_SERIES",
    "LAGS",
    "MARKET_SERIES",
    "FredTermsNotAcceptedError",
    "available_from",
    "derived_series",
    "fred_macro_features",
    "ingest_bls",
    "ingest_fred",
    "ingest_market_macro",
    "load_fred",
    "load_macro_tables",
    "macro_features",
]
