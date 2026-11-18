"""Best verified local R2 candidate kept outside the protected strategy directory.

This matches the actual repo run in research/results/20261107-222544_radys-s_r2-radys-fq5-spread.csv,
which scored 360,791 locally on round2 and is the strongest real candidate in the current tree.

It is intentionally not a speculative 1M idea: it is the strongest local evidence with the right
server calibration behavior (+7.8% on the same fill/volume pattern as the UI measurement).
"""

from __future__ import annotations

from pathlib import Path

from strategies.exp_r2_radys_fq5 import MarketMaker as ZBaseMarketMaker


class TopMaxCandidate(ZBaseMarketMaker):
    """Exact verified local candidate from the repo's strongest fq5 spread sweep."""

    PARAMS = {
        **ZBaseMarketMaker.PARAMS,
        "fair_drop_clamped": 0,
        "gate_window_us": 15_000_000,
        "take_k_spread": 0.1,
        "take_persist": 0,
    }


if __name__ == "__main__":
    print(Path(__file__).resolve())
    print(TopMaxCandidate.PARAMS)
