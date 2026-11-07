from strategies.mm_r2_fq2 import MarketMaker as _BaseMarketMaker


class MarketMaker(_BaseMarketMaker):
    """Best available branch model with higher queue pressure and bigger take volume."""

    PARAMS = {
        **_BaseMarketMaker.PARAMS,
        "quote_size": 250,
        "gate_thr": 9.0,
        "gate_window_us": 30000000,
        "gate_max_spread": 3,
        "fair_thr": 0.0,
        "fair_tau_ms": 70.0,
        "fair_levels": 20,
        "fq_k": 0.05,
        "fq_scale": 1.15,
        "fq_max_behind": 1,
        "fq_inside": 1,
        "fq_take": 250,
        "fq_slip": 0,
        "fq_take_cooldown_ms": 0.0,
        "fq_tol": 1,
        "fq_level": 0,
        "fq_take_k": 0.15,
        "fair_mid_any": 1,
    }
