"""Analýza pásky jednoho kola: statistiky trhu a prediktivní síla jednoduchých signálů.

Použití (z kořene repa):
    uv run python research/analyze_tape.py data/sample.csv

Spouští knihu přes stejný simulátor jako hodnocení (pasivní "recorder" strategie, nic neobchoduje),
takže stav knihy odpovídá tomu, co uvidí naše strategie. Výsledek tiskne jako markdown.
Lokální výzkumný nástroj, NENÍ to submise (používá pandas/sys bez omezení sandboxu).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from simulator.models import OrderBookSnapshot, Trade
from simulator.simulator import ExchangeSimulator, SimulationConfig
from simulator.strategy import BaseStrategy

HORIZONS_US = [10_000, 100_000, 1_000_000, 5_000_000]  # 10 ms, 100 ms, 1 s, 5 s


class Recorder(BaseStrategy):
    """Zaznamená top of book a toky obchodů po každé tržní události."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple] = []

    def on_order_book_change(self, snapshot: OrderBookSnapshot, trades: list[Trade]) -> None:
        bids, asks = snapshot.bids, snapshot.asks
        if not bids or not asks:
            return
        bb, ba = bids[0].price, asks[0].price
        bq1 = sum(o.quantity for o in bids if o.price == bb)
        aq1 = sum(o.quantity for o in asks if o.price == ba)
        bq3 = sum(o.quantity for o in bids if o.price > bb - 3)
        aq3 = sum(o.quantity for o in asks if o.price < ba + 3)
        flow = sum(t.quantity if t.book_side == 'ask' else -t.quantity for t in trades)
        tvol = sum(t.quantity for t in trades)
        self.rows.append((snapshot.timestamp_us, bb, ba, bq1, aq1, bq3, aq3, flow, tvol, len(trades)))


def record(csv_path: Path) -> pd.DataFrame:
    rec = Recorder()
    ExchangeSimulator(strategy=rec, config=SimulationConfig()).run(csv_path)
    df = pd.DataFrame(rec.rows, columns=['ts', 'bb', 'ba', 'bq1', 'aq1', 'bq3', 'aq3', 'flow', 'tvol', 'ntr'])
    df['mid'] = (df.bb + df.ba) / 2
    df['spread'] = df.ba - df.bb
    return df


def forward_mid(df: pd.DataFrame, h_us: int) -> np.ndarray:
    """Mid v prvním okamžiku >= t + h (NaN za koncem dat)."""
    ts = df.ts.to_numpy()
    idx = np.searchsorted(ts, ts + h_us, side='left')
    out = np.full(len(df), np.nan)
    ok = idx < len(df)
    out[ok] = df.mid.to_numpy()[idx[ok]]
    return out


def tape_stats(csv_path: Path) -> None:
    t = pd.read_csv(csv_path)
    ins = t[t.Operation == 'Insert']
    can = t[t.Operation == 'Cancel']
    print('## Páska')
    print(f'- řádků {len(t):,}, délka {t.Time.iloc[-1]:.0f} s, událostí/s {len(t) / t.Time.iloc[-1]:.1f}')
    print(f'- Insert {len(ins):,} (G {(ins.Lifespan == "G").mean():.0%}, F/IOC {(ins.Lifespan == "F").mean():.0%}), Cancel {len(can):,}')
    print(f'- velikost Insert: medián {ins.Volume.median():.0f}, p90 {ins.Volume.quantile(.9):.0f}, max {ins.Volume.max()}')
    print(f'- cena: min {ins.Price.min()}, max {ins.Price.max()}')
    life = can.merge(ins[['OrderId', 'Time']], on='OrderId', suffixes=('', '_ins'))
    life = life.Time - life.Time_ins
    if len(life):
        print(f'- životnost zrušených G příkazů: medián {life.median() * 1000:.0f} ms, p10 {life.quantile(.1) * 1000:.0f} ms, p90 {life.quantile(.9):.1f} s')


def book_stats(df: pd.DataFrame) -> None:
    dt = np.diff(df.ts.to_numpy(), append=df.ts.iloc[-1])
    sp = df.spread.to_numpy()
    print('\n## Kniha (časově vážené)')
    for s in sorted(set(sp))[:6]:
        print(f'- spread {s}: {dt[sp == s].sum() / dt.sum():.1%} času')
    print(f'- objem na touch: bid medián {df.bq1.median():.0f}, ask medián {df.aq1.median():.0f}')
    sec = df.set_index(pd.to_datetime(df.ts, unit='us')).mid.resample('1s').last().ffill()
    r = sec.diff().dropna()
    print(f'- mid: start {df.mid.iloc[0]}, konec {df.mid.iloc[-1]}, min {df.mid.min()}, max {df.mid.max()}')
    print(f'- volatilita 1s změny midu: std {r.std():.2f} ticku; autokorelace lag1 {r.autocorr(1):+.3f}, lag5 {r.autocorr(5):+.3f}')
    tr = df[df.ntr > 0]
    print(f'- obchodních callbacků {len(tr):,}, objem {df.tvol.sum():,}, buy-initiated podíl {(tr.flow > 0).mean():.0%}')


def signal_stats(df: pd.DataFrame) -> None:
    print('\n## Prediktivní síla (korelace s budoucí změnou midu)')
    imb1 = (df.bq1 - df.aq1) / (df.bq1 + df.aq1)
    imb3 = (df.bq3 - df.aq3) / (df.bq3 + df.aq3)
    micro = (df.bb * df.aq1 + df.ba * df.bq1) / (df.bq1 + df.aq1) - df.mid
    flow_ewm = df.flow.ewm(halflife=20).mean()
    feats = {'imbalance L1': imb1, 'imbalance L1-3': imb3, 'microprice-mid': micro, 'flow EWMA(20 ev.)': flow_ewm}
    print('| signál | ' + ' | '.join(f'{h / 1000:g} ms' for h in HORIZONS_US) + ' |')
    print('|---|' + '---|' * len(HORIZONS_US))
    fwd = {h: forward_mid(df, h) - df.mid.to_numpy() for h in HORIZONS_US}
    for name, f in feats.items():
        cells = []
        for h in HORIZONS_US:
            m = ~np.isnan(fwd[h]) & np.isfinite(f.to_numpy())
            cells.append(f'{np.corrcoef(f.to_numpy()[m], fwd[h][m])[0, 1]:+.3f}')
        print(f'| {name} | ' + ' | '.join(cells) + ' |')

    print('\n## Po obchodu: průměrný pohyb midu ve směru agresora (ticky)')
    print('Kladné = momentum (taker měl pravdu, maker prodělal = adverse selection), záporné = reverze.')
    tr = df.flow != 0
    sign = np.sign(df.flow[tr]).to_numpy()
    for big in (False, True):
        sel = (df.tvol[tr] >= df.tvol[tr].quantile(.9)).to_numpy() if big else np.ones(tr.sum(), bool)
        cells = [f'{np.nanmean(sign[sel] * fwd[h][tr.to_numpy()][sel]):+.2f}' for h in HORIZONS_US]
        label = 'velké obchody (p90+)' if big else 'všechny obchody'
        print(f'- {label}: ' + ', '.join(f'{h / 1000:g} ms {c}' for h, c in zip(HORIZONS_US, cells)))
    half = df.spread[tr].mean() / 2
    print(f'- průměrný půl-spread při obchodu: {half:.2f} ticku (zisk makera před adverse selection)')


def main() -> None:
    csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/sample.csv')
    print(f'# Analýza `{csv_path.name}`\n')
    tape_stats(csv_path)
    df = record(csv_path)
    book_stats(df)
    signal_stats(df)


if __name__ == '__main__':
    main()
