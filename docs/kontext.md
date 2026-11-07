# Algo Trade Arena 2026 — kontext soutěže

Podklad pro práci i pro prezentaci. Zdroje: web soutěže (Technical info), workshop
„Getting started“ (PDF, 51 slidů) a veřejné repo
[QminersOrg/qminers-algo-arena](https://github.com/QminersOrg/qminers-algo-arena).
Pořádá Qminers + MFF UK, IMPAKT Praha, 7.–8. 11. 2026.

Tým: **Tomáš** (`radys-s`, branch `radys`) + **Honza** (`Herz1c`, branch `herzic`).

## Úloha jednou větou

Napsat jednu Python třídu (`BaseStrategy`), která obchoduje **jeden anonymní instrument**
na simulované burze proti přehrávanému záznamu cizích příkazů. Skóre = PnL.

## Harmonogram

| Kdy | Co |
|---|---|
| So 09:30–10:30 | Workshop |
| So 10:30–12:00 | Playground (zkušební kolo) |
| So 13:00–18:00 | **Round 1** (váha 0,8) |
| So 19:00–24:00 | **Round 2** (váha 1,0) |
| Ne 01:00–06:00 | **Round 3** (váha 1,0) |
| Ne 07:00–12:00 | **Round 4** (váha 1,2, bez průběžného leaderboardu) |
| Ne 13:00–14:00 | Vyhlášení |

## Jak probíhá kolo

1. Vývoj lokálně na **veřejném backtestu** kola (dashboard → *Download public backtest data* → `data/`, rozbalit).
2. Upload jednoho `.py` na dashboard (arena.qminers.com). Server ho pustí na stejných veřejných datech — jen ověření, že kód běží v sandboxu.
3. Max **10 submisí na kolo**, jedna je *final* (výchozí = poslední úspěšná, dá se změnit do konce kola). Neúspěšná submise nikdy nepřepíše funkční final.
4. Po uzavření se finální strategie vyhodnotí na **privátním backtestu** → leaderboard.

**Pozor:** lokální PnL je na veřejných datech, hodnotí se na privátních → hlídat overfitting.

## Bodování

`Score = 0.8·z1 + 1.0·z2 + 1.0·z3 + 1.2·z4`

- `z_r` = PnL v kole standardizované vůči ostatním týmům (kolik směrodatných odchylek nad průměrem).
- Týmy s PnL ≤ 0 (i bez submise) se do průměru a std nepočítají a dostanou jako by PnL = 0.
- Důsledek: **záporné PnL nebolí víc než nula** v rámci kola, ale stabilně kladné PnL ve všech kolech je cennější než jeden zásah. Pozdější kola váží víc.

## Burza (simulátor = stejný kód jako hodnocení)

- **Páska (tape):** CSV `Time,Operation,OrderId,Side,Volume,Price,Lifespan`; `Insert`/`Cancel`; `B`/`A`; `G` = limit (resting), `F` = IOC. Páska je pevná a **na nás nereaguje**.
- **Matching:** price-time priorita, fill za cenu resting (maker) příkazu. Naše příkazy nikdy neobchodují mezi sebou. **Bez poplatků.**
- **Latence 5 ms:** `send_order` / `cancel_order` / `modify_order` se uplatní před první tržní událostí s časem ≥ t + 5 ms. Cancel může prohrát závod s fillem → inventář počítat **z fillů**, ne z odeslaných příkazů.
- **Typy příkazů:** `LIMIT` (maker, vydělá spread) a `IOC` (taker, platí spread). Market order = IOC přes knihu.
- **Callbacky:** `on_order_book_change(snapshot, trades)` po každé tržní události (book do hloubky 20 ticků, `owner` = `market`/`user`); `on_private_order_update(event)` pro naše příkazy (`ACCEPTED`, `PARTIALLY_FILLED`, `FILLED`, `CANCELED`, `MODIFIED`, `REJECTED`).
- **Limity:**
  - `|I + s·(R + q)| > 1000` → příkaz `REJECTED`.
  - `|I| > 100` → pokuta **10 000 $ za sekundu** mimo pásmo.
- **Skóre:** `PnL = cash + markout(I) − penalty`; markout = likvidace inventáře do resting tržních příkazů protistrany (posouvá se po knize → velký inventář na konci se prodá draze).
- Callback běží pro každý řádek pásky (sample má 53 758 řádků) → callbacky musí být lehké; limit sandboxu 15 min, 1 CPU, 2 GB.

## Pravidla submise

- Jeden soubor ≤ 1 MiB, jedna podtřída `BaseStrategy` bez argumentů konstruktoru.
- Povoleno: stdlib Python 3.13, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, `simulator.models`, `simulator.strategy.BaseStrategy`.
- Zakázáno (odmítnuto před spuštěním): `os, sys, io, socket, subprocess, threading, importlib, inspect, gc, pathlib`, `open, eval, exec, getattr, type`, dunder atributy (kromě `__init__`), `str.format`, privátní moduly, jiné importy ze `simulator`.
- Obcházení sandboxu = diskvalifikace.

## Povinné repo (hodnotí se i výzkum)

- Commit **aspoň každých 30 minut**, i bez zlepšení.
- Commitovat i **neúspěšné experimenty** s poznámkou, co se zkusilo a co vyšlo → `docs/progress.md`.
- **Žádný squash, rebase ani force-push** — historie se čte tak, jak vznikla. (Do `main` tedy merge, ne rebase.)
- Mechaniky nemazat, schovat za parametr/flag — v dalším kole se můžou hodit.
- Pozvat `antoninkriz` s **Read** přístupem (Settings → Collaborators) — ideálně před Round 1.
- Kvalitní výzkum si Qminers všimnou i mimo top 3.

## Příkazy

```bash
uv sync
uv run python -m gzip -d data/<file>.csv.gz
uv run exchange-simulator --strategy strategies/<s>.py --backtest data/<file>.csv --log-dir logs/<run> --progress
uv run exchange-simulator --strategy strategies/<s>.py --backtest data/<file>.csv --log-dir logs/<run> --no-logs   # jen skóre, rychle
uv run backtest-explorer --log-dir logs/<run>   # http://127.0.0.1:8050, potřebuje plné logy
```

Výstup: `logs/<run>/result.json` (`pnl`, `cash`, `inventory`, `breach_penalty`, `fills`, `pnl_se`, …) a `pnl.csv` (PnL po sekundách).
Referenční běh: `trend_follower.py` na `sample.csv` → `fills=305, breach_penalty=0, pnl=1878`.

## Výchozí nápady (k ověření, ne fakta)

- Sample je syntetický trend; skutečná kola jsou nahraná tržní data a **vypadají jinak**.
- Market making (resting LIMIT na obou stranách) + inventory skew, tvrdá ochrana pásma ±100.
- Order-flow imbalance / trade flow jako signál směru; dávat pozor na adverse selection (fill → cena proti nám).
- Na konci kola stáhnout inventář k nule (markout drahý při velké pozici).
