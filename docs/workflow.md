# Research workflow pro 5hodinové kolo

Cíl: za 5 hodin otestovat co nejvíc hypotéz, ale do submise pustit jen to, co přežije
protokol proti overfittingu. Hodnotí se na **privátních** datech, my vidíme jen veřejná.

## Tři principy

1. **Úzké hrdlo je CPU, ne tokeny.** Honzův stroj má 14 vláken → ~12 backtestů najednou.
   Simulátor jede ~65–85 tisíc událostí/s, plné kolo (až ~8,8 M řádků) je tedy ~2–3 min na běh.
   Za kolo to dává řádově stovky plných backtestů. Každý agent dostane pevný počet workerů
   (`--workers 2–3`), aby se nepřetahovali.
2. **Každý běh jde do ledgeru.** `research.run` zapisuje každý backtest do
   `research/results/<čas>_<user>_<tag>.csv` (parametry, PnL, holdout, PnL po chuncích, commit)
   a ukládá snapshot strategie do `research/results/strategies/`. Soubory se nikdy nepřepisují →
   žádné merge konflikty mezi větvemi a Qminers vidí celý výzkum, i neúspěšné pokusy.
3. **Champion/challenger s tvrdým protokolem.** `submission/best_strategy.py` je to, co je na aréně.
   Challenger ho nahradí jen když `research.compare` řekne ACCEPT:
   - žádná pokuta za |I| > 100 na žádném datasetu,
   - na aktuálním kole vyhraje aspoň 6 z 8 párových časových chunků (trénovací 75 % času),
   - na holdoutu (posledních 25 % času, při ladění se na něj nekoukáme) je lepší,
   - na předchozích datasetech (playground, minulá kola), kde byl champion kladný, je kladný taky,
   - `research.check` bez ERROR a odhad běhu pro 8,8 M řádků pod 7 min (sandbox: 15 min, 1 CPU).
   Parametry bereme ze středu plató (sousední hodnoty taky dobré), ne z ostrého maxima.

Proč tak přísně: skóre kola je z-score vůči ostatním týmům a PnL ≤ 0 se počítá jako 0.
Stabilně kladné PnL ve všech čtyřech kolech vyhrává; jediná věc, která nás může zabít, je
pokuta 10 000 $/s za |I| > 100 → v každé strategii platí tvrdý invariant pozice.

## Role

| Kdo | Co |
|---|---|
| **Honza + Claude Max** (hlavní stroj) | Orchestrace: research waves se subagenty, integrace championa, ledger, `progress.md`, commity. |
| **Tomáš + Claude Pro** (šetřit limit) | Submise na arénu a správa *final* (přihlásit se může jen člověk), Discord + leaderboard, Colab track (trénink malých sítí na exportovaném datasetu → váhy zpět jako numpy), lidská kontrola fillů championa v exploreru, review před každou submisí. |
| **Google Colab** | GPU pro PyTorch MLP nad `research.signals` exportem. Do strategie jde jen inference v numpy (soubor ≤ 1 MiB, levná na událost). |

Paralelní práce bez konfliktů: každý experiment má vlastní soubor `strategies/exp_<jméno>.py`
(kopie championa + změna za parametrem), `submission/best_strategy.py`, `STATUS.md` a `progress.md`
mění jen integrátor (Honzův hlavní Claude). Claude Max má taky limity za 5h okno — waves po
~6 agentech, ne stovky najednou.

## Časový plán kola (T = začátek kola)

| Čas | Krok | Submise |
|---|---|---|
| přestávka před kolem | Retro minulého kola (privátní výsledek vs náš holdout odhad), champion připravený, merge do `main` | |
| T+0:00–0:10 | Stáhnout public data → `data/`, rozbalit. Spustit kickoff: features + data card + champion a baseliny na novém datasetu | |
| T+0:10–0:20 | Je-li champion kladný na novém public → odeslat jako pojistku | #1 |
| T+0:20–1:20 | **Wave 1**: ~6 hypotéz paralelně (návrh → implementace → sweep → skeptik) | |
| T+1:20 | Integrace, `research.compare` → ACCEPT → submise | #2 |
| T+1:20–2:30 | **Wave 2** (+ Colab model z wave 1 dat) | #3 |
| T+2:30–3:40 | **Wave 3** | #4 |
| T+3:40–4:20 | **Robustness wave**: stres testy, plató parametrů, runtime, check. Freeze. | |
| T+4:20–4:40 | Finální submise, ověřit že serverové PnL na public = lokální, **ručně nastavit final** | #5 |
| T+4:40–5:00 | Rezerva, `progress.md`, commit, merge do `main` | |

Max 10 submisí na kolo: plán 5, 5 v záloze na opravy. Pozor: *final* je defaultně poslední
úspěšná submise — po experimentální submisi vždy zkontrolovat, co je final.
Noční kolo (01:00–06:00): jeden spí, druhý drží submise; Claude jede waves dál.

## Research wave (co dělají agenti)

1. **Návrh**: z data card + ledgeru vybrat ~6 hypotéz s nejvyšší očekávanou hodnotou.
2. **Implementace**: `strategies/exp_<slug>.py` = champion + nová mechanika za flagem v `PARAMS`.
3. **Sweep**: `research.run` přes malou mřížku, aktuální kolo + předchozí datasety, holdout.
4. **Skeptik**: nezávislý agent se snaží výsledek vyvrátit (overfit na jeden chunk? jen jeden dataset?
   artefakt?) a spustí `research.compare` proti championovi.
5. **Integrace**: přijaté změny spojit do nového championa, znovu `compare` + `check`.

Rodiny hypotéz (výchozí zásobník):
- **Market making**: velikost kotace, skew podle inventáře, zlepšení ceny o tick při spreadu ≥ 2, re-quote throttling.
- **Alpha → kotace**: imbalance v knize, microprice, order-flow imbalance, tok obchodů → posun fair value.
- **Adverse selection filtr**: stáhnout stranu, když signál říká, že nás trh přejede.
- **Taker alpha**: IOC jen při silném signálu (> půl spreadu, poplatky nejsou).
- **Mechanika replaye**: páska na nás nereaguje → pozice ve frontě, chování po našich fillech (legální využití pravidel simulátoru; obcházení sandboxu = diskvalifikace, nikdy).
- **Režimy**: volatilita/spread/aktivita → přepínání parametrů.
- **ML**: walk-forward modely z `research.signals`, export lineárního modelu nebo malé MLP.

## Příkazy

```bash
# nová data kola
uv run python -m gzip -d data/<kolo>.csv.gz
uv run python -X utf8 research/analyze_tape.py data/<kolo>.csv   # Tomášův rychlý první pohled (3 s)
uv run python -m research.features data/<kolo>.csv      # dataset pro ML (data/features/, git-ignored)
uv run python -m research.profile data/<kolo>.csv       # data card → research/reports/<kolo>.md

# sweep (zapisuje ledger)
uv run python -m research.run --strategy strategies/mm.py --data data/<kolo>.csv data/<minulé>.csv \
    --grid '{"quote_size": [2, 5, 10], "skew_per_lot": [0, 0.05, 0.1]}' --workers 6 --tag mm-size-skew

# velká mřížka levně: successive halving (10 % dat → nejlepší třetina na 30 % → nejlepší třetina na vše)
# ~3× méně CPU, ale najde DOBRÉ nastavení, ne nutně nejlepší → potom plný sweep na úzké mřížce kolem přeživších
uv run python -m research.run --strategy strategies/mm.py --data data/<kolo>.csv --grid '{...}' --halving 0.1,0.3 --workers 10

# nejlepší běhy z ledgeru (obou členů týmu)
uv run python -m research.ledger --data <kolo> --top 20
uv run python -m research.ledger --robust --data <kolo> <minulé>

# champion vs challenger → ACCEPT/REJECT
uv run python -m research.compare --a submission/best_strategy.py --b strategies/exp_x.py --data data/<kolo>.csv data/<minulé>.csv

# před každou submisí
uv run python -m research.check submission/best_strategy.py --data data/<kolo>.csv

# signály a ML (walk-forward ridge + HGB, export lineárního modelu, dataset pro MLP)
uv run python -m research.signals data/features/<kolo>.npz --horizon-ms 1000
uv run python -m research.signals data/<kolo>.csv --horizon-ms 100 --export-linear
uv run python -m research.signals data/features/<kolo>.npz --horizon-ms 1000 --export-dataset data/features/<kolo>_ml.npz
uv run --with torch python -m research.train_mlp data/features/<kolo>_ml.npz --hidden 32 --epochs 30

# lidská kontrola v exploreru (plné logy pro konkrétní parametry)
uv run python -m research.run --strategy submission/best_strategy.py --data data/<kolo>.csv --full-logs --tag inspect --no-ledger
uv run backtest-explorer --log-dir logs/inspect/000_<kolo>
```

## Git

- Honza pracuje ve `herzic`, Tomáš v `radys`; commit aspoň každých 30 min (ledger, strategie, `progress.md`).
- Po každé submisi merge do `main` → `main` = to, co je na aréně. Žádný rebase/squash/force-push.
- Experiment, který nevyšel, se taky commitne a zapíše do `progress.md` (hypotéza, výsledek, proč).
