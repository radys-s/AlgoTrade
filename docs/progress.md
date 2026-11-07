# Progress log

Append-only deník kroků, experimentů a výsledků (i neúspěšných). Podklad pro prezentaci
a pro review od Qminers. Časy místní (CET), 7.–8. 11. 2026.
Okna/větve: `radys` = Tomáš, `herzic` = Honza. Každý přidává řádky a sekce, staré nepřepisuje.

Kontext soutěže: [kontext.md](kontext.md).

## Časová osa

| Datum čas | Okno/branch | Krok |
|---|---|---|
| 2026-11-07 10:15 | radys / main | Import starteru, repo a větve `main`, `radys`, `herzic`, kontext soutěže |
| 2026-11-07 10:22 | herzic | Honza: uv + `uv sync`, napojení na týmové repo, referenční běh `pnl=1878` ověřen |
| 2026-11-07 11:05 | radys | Analýza playground dat (`research/analyze_tape.py`), grid runner, `momentum_hold.py`: sample `pnl=50 965` |
| 2026-11-07 11:15 | radys | Playground submise `momentum_hold` (výchozí parametry): server `pnl=44 116` = lokálně přesně; výchozí parametry přepnuty na nejlepší z gridu |
| 2026-11-07 11:20 | radys → main | Submise s nejlepšími parametry: server `pnl=50 965` (final). Předávka `docs/poznatky.md`, merge `radys` do `main` |
| 2026-11-07 11:20 | herzic | Research infrastruktura: paralelní runner + ledger, compare, check, MM baseline, workflow, skills |
| 2026-11-07 11:25 | herzic | Data/ML pipeline: features (1 průchod simulátorem), data card, walk-forward signály, export modelu; sample: ridge IC +0.79 (7/7 foldů) |
| 2026-11-07 11:40 | radys → main | Added `submission/best_strategy.py` (the single file to upload, always the current best) + `submission/README.md` |
| 2026-11-07 11:45 | herzic → main | Sjednocení nástrojů: runner ladí i konstanty třídy (`QTY`, `HOLD_US`), ledger přeskakuje výstupy `grid.py`, champion = `submission/best_strategy.py`. Sample plató: HOLD 5–8 s všech 8/8 chunků kladných (8 s: pnl 52 029, holdout 12 209; 5 s: 50 965, holdout 12 622), ledger tag `sample-momentum-plateau` |
| 2026-11-07 11:50 | herzic | `momentum_hold_v2`: konec dat relativní a vypnutý; v1 na pásce od 10 h neobchoduje vůbec a na 2h pásce stojí po hodině; v2 → `submission/best_strategy.py` |
| 2026-11-07 11:50 | home | Domácí výpočetní uzel (desktop Core Ultra 7 265KF, 20 vláken, RTX 5060 Ti 16 GB): ověřené referenční běhy, benchmark 20 workerů = 12,8 plných běhů/min, CUDA torch env, MLP test IC +0,80 |
| 2026-11-07 12:10 | herzic | Pipeline kola `research.round` + taker scan + MM edge; generálka na `sample_v2`: momentum -2,44 M, MM +28 025 (8/8), trh mean-reverting |
| 2026-11-07 12:15 | radys | Playground v2: momentum loses (-1.52M server). Passive MM on Honza's `mm.py` core, sweep -> `mm_v2` pnl 237,306, now `submission/best_strategy.py` |
| 2026-11-07 12:30 | radys | Tuning on mm_v2 core: skew, unwind, one-sided jump, 2nd level all <= champion; fill markout shows ~no adverse selection, volume-limited |
| 2026-11-07 12:20 | herzic | Practice round: 3. místo (+240 995, server ≈ lokál ±1,5 %); MM sweep na `sample_v2`: velikost dominuje, microprice skew +11 % |
| 2026-11-07 12:30 | home | Generálka `research.round --fetch sample_v2`: 144 s end-to-end (kroky ~2,2× rychlejší než laptop), baseliny sedí; přesný runtime championa 14,1 s / 2 M řádků → 65 s na 8,8 M; sweep kolem `mm_v2` (162 halving + 144 + 8 běhů): výchozí parametry na hraně mřížky, `imb_pull` 0,875–0,975 plató +1,2–1,8 %, ale compare REJECT (4/8 chunků) |
| 2026-11-07 13:20 | radys | Round 1: mm_v2 321,024 vs exp_full_capacity 425,708 (+33%) -> submission |

## 2026-11-07 10:15 — Založení repa (radys)

**Co a proč:** veřejný starter `QminersOrg/qminers-algo-arena` (upstream commit `3f5bea7`)
importován jako první commit do soukromého týmového repa — stejný výsledek jako
„Use this template“, ale do repa, které už existovalo. Remote `upstream` zůstává pro
případné aktualizace starteru od organizátorů. Založeny větve `radys` a `herzic` z `main`.

**Proč je to důležité:** pravidla vyžadují jedno týmové repo s nepřepsanou historií
(commit každých 30 min, bez squash/rebase/force-push). Práce v osobních větvích
a merge do `main` drží historii čitelnou.

**Další krok:** `uv sync`, rozbalit `data/sample.csv.gz`, ověřit referenční běh
`trend_follower.py` (`pnl=1878`), pozvat `antoninkriz` (Read), stáhnout playground data.

## 2026-11-07 10:22 — Setup u Honzy (herzic)

**Co:** nainstalované `uv` 0.13.7, `uv sync` (Python 3.13 + balíčky), rozbalený `data/sample.csv`.
Lokální složka napojená na týmové repo (`origin` = `radys-s/AlgoTrade`, `upstream` = starter),
práce ve větvi `herzic`.

**Výsledek:** referenční běh `trend_follower.py` na `sample.csv` → `fills=305,
breach_penalty=0.00, pnl=1878.00` (shodné s workshopem). Plné logy ~12 s, `--no-logs` rychleji.

**Další krok:** stáhnout playground data z dashboardu, první vlastní strategie.

## 2026-11-07 11:05 — Analýza playground dat a první experiment (radys, playground)

**Zjištění o datech:** playground `sample.csv` z dashboardu je bajtově shodný s `data/sample.csv.gz`
ze starteru (syntetický trend). Starter README říká, že kola jsou nahraná reálná data a vypadají jinak →
závěry níže jsou o samplu, ne o kolech. Hodnota je hlavně v nástrojích, které se pustí na data každého kola.

**Nástroje (znovupoužitelné každé kolo):**
- `research/analyze_tape.py <csv>` — statistiky pásky, knihy a prediktivní síla signálů (book se staví
  stejným simulátorem jako hodnocení přes pasivní recorder). Na samplu běží 3 s.
- `research/grid.py <strategie> <csv> PARAM=a,b,c ...` — paralelní grid, výsledky do `research/results/`,
  PnL rozpadlé na čtvrtiny času (kontrola, že zisk není z jednoho úseku).

**Sample v číslech:** 53 758 řádků / 1 h (15 událostí/s), 98 % Insert jsou G (limit), jen 2 % IOC.
Spread **vždy 2 ticky**, na každé straně touch typicky jediný příkaz ~25 lotů. Zrušené příkazy žijí medián 190 ms.
Mid se pohyboval 9 867–10 073. Autokorelace 1s změn midu +0,59 (lag 1) → silné krátkodobé momentum.

| signál (korelace s budoucí změnou midu) | 10 ms | 100 ms | 1 s | 5 s |
|---|---|---|---|---|
| imbalance L1 / L1-3 / microprice | +0,03 | +0,03 | +0,06 | +0,08 |
| tok obchodů EWMA | +0,39 | +0,46 | **+0,65** | +0,39 |

Po obchodu jde mid ve směru agresora o +0,5 ticku za 100 ms, **+3,7 za 1 s, +10 za 5 s**.
Maker by na každém fillu bral 1 tick (půl-spread) a ztrácel ~3,7 → market making na samplu prodělává;
imbalance je bezcenná (na každé straně jen jeden příkaz).

**Experiment `strategies/momentum_hold.py`:** IOC vstup po směru toku, držet `HOLD_US`, pak IOC výstup;
inventář jen z fillů, strop `MAX_POS=80`. Grid QTY × HOLD_US × SLIP_TICKS (40 kombinací, 7 s):

| varianta | pnl | pnl_se | fills | penalty | q997 abs(I) | čtvrtiny |
|---|---|---|---|---|---|---|
| reference `trend_follower` (1 lot) | 1 878 | 102 | 305 | 0 | 1 | — |
| QTY=50, HOLD=5 s, SLIP=3 | **50 965** | 2 678 | 913 | 0 | 45 | 12,9k / 11,5k / 13,9k / 12,6k |
| QTY=25, HOLD=5 s, SLIP=0 | 49 827 | 2 539 | 908 | 0 | 41 | 11,9k / 11,5k / 13,8k / 12,6k |
| QTY=50, HOLD=10 s, SLIP=3 | 49 589 | 3 143 | 845 | 0 | 58 | 11,2k / 11,6k / 17,0k / 9,8k |

**Co fungovalo:** delší držení (5 s ~ horizont, kde je momentum nejsilnější) a větší velikost; zisk je
rovnoměrný ve všech čtvrtinách. QTY nad ~25 už skoro nepomáhá — likvidita na touch je ~25 lotů.
**Co ne / rizika:** HOLD 0,5 s je výrazně horší. Na reálných datech momentum nemusí existovat → stejný
grid i `analyze_tape.py` pustit na data Round 1 hned po stažení; strategie je baseline, ne finále.

**Další krok:** na datech Round 1 nejdřív `analyze_tape.py` (spread, momentum vs. reverze, imbalance),
pak podle toho grid; Honza vede návrh strategie, radys reviewuje.

## 2026-11-07 11:15 — První submise na serveru (radys, playground)

Nahraná `momentum_hold.py` omylem s výchozími parametry (QTY 20, HOLD 3 s, SLIP 2), ne s nejlepšími z gridu.
Server: `pnl=44 116`, fills 1 000, objem 11 932, pokuta 0, |I| q99,7 = 36, simulace 0,9 s.
Lokální běh se stejnými parametry dal **přesně stejná čísla** → lokální simulátor = hodnocení,
lokální grid je spolehlivý odhad veřejného skóre. Výchozí parametry přepnuty na QTY 50, HOLD 5 s, SLIP 3 (lokálně 50 965).

## 2026-11-07 11:20 — Research infrastruktura (herzic)

**Co a proč:** v 5h kole je úzké hrdlo CPU na backtesty a disciplína proti overfittingu, ne
nápady. Postaveno (plán kola a protokol v [workflow.md](workflow.md)):
- `research/run.py` — paralelní sweep strategie × mřížka parametrů × datasety; PnL po 8 časových
  chuncích + holdout (posledních 25 % času); `--halving` (successive halving přes prefixy dat);
  každý běh do ledgeru `research/results/` se snapshotem strategie. `research/ledger.py` výpis.
- `research/compare.py` — champion vs challenger po párových chuncích, verdikt ACCEPT/REJECT.
- `research/check.py` — statická kontrola pravidel sandboxu + odhad doby běhu na 8,8 M řádků.
- `strategies/mm.py` — parametrický market maker s dokázaným invariantem |I| + čekající ≤ 100.
- `research/train_mlp.py` — PyTorch MLP (Colab/GPU) → váhy jako numpy kód do strategie.
- `research/workflows/research_wave.js` — vlna výzkumu se subagenty (návrh → experimenty → skeptik → integrace).
- `.claude/skills/` — vlastní skills (mechanika simulátoru, kickoff kola, experiment, pre-submit)
  + 5 prověřených cizích (walk-forward validace, mikrostruktura, backtest review, sklearn, statsmodels).

**Výsledek (sample, syntetický trend):** runner reprodukuje `pnl=1878`; MM s výchozími parametry
`pnl=-9056`, breach 0 (trend na samplu MM přejede do 1 s po fillu, 77 % fillů); halving na 32
sadách najde dobré, ne nutně nejlepší nastavení → halving na široký scan, pak úzký plný sweep.

**Další krok:** playground = sample (viz radys 11:05), takže první reálná data jsou Round 1: kickoff podle `.claude/skills/round-kickoff`, data card, MM i `momentum_hold` na nových datech.

## 2026-11-07 11:50 — momentum_hold_v2: konec dat nezávislý na délce kola (herzic, playground)

Hypotéza: absolutní `FLATTEN_AT_US = 3 590 s` je riziko na reálných kolech jiné délky nebo s jiným
začátkem času; dřívější zavírání nic nepřináší, protože markout na konci je sám IOC likvidace do knihy.
Změna: `strategies/momentum_hold_v2.py` — `FLATTEN_AFTER_US` měřený od první události, výchozí 0 (vypnuto).
Data: `sample.csv` + zátěžové pásky v `data/stress/` (sample posunutý na 10 h; sample 2× za sebou = 2 h).
Výsledek (ledger `flatten-v1-stress`, `flatten-v2-stress`), breach 0 všude:

| páska | v1 `momentum_hold` | v2 výchozí | v2 se zavíráním po 3 590 s |
|---|---|---|---|
| sample 1 h | 50 965 | **51 225** | 50 965 (= v1, parita) |
| posunutý na 10 h | **0** (vůbec neobchoduje) | 51 225 | 50 965 |
| 2 h | 50 965 (2. hodinu stojí) | **99 665** | 50 965 |

Závěr: funguje; v2 povýšena do `submission/best_strategy.py` (research.check PASS, odhad 1,4 min na
8,8 M řádků). `research.compare` by dal REJECT (trénovací chunky jsou totožné, liší se jen konec) —
vědomý override: jde o robustnost, ne o alfu.
Další krok: na datech Round 1 ověřit momentum vs. reverze (`analyze_tape.py`, data card) dřív, než v2 nahrajeme.

## 2026-11-07 11:50 — Domácí výpočetní uzel (home)

**Co:** Honzův domácí desktop jako výpočetní uzel týmu, větev `home` (z `main` c65737a), commity jako
`Herz1c-home`. Stroj: Intel Core Ultra 7 265KF (20 jader / 20 vláken), 31,6 GB RAM,
RTX 5060 Ti 16 GB (driver 616.92), 462 GB volného disku; git 2.55, gh 2.100, uv 0.12.14, Python 3.13.15.

**Ověření simulátoru (sample):** `trend_follower` pnl 1 878, `submission/best_strategy.py` pnl 50 965
(hold 12 622, 8/8 chunků), `research.check` PASS (odhad 33 s na 8,8 M řádků).
Pozn.: `python -m gzip` nezná `-k`; `-d` vstupní `.gz` nemaže, stačí `-d`.

**Benchmark paralelismu:** sample (0,5 s/běh) měří jen start procesů, proto páska velikosti kola:
sample 160× za sebou (časy, ID a ceny posunuté) = 8,6 M řádků; 1 běh `mm.py` = 76 s (~113 tis. událostí/s).
Jedna vlna, běhů = workerů:

| workerů | wall | s/běh | plných běhů/min | RAM špička |
|---|---|---|---|---|
| 8 | 80 s | 76,5 | 6,0 | 1,0 GB |
| 12 | 84 s | 80,9 | 8,6 | 1,4 GB |
| 16 | 88 s | 84,0 | 10,9 | 1,9 GB |
| 20 | 94 s | 88,0 | 12,8 | 2,3 GB |

Paměť ~150 MB/proces, nehraje roli. **Limit uzlu: 20 workerů celkem** (18, když vedle běží features
nebo trénink); E-jádra zpomalí běh jen o ~15 %.

**GPU:** separátní `%USERPROFILE%\torch-env` (Python 3.13, torch 2.11.0+cu128, numpy 2.5.4; projektové
prostředí zůstává shodné s arénou). CUDA ok (sm_120). Sample dataset 13,8 tis. řádků × 23 features
(h = 1 000 ms; ridge OOS IC +0,79, HGB +0,80, 7/7 foldů): MLP h32, 30 epoch, batch 512 →
GPU test IC +0,803 za 21 s, CPU test IC +0,801 za 11 s. Na tak malých datech je CPU rychlejší
(start CUDA); GPU má smysl až na datech celého kola a větších sítích.

**Další krok:** Round 1 (13:00): stáhnout data z release `data-r1`, merge `origin/main` → `home`,
features + data card na celém kole a široké sweepy s `--halving` na 20 workerech.

## 2026-11-07 12:10 — Pipeline kola + generálka na nových playground datech `sample_v2` (herzic)

**Co:** `research/round.py` = jeden příkaz na start kola (rozbalení, privátní release pro domácí PC,
baseliny + check championa, analyze_tape, features, data card, signály, taker scan, MM edge, kickoff
report s verdiktem pojistné submise). Nové nástroje: `research/taker_scan.py` (rychlý přibližný
backtest taker strategií nad npz, + `strategies/signal_taker.py` se stejnou logikou), `research/mm_edge.py`
(kdy se vyplácí kotovat na touch) a filtr adverse selection v `strategies/mm.py` (`as_filter`, vypnutý).

**Data:** organizátoři vyměnili playground data: `sample_v2`, 2 000 000 řádků, 1,6 h, úplně jiný trh
než syntetický sample. Celá pipeline 6 min (z toho upload ~6 min souběžně, analýza ~5 min).

**Výsledek:**
- Champion `momentum_hold_v2`: **-2 440 965**, 0/8 chunků → na realistických datech momentum pálí peníze.
- `mm.py` výchozí: **+28 025**, 8/8 chunků, holdout +7 084; `trend_follower` -531.
- Trh je mean-reverting (P(reverze další změny midu) = 0,63), spread medián 2 ticky (1 tick jen 41 %),
  1s volatilita 0,5 ticku. Taker scan: 1 152 konfigurací za 43 s, žádná kladná.
- MM edge: maker na čele fronty +0,77 ticku/fill, 8/8 chunků; žádný filtr nepřekoná „kotovat vždy“.
- Signály: ridge IC +0,41 / HGB +0,42 na 1 s (HGB nic nepřidává → lineární stačí);
  nejstabilnější `micro_minus_mid`, ale predikovaný pohyb << spread → na skew kotací, ne na taking.

**Závěr:** syntetický `sample.csv` vyřazen z robustnostní sady (penalizoval by vše, co funguje na
realistických datech). Pro Round 1 je výchozí rodina market making; momentum champion bez nových
dat nesubmitovat.
**Další krok:** sweep MM na `sample_v2` (běží, tag `sample_v2-mm-wave0`), pak vlna 1 na domácím PC.

## 2026-11-07 12:20 — Practice round: 3. místo; MM sweep na `sample_v2` (herzic)

**Practice round (leaderboard):** tým `p<0.05` **3. místo, +240 995** s `mm_v2` (Tomáš). Lokálně na
veřejném `sample_v2` 237 306 → server vs. lokál ~1,5 % → soukromá data se chovají jako veřejná, lokální
ladění se přenáší. Vedoucí: ThatsMyQuant +291 456, Podbranč +286 942 (náskok ~20 %). Naše PnL křivka
roste plynule celým kolem (zisk rozložený v čase, ne jeden úsek). Spousta týmů je v mínusu.

**Sweep `mm.py` na `sample_v2`** (36 kombinací, ledger tag `sample_v2-mm-wave0`), všechny 8/8 chunků, breach 0:

| quote_size | skew_per_lot | alpha_imbalance | improve | pnl | holdout |
|---|---|---|---|---|---|
| 20 | 0.1 | 1.0 | off | **39 704** | 9 059 |
| 20 | 0.1 | 0 | off | 35 661 | 9 001 |
| 20 | 0.1 | 1.0 | 3 | 38 804 | 8 836 |
| 5 | 0.05 (výchozí) | 0 | off | 28 025 | 7 084 |

Závěr: velikost kotace dominuje (Tomášův `mm_v2` se 100 loty: 237 306); zlepšování ceny uvnitř spreadu
(`improve_min_spread` 3) je v každém páru mírně horší → vypnuto; microprice skew (`alpha_imbalance` 1.0)
při větší velikosti +11 % a lepší holdout → kandidát na `mm_v2` (tam je 0).
Další krok: domácí PC ladí `mm_v2` (quote_size, max_position 80–100, skew, alpha_imbalance, imb_pull) s halvingem.

## Šablona experimentu

```
## YYYY-MM-DD HH:MM — <název> (radys|herzic, kolo N)
Hypotéza:
Změna (soubor, parametr/flag):
Data: public backtest kola N
Výsledek: pnl=…, fills=…, breach_penalty=…, max|I|=…  (předtím: …)
Závěr: funguje / nefunguje, proč
Další krok:
```

## 2026-11-07 12:15 — Playground v2: from taker momentum to passive market making (radys)

**What and why:** the organizers swapped the playground data (`sample_v2.csv`). `analyze_tape.py` shows a different
market: spread 1 tick 41% / 2 ticks 58% of the time, deep touch (~770 lots), no momentum after trades (slight
reversion, -0.03 tick), L1 imbalance corr with forward mid +0.35 at 100 ms. Momentum IOC pays the spread on every
trade and the price does not follow, so `momentum_hold` scored -1,521,297 on the server. A passive LIMIT quote
earns the spread instead.

**Experiments (all in the ledger `research/results/2026...radys-s_*`):**
- `strategies/mm_imbalance.py` (my own MM, imbalance pull filter): grid best 112,983.
- `strategies/exp_imb_pull.py` = Honza's `mm.py` + `imb_pull` / `inv_stop` flags (off = identical to mm.py).
  36-run sweep: the pull filter HURTS on this core (best with pull 151k vs 195k without); skew_per_lot 0.05 halves pnl.
- Size sweep (24 runs): pnl grows smoothly with quote_size and max_position; requote_tolerance 1 and
  alpha_imbalance 1 both worse. Best: quote_size 100, max_position 100, skew 0 -> **pnl 237,306**, holdout 54,929,
  8/8 chunks positive, penalty 0, runtime projection 2.2 min for 8.8M rows.

**Risk:** |I| sits at the 100 cap 32%+ of the time (q68 = 100). No penalty is possible (mm.py invariant proves
|I| <= max_position), but a trending private tape would leave us holding 100 lots against the move.
A max_position 80 variant (219,994) is the safer fallback.

**Next:** Round 1 data at 13:00 -> rerun the same sweep (quote_size, max_position, skew) on it before submitting.

## 2026-11-07 12:30 — Why mm_v2 is capped at ~237k on sample_v2 (radys)

**Experiments on the champion (ledger `v2-skew-unwind-jump`, `v2-layers`), all at quote 100 / max_pos 100:**

| change | best pnl | vs 237,306 |
|---|---|---|
| skew_per_lot 0.01 / 0.02 | 159k / 118k | -33% / -50% |
| IOC unwind at |I| >= 90 (10 lots) | 212,680 | -10% |
| one-sided +1 tick at spread 2 when |imb| > 0.6 (`jump_imb`) | 235,223 | -1% |
| second level 20-40 lots, 1-2 ticks behind (`strategies/exp_layers.py`) | 237,237 | 0% (layer 0 = exact parity) |

**Fill analysis (14,819 fills, 453k lots):** edge vs mid at fill 0.61 $/lot; markout 10 ms 0.48, 1 s 0.50,
60 s 0.53 -> almost no adverse selection, the price even reverts slightly in our favour. Spread-2 fills earn
0.94 $/lot, spread-1 fills 0.38. So pnl ~= lots x 0.52: the strategy is **volume-limited**, not
selection-limited. Anything that moves us off the touch (skew, pull, deeper levels) loses volume and pnl.

**Penalty mechanics (simulator.py `_accrue_breach_penalty`):** prorated per microsecond, 10,000 $/s = 10 $/ms
while |I| > 100. Going over the cap for extra volume is not worth it at ~0.5 $/lot.

**Next ideas for more volume:** queue position (don't lose priority on requote), quoting earlier into a fresh
spread-2 level, different sizes per spread state. Re-test everything on Round 1 data first.

## 2026-11-07 12:30 — Generálka kola a sweep kolem championa `mm_v2` (home, sample_v2)

**Pipeline kola:** `uv run python -m research.round --fetch sample_v2` (data předtím odsunutá, tedy včetně stažení):
**144 s celkem** (stažení + rozbalení ~6 s). Lane A (baseliny + check) 45 s, lane B (analyze_tape 31, features 19,
profile 7, signals 26 + 25, taker_scan 28, mm_edge 3 s) 138 s = kritická cesta. Kroky ~2,2× rychlejší než laptop.
Baseliny shodné s laptopem: `mm.py` +28 025, `momentum_hold_v2` −2 440 965 (ledger `sample_v2-baseline-home`),
champion `mm_v2` 237 306. Report: `research/reports/sample_v2_kickoff_home.md`.

**Runtime championa bez zátěže:** `research.check --rows 0` → 14,07 s na 2 M řádků (2,9 µs/událost nad prázdnou
strategií), odhad **65 s na 8,8 M řádků** (sandbox 15 min). Laptopových 5–8,6 min byl odhad pod zátěží.

**Hypotéza:** champion `mm_v2` (quote 100, max_position 100, skew 0) není v optimu; menší kotace, skew nebo
adverse-selection filtr `imb_pull` zlepší PnL i holdout.
**Změna:** žádná v kódu, jen overrides `submission/best_strategy.py`.
**Data:** sample_v2 (sample je syntetický, mimo robustnost).
**Výsledek:**
- `home-mmv2-sweep1`: 162 buněk, halving 0.1,0.3 → 20 přeživších, 62 s na 20 workerech. Nejlepší = výchozí parametry.
- `home-mmv2-sweep2-narrow`: plná mřížka 144 buněk (quote 70/85/100, max_position 95/100, skew 0/0.003/0.006,
  improve 0/3, imb_pull 0/0.7/0.8/0.9), 170 s, breach 0 všude. quote a max_position rostou monotónně až k hraně
  (100 = strop); skew 0.003 = 0 (zaokrouhlení posunu na celé ticky), 0.006 pnl −14 %; improve 3 ≈ 0 (−0,5 %).
- `home-mmv2-imbpull-fine`: imb_pull 0.82–0.99 → hladké plató 0.875–0.975 (240–241,7k, max 0.925: 241 670,
  hold 56 504), 0.82–0.85 a 0.7 horší než bez filtru; filtr ubere 15–35 % fillů, runtime +25 %.

| set | pnl | hold | chunky | chunk_min | fills | inv_q997 | runtime |
|---|---|---|---|---|---|---|---|
| champion (imb_pull 0) | 237 306 | 54 929 | 8/8 | 22 010 | 10 333 | 100 | 17 s |
| imb_pull 0.925 | 241 670 | 56 504 | 8/8 | 22 160 | 8 206 | 100 | 21 s |
| imb_pull 0.95 | 241 387 | 56 898 | 8/8 | 22 083 | 8 873 | 100 | 21 s |
| imb_pull 0.9 | 240 146 | 56 650 | 8/8 | 21 744 | 7 668 | 100 | 21–25 s |
| imb_pull 0.9, improve 3 | 239 114 | 55 996 | 8/8 | 21 636 | 7 626 | 100 | 24 s |

`research.compare` champion vs `{"imb_pull": 0.9}`: **REJECT**, B vyhrává jen 4/8 párových chunků (+2 840 pnl,
+1 721 hold, zisk soustředěný ve 3 chuncích).
**Závěr:** champion je na sample_v2 lokálně optimální v testovaných osách; `imb_pull` ≈ 0.9–0.95 je malé plató
(+1–2 %), ale neprojde protokolem → nepovyšovat. Pozor: inv_q997 = 100, champion jezdí až na limitu pozice.
**Další krok:** na Round 1 stejné osy + `alpha_imbalance` (imbalance L1 má na sample_v2 IC +0,35 při 100 ms–1 s)
a `inv_stop`/`unwind_threshold` kvůli pozici na limitu.

## 2026-11-07 12:45 — Před Round 1: tři rychlé hypotézy nad championem na `sample_v2` (home, mozek)

**Hypotéza H1 (`full_capacity`):** champion `mm_v2` je limitovaný objemem (~0,5 $/lot) a třetinu času sedí
na stropu |I| = 100. Invariant (B) I + OUT_buy ≤ M, (S) −I + OUT_sell ≤ M dovoluje straně, která |I| snižuje,
až M + |I| lotů (200 při |I| = 100), ale `quote_size` 100 ji kapoval na 100.
**Změna:** žádný kód, jen `quote_size` 200 (`_quote` posílá min(quote_size − have, free), takže naprázdno
kotuje 100/100 jako dřív a při I = +60 má ask 160). Zapečeno v `strategies/exp_full_capacity.py`.
**Data:** sample_v2 (ledger `home-mmv2-qsize-over`).
**Výsledek:** quote_size 100 → 237 306 | 125 → 244 627 | 150 → 252 437 | **200 → 263 618 (+11,1 %)** | 300 → 263 618
(saturace na M + |I|). Holdout 54 929 → 61 917 (+12,7 %), chunk_min 22 010 → 24 340, fills 10 333 → 8 116
(méně fillů, větší), breach 0, inv_q997 100. `research.compare` champion vs `{"quote_size": 200}`:
**ACCEPT**, B vyhrává 8/8 párových chunků (+2 141 až +2 802 každý), hold +6 988. `research.check --rows 0`: PASS,
68 s projekce na 8,8 M řádků.
**Závěr:** funguje; čistě mechanická páka bez nové mechaniky a bez nového rizika (invariant beze změny,
|I| ≤ 100 dál platí). Kandidát na pojistnou submisi R1, pokud compare na R1 public řekne ACCEPT.

**Hypotéza H3 (`improve_unwind`, `strategies/exp_improve_unwind.py`):** při |I| ≥ N a spreadu ≥ 2 kotovat
odvíjející stranu o tick uvnitř (čelo prázdné hladiny) → rychleji flat, uvolněná kapacita zaplatí půl tick.
**Výsledek** (ledger `v2-improve-unwind`, parita 237 306 ✔, check PASS): N = 100 → 198 395 (−16 %), 90 → 196 187,
80 → 192 529, 60 → 177 449, 40 → 163 992; fills +1,5 %. **Nefunguje:** filly uvnitř spreadu jsou adverzní
(cena jde skrz), a to i když se spouští jen na stropu.

**Hypotéza H4 (`hold_inside`, `strategies/exp_hold_inside.py`):** když trh ustoupí od naší hladiny (jsme sami
o tick uvnitř nového touch), zůstat na čele místo návratu na konec hluboké fronty; trh se na úrovni ticku vrací
(P = 0,63). **Výsledek** (ledger `v2-hold-inside`, parita ✔, check PASS): hold 1/2/3 ticky → shodně 210 974
(−11 %) při 13 110 fillech (+27 %), holdout 49 501. **Nefunguje:** víc fillů, každý o třetinu horší; ústup
trhu je informovaný a držená kotace je první na ráně.

**Poučení:** edge je jen na touch; páka je velikost na touch (H1), ne cena. `improve`, `skew`, `hold_inside`
i `improve_unwind` shodně prodělávají.
**Další krok:** po kickoffu R1 compare champion vs `exp_full_capacity` na round1 + sample_v2; CPU sweep
championa na R1 (quote_size 100/200 × imb_pull × alpha_imbalance × min_requote_us, max_position 80/90).

## 2026-11-07 13:15 — Round 1: `exp_full_capacity` ACCEPT (+33 %), sweep os = nic dalšího nepomáhá (home, mozek)

**Režim R1** (data card `research/reports/round1.md`, 4,41 M řádků, 3,33 h): spread 1 tick 93 % času (sample_v2 41 %),
touch ~900 lotů / 17 příkazů, mid ~náhodná procházka (P(reverze) 0,46, lag-1 autokorelace +0,08), po obchodech
mírné momentum (flow_100ev IC +0,16 na 100 ms, +0,24 na 1 s, 8/8), imb_l1 IC +0,30 na 100 ms. mm_edge: maker na
čele fronty +0,25 ticku/fill na 1 s (efektivní půlspread 0,53, impact −0,28), joiner na konci fronty (fill jen při
vyprázdnění hladiny) **−0,22**, „swept beyond“ −0,37; edge podle lotů před nami Q: 0 → +0,25, 100 → +0,10,
200 → 0, 500 → −0,21. Taker scan 1 152 konfigurací: nic (+206, 1/8). Baseliny: champion 321 024, mm.py 27 626.

**Compare champion vs `strategies/exp_full_capacity.py` (round1 primární + sample_v2):** round1 **321 024 → 425 708
(+104 684, +33 %)**, holdout 82 271 → 110 280, B vyhrává 8/8 párových chunků (+7,8 k až +11,3 k), fills 42 193 →
30 548; sample_v2 jako dřív (+26 312, 8/8). **ACCEPT.** `research.check` na round1 PASS, 24 s, projekce 49 s na 8,8 M.
Notebook výsledek nezávisle reprodukoval. Efekt je na R1 silnější než na sample_v2 (víc IOC toku → víc depletion
fillů, kde se kapacita 200 na odvíjející straně uplatní). Honza: nespěchat se submisí (10 slotů, server = lokál).

**Sweep os kolem kandidáta na round1** (ledger `r1-fullcap-axes`, 36 buněk, všechny 8/8, breach 0): výchozí
nastavení vyhrává všechno. max_position 90 −4 %, 80 −9 %; imb_pull 0,95 −4,6 %, 0,9 −11 %; min_requote_us 100 ms
−14 %; alpha_imbalance 1,0 **−44 %** (při spreadu 1 jen sundá stranu z touch). `r1-fullcap-imbpull-low`: imb_pull
0,8 → 329 519 (−23 %), 0,7 → 271 936, 0,6 → 227 837, 0,5 → 190 913 (−55 %), monotónně. `r1-hold-inside-cap200`:
hold_inside 1 → 389 859 (−8,4 %). `r1-jump-imb-cap200` (Tomášův exp_imb_pull, pozor: jeho výchozí hodnoty dávají
jinou základnu 203 144): jump_imb 0,6 −1,5 %, 0,3 −3,7 %.

**Závěr:** na R1 platí ještě silněji než na sample_v2: **cokoliv, co nás sundá z touch nebo nás posune dovnitř,
prodělává**, včetně imbalance filtru, který mm_edge doporučuje (+0,33 vs +0,25 na fill). Důvod: filtr pull
odstraňuje právě naše nejlepší filly (čelo svěží hladiny = tenká vlastní fronta = „imbalance proti nám“) a při
návratu jsme zase na konci fronty. Páka zůstává objem na touch a pozice ve frontě.
**Další krok:** H5 `exp_queue_join` (vstupovat jen do tenkých hladin / pull při hluboké frontě), H6 `exp_flow_pull`,
fill attribution kandidáta na round1 (`research/reports/round1_fill_attribution.md`).
## 2026-11-07 13:20 — Round 1 kickoff (radys)

The browser download of `round1.csv` was corrupt (two runs of zero bytes, 27 MB, 33-76% of the file); the clean
copy came from Honza's release `data-round1` (4,411,337 rows, 12,000 s). Check downloads for NUL bytes first.

Ledger `r1-champ-vs-fullcap` (exp_imb_pull with filters off = mm_v2 / exp_full_capacity):

| quote_size | max_position | pnl | holdout |
|---|---|---|---|
| 100 | 80 | 317,240 | 81,548 |
| 100 | 100 | 321,024 (= mm_v2, current champion) | 82,271 |
| 200 | 80 | 387,053 | 99,956 |
| 200 | 100 | **425,708** (= `exp_full_capacity.py`) | 110,280 |

Full simulator run of `strategies/exp_full_capacity.py`: pnl 425,708, fills 30,548, penalty 0, q997 |I| = 100,
check_submission OK. Higher than mm_v2 on the same data -> promoted to `submission/best_strategy.py`.
Next: tuning sweep on R1 (20% halving) around quote 200 / max_pos 100.

## 2026-11-07 13:20 — H6 `flow_pull`: pull strany proti toku obchodů nefunguje (home, mozek, R1)

**Hypotéza:** na R1 má podepsaný tok obchodů kladné IC s dalším pohybem midu (flow_100ev +0,16 na 100 ms, +0,24 na 1 s);
když tok běží proti naší straně, stáhnout ji (nekupovat těsně před poklesem).
**Změna:** `strategies/exp_flow_pull.py` = exp_full_capacity + `flow_pull` (lotů; 0 = off) a `flow_window_ev`
(klouzavý součet podepsaného objemu posledních N taker příkazů, O(1)); OR k `imb_pull`. Parita 425 708 ✔, check PASS.
**Data:** round1 (+ sample_v2 v compare). Prahy = percentily |flow| p70–p95 (3 720 / 4 710 / 6 640 / 8 670).
**Výsledek** (ledger `r1-flow-pull`, 8 buněk): monotónně horší s častějším pullem: 3 720/100ev → 358 472 (−16 %,
−20 % fillů); nejméně pullující buňka 8 670/30ev → 425 412 (−296), compare vs exp_full_capacity: B vyhrává **0/8**
chunků, holdout stejný → REJECT. Na sample_v2 se flag při těchto prazích nespustí (263 618 = parita).
**Závěr:** stažené filly nejsou netto adverzní: získaný spread převáží ~0,28 ticku post-fill driftu a pull navíc
ztrácí prioritu ve frontě. Potvrzuje vzorec z os: žádný pull filtr na R1 nepomáhá.
**Další krok:** tok případně jako posun velikosti (modify zmenšuje a drží místo ve frontě), ne jako pull.

## 2026-11-07 13:25 — H5 `queue_join`: vstup jen do tenkých hladin / pull při hluboké frontě nefunguje (home, mozek, R1)

**Hypotéza:** mm_edge na R1: edge fillu klesá s loty před nami (Q 0 → +0,25, 200 → 0, 500 → −0,21), filly z konce
fronty při vyprázdnění hladiny −0,22. Kotovat stranu jen tam, kde dostaneme dobrou pozici: `join_max_qty` (vstoupit
jen když tržní množství na touch ≤ N) a `max_queue_ahead` (stáhnout, když je před nami > N lotů).
**Změna:** `strategies/exp_queue_join.py` = exp_full_capacity + oba flagy (0 = off). Parita 425 708 ✔, check PASS.
**Data:** round1, compare i sample_v2 (ledger `r1-queue-join`, 18 buněk).
**Výsledek:** monotónně horší, čím přísnější: join_max_qty 1800 → 415 545 (−2 %), 1200 → 404 831, 800 → 380 547
(−11 %), 400 → 306 998, 200 → 239 496 (−44 %); max_queue_ahead 150 samo → 167 333 (−61 %), 300 → 241 237, 1000
→ −9 %. Compare {join_max_qty 800}: round1 −45 161 (B 0/8 chunků), sample_v2 −26 528 (0/8) → REJECT.
**Závěr:** filly z konce fronty jsou pro nás netto kladné (fill attribution: queue_ahead 201+ dává +0,12 $/lot,
sweepy ≥ 100 lotů +0,10 $/lot), protože příkazy před nami se ruší (medián života 0,34 s) a naše pozice se zlepšuje
čekáním; mm_edge měří statickou pozici při fillu a to podhodnocuje. Pull navíc po návratu zase skončí na konci fronty.
**Další krok:** rozklad času bez živé kotace (74 % času živá kotace na stranu, strop vysvětlí jen ~17 %) a společná
tabulka imbalance × pozice ve frontě z fill attribution; pak rozhodnout, zda zbývá jiná páka než `exp_full_capacity`.

## 2026-11-07 13:40 — H7 `deep_pull` REJECT; fill attribution kandidáta na R1: strop drží 50 % času (home, mozek)

**H7 (`strategies/exp_deep_pull.py`, ledger `r1-deep-pull`):** pull strany jen při imbalance proti ní **a** > N lotů
před nami (jediný záporný segment z fill attribution: imbalance < −0,5 × queue_ahead 201+ = −46 k na 276 k lotů).
Parita 425 708 ✔, check PASS. 9 buněk: jen 0,85/400 je +217 (4/8 chunků, holdout −70, šum), ostatní −0,3 až −12 %;
compare → REJECT. Stažený tok byl v průměru ziskový, i když podmnožina měřená markoutem při fillu vypadala záporně:
pravidlo pro zrušení příkazu musí být podmíněné stavem při odeslání, ne při fillu.

**Fill attribution (`research/reports/round1_fill_attribution.md`, `research/fill_attribution.py`), opraveno o 131
taker fillů (limitky, které v 5 ms letu překřížily trh):** kandidát sedí na |I| = 100 **50,4 % času** (champion
28 %); čas bez živé kotace na stranu 26 % = strop 25 % + in-flight 0,6 % + jiné 0,2 % → **žádná kódová cesta
s mrtvým časem**, dřívější podezření bylo chybou rekonstrukce inventáře. Markout 1 s ≈ PnL; edge při fillu
0,505 $/lot, markout 0,16; filly snižující pozici +437 k, filly zvyšující ≈ 0; |I| ≥ 90 vydělává nejvíc na lot
(0,24). Filly z konce fronty kladné (+0,12 $/lot), sweepy ≥ 100 lotů +0,10.
**Závěr:** limit objemu = strop pozice (polovina času jedna strana mrtvá). Všechny pokusy o rychlejší odvíjení za
cenu spreadu prodělaly; zbývá zkusit (a) asymetrickou velikost (menší kotace na straně, která |I| zvyšuje, aby
sweep nehodil pozici na strop jedním fillem), (b) hold_inside jen pro odvíjející stranu při |I| ≥ N.
