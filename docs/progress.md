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
| 2026-11-07 13:45 | radys | R1 tuning on full_capacity: layers/size/imb_pull/jump/flow_pull all <= 426k; fill analysis: inventory-adding fills lose |
| 2026-11-07 14:15 | radys | Queue-position diagnostics on R1; imb/thin/requote pulls all lose; handoff `docs/radys-findings.md` |
| 2026-11-07 14:45 | radys | Park stubs via modify instead of cancel: `mm_park` round1 452,651 (+6.3 %), compare ACCEPT -> submission |
| 2026-11-07 15:10 | radys | k-fold: mm_park not overfit (oof +6.3 %, 8/9 blocks); back-pull and ladder rejected; ridge IC +0.38 stable |
| 2026-11-07 15:40 | radys | ML shrink: ridge 100 ms signal caps the predicted-adverse side at 5 lots -> `mm_ml` round1 490,460 (+8.4 %) -> submission |
| 2026-11-07 16:30 | radys | Rejected on mm_ml: over-cap quoting, inventory lockout/threshold, post-fill freeze, gap jump; structural levers for Honza in radys-findings s.8 |
| 2026-11-07 16:55 | radys | ahead_stub rejected (deep queue = investment); handoff for Honza's Claude in radys-findings s.9 |
| 2026-11-07 17:55 | radys | R1 end: team 511,422 (12th), leader 792,696; stack 503.7k local; fresh gate rejected; R2 plan in radys-findings 9c |
| 2026-11-07 18:40 | radys | Capacity value: no-limit 150/200/300 lots = +101/+137/+181k; IOC unwind with slack rejected (-45k); prompt for Fable |

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

## 2026-11-07 13:17 — H6 `flow_pull`: pull strany proti toku obchodů nefunguje (home, mozek, R1)

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

## 2026-11-07 13:19 — H5 `queue_join`: vstup jen do tenkých hladin / pull při hluboké frontě nefunguje (home, mozek, R1)

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

## 2026-11-07 13:24 — H7 `deep_pull` REJECT; fill attribution kandidáta na R1: strop drží 50 % času (home, mozek)

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

## 2026-11-07 13:29 — H8 `inc_size` a H9 `hold_unwind` REJECT → výzkum R1 uzavřen s `exp_full_capacity` (home, mozek)

**H8 (`strategies/exp_inc_size.py`, ledger `r1-inc-size`):** menší kotace jen na straně, která |I| zvyšuje (odvíjející
strana dál celá kapacita), aby jeden sweep nehodil pozici na strop. Parita ✔, check PASS. inc_size 30 → 413 374
(−2,9 %), 50 → 420 709, 70 → 424 251 (−0,3 %), 100/150 = parita. Compare {70}: round1 −1 457 (1/8 chunků), sample_v2
−548 (1/8) → REJECT. inv_q997 zůstává 100 i při 30: strop se dosáhne opakovanými sweepy; objem sweepů je kladný.
**H9 (`strategies/exp_hold_unwind.py`, ledger `r1-hold-unwind`):** hold_inside jen pro odvíjející stranu při |I| ≥ N.
Parita ✔, check PASS. N 100 → 423 757 (−0,5 %, 3/8 chunků, holdout +24), 80 → 419 833, 50 → 415 875; sample_v2 −0,4 %
(3/8) → REJECT. Většina ztráty hold_inside (−8 %) šla ze strany zvyšující |I|, ale i odvíjející hold je mírně záporný.

**Bilance wave 1+2 na R1 (~110 běhů, 9 mechanik + 5 os):** jediné zlepšení championa je `quote_size` 200 (+33 %).
Všechny filtry (imbalance, tok, fronta, hluboká fronta), throttling, posuny dovnitř spreadu, asymetrické velikosti
i nižší max_position prodělávají, většinou monotónně s intenzitou zásahu. Fill attribution: strop drží 50 % času,
mimo strop je kotace živá; filly z konce fronty i sweepy jsou kladné. **Kandidát R1 = `submission/best_strategy.py`
(= exp_full_capacity, main 1ab3b4d), očekávané veřejné PnL 425 708, check PASS (49 s / 8,8 M).**
**Doporučení:** nahrát nejpozději kolem 17:00 (ověřit server = 425 708, nastavit final); do té doby žádný další
výzkum na R1 mechanikách, šetřit tokeny na R2 (19:00).

## 2026-11-07 13:33 — Layering s plnou kapacitou na R1: neutrální (home, mozek)

**Hypotéza (návrh z notebooku):** část kapacity o 1–2 ticky za touch, abychom po spotřebování touch byli na čele nové
fronty. **Změna:** žádný kód, Tomášův `strategies/exp_layers.py` s overrides championa (quote_size 200, max_position
100, skew 0, book_scan 1000); layer_size 0 = parita 425 708 ✔. **Data:** round1 (ledger `r1-layers-cap200`, 8 buněk).
**Výsledek:** layer 30 @1 tick → 426 034 (+326, +0,08 %), 30 @2 → 425 940, 60 @1 → 425 589 (−119), 100 @1 → 425 705,
100 @2 → 425 648; fills +40 až +100. **Závěr:** v šumu (±0,1 %), stejně jako na sample_v2 (0 %); hlubší hladina se
skoro neplní a nic nepřidá. Neprochází protokolem (zisk < šum). Kandidát beze změny.
## 2026-11-07 13:45 — Round 1 tuning on exp_full_capacity (radys)

**Tape (`analyze_tape.py`):** spread 1 tick 93% of the time, touch ~900 lots, 1 s mid vol 0.92 tick. Unlike v2 the
mid FOLLOWS trades: +0.22 tick after 1 s, +0.52 after p90+ trades (adverse selection for makers). L1 imbalance
corr +0.30 at 100 ms.

**Sweeps (all vs champion 425,708):**
- `r1-tune1` (halving 20%, 32 sets, `exp_layers`): best = layer 30 / quote 300 -> 426,034 (+0.1%, noise);
  quote 300 == 200 (saturated); max_pos 90 -10%; imb_pull 0.5 -70%; jump_imb 0.6 -3%.
- `r1-flow-pull` (`strategies/exp_flow_pull.py`: pull the side in the aggressor's way after a big trade):
  best 1000 lots / 50 ms -> 426,506 (+0.2%); 100 lots / 1 s -> 85,678. Pulling quotes costs more volume than it saves.

**Fill analysis (champion, 50,754 fills, 2.65M lots):** edge vs mid at fill 0.52 $/lot, markout 1 s 0.16, 60 s 0.16.
Spread-2 fills 0.53, spread-1 fills 0.15. Fills that ADD to |I|: 517k lots, 1 s markout -0.015; fills that REDUCE |I|:
2.13M lots, +0.20. By |I| before the fill: [0,30) 0.02, [30,70) 0.08, [70,200] 0.25.
**Next:** `research.mm_edge` and `research.taker_scan` on round1 to see if any filter or taker signal has edge.

## 2026-11-07 14:15 — Queue position is the edge (radys)

Full write-up and tables: [radys-findings.md](radys-findings.md). Ledgers `r1-queue-keep`, `r1-thin-pull`:
requote_tolerance 1 -27 %, imb_pull 0.9 -11 %, thin_pull 30 -8 %. Our fills with 0 lots ahead earn +0.21 $/lot,
with 500+ ahead -0.22..-0.52; any cancel sends us to the back of a ~900-lot queue, which is why every filter loses.
Next: shrink via modify_order instead of cancel; pull only when already at the back.

## 2026-11-07 14:45 — Park a stub instead of cancelling (radys, our model)

**Idea:** every cancel sends us to the back of a ~900-lot queue, and our Q = 0 fills earn +0.21 $/lot. When the
market improves past our quote, `strategies/exp_shrink.py` shrinks the old order with `modify_order` to a stub
(the simulator reduces in place, queue place kept) instead of cancelling it; stubs further than park_depth
ticks behind the touch are cancelled, at most park_max per side.
**Screen** (ledger `r1-shrink`, 15 sets on 25 % of round1, best 5 on full): park 20 / depth 1 **452,651**,
depth 2 447,771, depth 4 447,062; park 10 442,882; park 50 439,606; off 425,708. Plateau, not a spike.
**Fairness check** (`research.compare`, champion exp_full_capacity vs `strategies/mm_park.py`):
round1 +26,943, holdout +7,605, 8/8 paired chunks; sample_v2 -12,382 (-4.7 %), 0/8 chunks. VERDICT ACCEPT.
check_submission OK, research.check PASS, 2.3 min projected for 8.8M rows.
**Result:** `mm_park` -> `submission/best_strategy.py` (not uploaded; Tomáš waits for a bigger jump).
**Caveat:** helps on momentum tapes, costs a little on mean-reverting ones.
**Next:** `exp_backpull` quick screen; heavy experiments listed in [radys-findings.md](radys-findings.md) section 5-6.

## 2026-11-07 13:55 — Ověření `mm_park`, jemný grid parkování, ML taker overlay (home, mozek, R1)

**Ověření Tomášova `mm_park` (submission/best_strategy.py, main 96e3506):** `research.compare` exp_full_capacity vs
mm_park: round1 425 708 → **452 651** (+26 943, holdout 110 280 → 117 886, 8/8 párových chunků +502 až +3 720),
sample_v2 263 618 → 251 236 (−12 382, 0/8, stále kladné) → **ACCEPT** pro R1. `research.check` na round1 PASS,
27 s, projekce 54 s na 8,8 M. Shoda s Tomášem i notebookem na jednotku.

**Jemný grid parkování** (ledger `r1-park-grid`, park_size 10/15/20/30/40 × park_depth 1/2/3 × park_max 2/3/5,
round1 + sample_v2, breach 0 všude): **park_depth 1 je jednoznačně nejlepší** (stub o 2–3 ticky za touch škodí
na obou datasetech, při max 5 až −25 %). round1 plató depth 1: 20/1/5 455 473, 30/1/3 453 321, 15/1/5 453 009,
**20/1/3 452 651 (baked)**, 15/1/3 448 999, 10/1/3 444 526 → rozptyl ±0,6 %, baked hodnoty ve středu.
sample_v2 chce menší stuby: 10/1/2 259 359, 15/1/2 256 696, 20/1/3 251 236 (bez parkování 263 618). Součet
r1+v2 plochý (15/1/5 703 986 ≈ 20/1/3 703 887 ≈ 20/1/5 703 759). **Závěr: parametry neměnit**; pro R2
v mean-reverting režimu zvážit park 10/1/2 nebo vypnout (4-buňkový test na kickoffu).

**H10 ML taker overlay (`strategies/exp_ml_taker.py`, ledger `r1-ml-taker`):** ridge model 1 s (fit jen na prvních
75 % round1), IOC na touch při |predikce| > práh, odvíjení pasivně makerem; parita 425 708 ✔. 12 buněk: nejlepší
práh 1,2 / 20 lotů / max |I| 60 → 385 612 (**−9 %**), práh 0,6 → 379 k, 50 lotů → 268–292 k (−32 až −37 %), monotónně
s intenzitou. **REJECT**; „spread jen jednou“ v simulátoru neobstojí (crossing + pozice zabraná takerem stojí víc,
než horní 1 % predikcí vrátí). GPU/HGB verze se nestaví.

## 2026-11-07 13:55 — H11 standing ladder REJECT, detail H10 ml_taker (home, mozek, R1) → R1 uzavřeno s `mm_park`

**H11 (`strategies/exp_ladder.py`, ledger `r1-ladder`, báze mm_park):** pevné malé příkazy na touch−1..−L, které se
nehýbou (čekají, až k nim cena přijde → čelo fronty). Parita 452 651 ✔, check PASS. S `ladder_reserve` 1 (touch kotace
má přednost) jsou příčky jen ze zbytků kapacity: depth 2 → 453 094–453 170 (+0,1 %, compare 5/8 chunků, sample_v2
−673, 2/8) → REJECT (šum); depth 4 → −1 až −2 %. S `ladder_reserve` 0 (příčky napřed) propad až na 169 693
(−63 %): kapacita v hloubce se skoro neplní a chybí na touch. Netestováno: příčky místo parkovaných stubů.
**H10 detail:** `--max-rows` v research.signals je jen krok, ne useknutí → agent fitoval vlastním skriptem (čas ≤ 75 %,
purge 1 s). Ridge 1 s: walk-forward IC 0,397 (7/7), horní 1 % predikcí +0,81/−0,89 ticku v holdoutu; 100 ms model
nikdy nepřekročí 0,6 ticku. Plná kotace nenechává IOC žádnou kapacitu → agent přidal `taker_reserve` (rezerva lotů
na stranu): samotná rezerva 20 lotů stojí 39 k, 50 lotů 129 k. Overlay nad rezervou ještě −1,4 až −18 k; s rezervou 0
(IOC jen do volné kapacity) −4,5 k (práh 1,2) až −17 k (práh 0,6) při 15–143 k přešlých lotech ≈ −0,1 až −0,3 ticku
na lot. Compare 1,2/20/60: round1 −40 096 (0/8), sample_v2 −29 375 (0/8). Overhead 18,7 µs/událost (3,3 min / 8,8 M).
**Závěr R1:** finál `submission/best_strategy.py` = mm_park 20/1/3 (452 651). Celkem na R1 ~180 běhů, 11 mechanik
+ 5 os; jediná vítězná mechanika nad plnou kapacitou je parkování stubů (Tomáš). Kapacita makera na touch je
cennější než jakýkoli nalezený taker edge; cokoli, co bere kapacitu z touch nebo ruší příkazy, prodělává.

## 2026-11-07 14:15 — H12 stub při ústupu / stub dovnitř při spreadu 2: REJECT (home, mozek, R1)

**Hypotéza:** mm_park ruší příkazy už jen při ústupu trhu (naše kotace zůstane sama o tick uvnitř). (a) `retreat_park`:
místo cancelu zmenšit přední příkaz přes modify na 10–40 lotů (drží místo; při návratu hladiny jsme první).
(b) `improve_stub`: při spreadu ≥ 2 poslat stub o tick dovnitř na straně snižující |I| (být před improverem).
**Změna:** `strategies/exp_retreat_park.py` = mm_park + flagy (0 = off), parita 452 651 ✔, check PASS.
**Data:** round1 + sample_v2 (ledger `r1-retreat-park`).
**Výsledek:** (a) monotónně horší s velikostí stubu: round1 10 → 448 017 (−1 %), 20 → 444 958, 40 → 441 313; sample_v2
10 → 249 370, 40 → 243 739; retreat_depth 1 vs 2 bez rozdílu → stub uvnitř je při pokračujícím pohybu sebrán, stejně
jako hold_inside. (b) round1 10 → 453 718 (+1 067, 7/8 chunků, holdout −78), 20 → 454 988 (+2 337, **8/8 chunků**,
holdout −370); sample_v2 10 → −8 050 (0/8), 20 → −16 036 (0/8); fills +20 %. Compare: REJECT (holdout horší, v2 ztráta
výrazně větší než zisk na round1).
**Závěr:** poslední „nerušit“ místo v mm_park nemá zisk; stub dovnitř je mírně kladný jen v režimu round1 a v režimu
v2 je sebrán trendem → nerobustní, nepovyšovat. Pro R2 si pamatovat jako případnou drobnou páku v R1-like režimu.

## 2026-11-07 14:20 — Stres testy finálu `mm_park` (home, mozek, R1)

**Zrcadlená páska** (strany B↔A, cena 20000 − p; `data/round1_mirror.csv`, ledger `r1-mirror-stress`): **452 651 = přesně
originál** → strategie je stranově symetrická, žádný bias. **Umělý trend** (lineární posun všech inzertů o +40 resp.
+120 ticků přes celé kolo; ledger `r1-drift-stress`): oba kladné, min 448 795, součet 899 332 (≈ −1 % vs 452 651),
breach 0. Pomalý trend (0,01 ticku/s vs 1 s volatilita 0,92) maker neohrozí; rychlejší režim takto syntetizovat nejde.
**Závěr:** riziko záporného PnL na privátní pásce stejného trhu je malé; finál `submission/best_strategy.py` beze změny.

## 2026-11-07 14:25 — H13 `move_min`: sledovat zlepšení ceny jen při „substanci“ nové hladiny: REJECT (home, mozek, R1)

**Hypotéza:** malý improver (medián 14 lotů, život 0,34 s) brzy zmizí; přesun bulku za něj a následný návrat na konec
staré fronty stojí pozici → přesunout se jen když má nová hladina ≥ N tržních lotů.
**Změna:** `strategies/exp_move_min.py` = mm_park + `move_min_qty` (0 = off), parita 452 651 ✔, check PASS.
**Výsledek** (ledger `r1-move-min`, round1 + sample_v2): monotónně horší s prahem: round1 4 → 447 663 (−1,1 %), 8 → 445 266,
15 → 438 371, 30 → 429 511, 250 → 341 317 (−25 %); sample_v2 ≈ 0 do 30 lotů, pak záporné. Compare {8}: round1 −7 385
(0/8), sample_v2 +8 (šum) → REJECT.
**Závěr:** sledovat i drobné improvery se vyplácí — nová hladina je tam, kde jsou filly, a stub už staré místo drží.
Tím je rodina „pořadí ve frontě“ kolem mm_park prozkoumaná: 13 mechanik, jediný zisk je parkování (Tomáš).

## 2026-11-07 14:45 — Fill attribution `mm_park` (home, mozek): kde vzniká +6,3 % a proč na v2 stojí 4,7 %

Report `research/reports/round1_mm_park_attribution.md` (`research/fill_attribution.py`, nové módy `park`, `compare`;
logy `logs/r1-park-inspect`, `logs/v2-park-inspect`, `logs/v2-fullcap-inspect`). Proxy $ = loty × markout 1 s (round1
+21,1 k vs skutečných +26,9 k; v2 −11,7 k vs −12,4 k).
- **Stuby:** round1 19 922 vzniklých, ~40 % se naplní (průměr 8 z 20 lotů), 59 % zrušeno nenaplněných; život do fillu
  1,1 s. **Markout stubu 0,39 $/lot vs 0,18 u touch fillů** (v2: 0,78 vs 0,50) → stuby nejsou adverzní, jsou to naše
  nejlepší filly; 60 % lotů stubů se na round1 plní s 0 loty před nami (v2 jen 28 %). Stuby +63 k na 160 k lotech (7 %
  lotů, 14 % $).
- **Rozklad round1 +26,9 k:** touch filly stejný počet, ale **−472 k lotů (−18 %)** (stuby vážou kapacitu) = −77 k;
  lepší kvalita touch fillů (0,179 vs 0,163) +34 k; stuby +63 k; zbytek chyba proxy.
- **Čas na stropu |I| = 100 klesl z 50 % na 26 %** (v2 53 % → 22 %): stuby na odvíjející straně jsou na čele → rychlejší
  obrat. Filly proti imbalance z −39,7 k na +6,6 k.
- **v2 náklad:** ztracený bulk 68 k lotů (−13 %) za 0,49 $/lot ≈ −33 k; stuby jen +18 k, kvalita +3 k → −11,7 k. Cena
  parkování = vytlačený bulk, a ten je na v2 3× cennější než na round1.
- **Události:** zlepšení ~0,65/s na stranu (round1), ~0,4 (v2); ústupy podobně; **37 % zlepšení se do 500 ms vrátí**
  (flicker) na obou datasetech — ale sledovat je se přesto vyplácí (H13, min_requote).
**Závěr:** mechanismus parkování je potvrzený a správně naladěný pro round1 (víc stubů nad max 5 nepomáhá, stuby
hloubky 2 umírají). Páka pro R2 = režimově závislé parkování (v mean-reverting režimu s cenným bulkem méně/menší
stuby nebo vypnout): rozhodnout 4-buňkovým testem na kickoffu. Pro R1 žádná nová páka.
## 2026-11-07 15:10 — Overfit check, back-pull, ladder, ML signal (radys, our model)

- **Blocked k-fold of mm_park** (`research/diag/kfold.py`, 12 full runs, 9 time blocks): params chosen on 8 blocks,
  scored on the 9th. Out-of-fold 452,550 vs champion 425,708 (+6.3 %), in-sample best 453,321, wins 8/9 blocks.
  Not overfit on round1. Plateau: park 10-30 at depth 1 = 444-453k; park 50 / depth 2 drops.
- **Back-pull** (`exp_backpull.py`): all 9 settings below off on the 25 % screen. Rejected.
- **Standing ladder** (`exp_ladder.py`): best 452,244 = off; ladder first -15..-60 %. Rejected.
- **Ridge signal** (100 ms): walk-forward IC +0.38 in 7/7 folds, stable coefficients. Embedded in `exp_ml_shrink.py`
  (shrinks the predicted-adverse side via modify, never cancels the oldest order). Screen running.

## 2026-11-07 14:35 — R1-only ladění: park_max 5 + improve_stub 20 → 457 288 (+1,0 %), větší stuby dovnitř horší (home, mozek)

**Compare mm_park vs `exp_retreat_park` {"park_max": 5, "improve_stub": 20}:** round1 452 651 → **457 288 (+4 637, +1,0 %)**,
holdout +786, **6/8 párových chunků** → ACCEPT nástrojem; sample_v2 251 236 → 230 674 (−8,2 %, 0/8, stále kladné).
Čistě režimové ladění pro round1 (spread 1, rychlé doplnění spreadu 2 do ~50 ms). Větší stub dovnitř (ledger
`r1-improve-stub-big`, park_max 5): 20 → 457 288, 40 → 454 569, 80 → 436 261, 120 → 424 635 → stub dovnitř je věc
20 lotů, víc je sebráno pohybem skrz.
**Závěr:** k dispozici jako „R1-tuned“ druhý upload (+1 % veřejně; privátní data R1 = stejný trh), za cenu robustnosti
v režimu v2. Výchozí robustní volba zůstává mm_park 20/1/3. Rozhodnutí Honza.

## 2026-11-07 14:40 — H14 `early_join` REJECT: rezerva pro rychlý vstup na svěží hladinu se nevyplatí (home, mozek, R1)

**Hypotéza:** po zlepšení ceny trhem náš bulk přistává na nové hladině až po potvrzení modify (~10 ms), ne za 5 ms;
držet R lotů stále volných a poslat je na novou hladinu hned ve stejném callbacku.
**Změna:** `strategies/exp_early_join.py` = mm_park + `early_join` (0 = off), parita 452 651 ✔, check PASS; mechanismus
ověřen (round1: 12 180 rychlých vstupů, 60 900 lotů, vždy plná rezerva).
**Výsledek** (ledger `r1-early-join`, round1 + sample_v2): R = 5 → 442 897 (−2,2 %), 10 → 431 529, 20 → 405 844, 40 →
336 661 (−26 %); sample_v2 stejně monotónně (5 → −3,2 %). Compare {5}: 0/8 chunků na obou → REJECT. Fills +26 %, ale
horší kvality.
**Závěr:** trvale nevyužitá kapacita stojí víc než pořadí získané o 5 ms dřív; cizí příkazy v tom okně nerozhodují
o kvalitě fillů. Potvrzuje: každý lot kapacity musí ležet na touch.
## 2026-11-07 15:40 — ML shrink on mm_park (radys)

- **What:** the ridge 100 ms mid-move prediction (walk-forward IC +0.38) runs inside the strategy. When |pred| > 0.15 ticks
  against a side, that side's touch quote is shrunk via `modify_order` to a 5-lot stub (keeps queue priority; never cancels).
- **Why:** the main loss is adverse selection on the side about to be run over; shrinking keeps the queue spot but cuts the lots
  that get picked off.
- **Results (round1, full tape, 8 chunks + holdout):** thr 0.15 / stub 5 = 490,460 (8/8 chunks, holdout 126,291, pen 0);
  stub 10 = 486,594; thr 0.1 = 485-488k; thr 0.3 = 454k (~ off); thr 0.05 = 388-401k (too aggressive). Plateau 0.1-0.15.
- **Did not help:** thin-level cap (150/300 lots), flow pull, park 30 vs 20 (72-combo screen, 25 % then full).
- **Next:** server upload to confirm; runtime check on idle CPU (ML adds per-event cost); Honza iterates further.

## 2026-11-07 14:45 — Ověření Tomášova `mm_ml` (home, mozek, R1)

`research.compare` mm_park vs mm_ml (main 1769eaa = submission): round1 452 651 → **490 460 (+37 809, +8,4 %)**, holdout
117 886 → 126 291 (+8 405), **8/8 párových chunků** (+3 265 až +4 448, rovnoměrně), fills 41 934 → 46 527; sample_v2
251 236 → 254 061 (+2 825, holdout +1 835, 5/8) s váhami z round1 → signál pomáhá i cross-market. `research.check`
na round1 (pod zátěží 20 procesů): PASS, 25 µs/událost nad prázdnou strategií, 131 s na 4,4 M, projekce 4,4 min na
8,8 M (notebook bez zátěže 7,0 min — hranice našeho cíle, sandbox 15 min).
**Výhrady (notebook):** ridge byl fitnut `--export-linear` na celém round1 včetně holdoutu → holdout zisk je zčásti
in-sample; řeší agent (čistý fit na prvních 75 % času, `strategies/exp_ml_clean.py`) spolu s `ml_every` kvůli runtime.
**Sebekritika mozku:** drift 0,33 ticku po fillu byl největší číslo v attribution; odepsal jsem ho kvůli ztrátám
cancel-filtrů („shrink ≈ cancel“). Shrink přes modify drží čelo fronty → +8,4 %. Pravidlo: každé „kdy nekotovat“
implementovat jako zmenšení, nikdy jako cancel.

## 2026-11-07 14:50 — Jemný grid `mm_ml`: ml_thr × ml_stub na round1 + sample_v2 (home, mozek)

Ledger `r1-mlml-grid` (15 buněk × 2 datasety, váhy ridge z celého round1 = stejné jako v mm_ml, breach 0 všude, ~130 s/běh):
round1: **thr 0,12 / stub 3 → 497 292** (+6 832, holdout 127 553), 0,12/5 → 496 001, 0,12/8 → 494 597, 0,15/3 → 492 647
(holdout 126 652), **0,15/5 → 490 460 (baked)**, 0,10 → 487–488 k, 0,20 → 462–464 k, 0,08 → 458–461 k.
sample_v2: 0,15–0,20 → 254–255 k, 0,12 → 244–246 k (−10 k), 0,10 → 232 k, 0,08 → 217–222 k. Velikost stubu 3/5/8 je
druhořadá (±2 k).
**Závěr:** plató thr 0,12–0,15; vrchol je na práh citlivý (±5 % mezi 0,10 a 0,20) a práh závisí na škále predikce
→ **na R2 se musí po refitu modelu práh znovu naladit**. Round1-optimum 0,12/3 (+1,4 %) je režimové ladění (na v2 −4 %);
robustní volba 0,15 se stubem 3 (round1 +2,2 k, v2 ±0). Baked 0,15/5 zatím neměnit, dokud nedojde čistý (neleakující)
model; pak sestavit „R1 final“ stack (thr 0,12/3 + park_max 5 + improve_stub 20) a zvážit jako druhý upload.

## 2026-11-07 14:55 — H15 asymetrické prahy ML shrinku: REJECT (home, mozek, R1)

**Hypotéza:** zmenšovat dychtivěji stranu zvyšující |I| (filly ~0 $/lot) a líně stranu snižující |I| (filly +0,21).
**Změna:** `strategies/exp_ml_asym.py` = mm_ml + `ml_thr_add` / `ml_thr_red` (0 = ml_thr), parita 490 460 ✔, check PASS.
**Výsledek** (ledger `r1-ml-asym`, 16 buněk, round1): symetrický 0,15 je optimum. Líná odvíjející strana stojí hodně:
red 0,25/0,4/nikdy → 472 k (−19 k); dychtivá přidávající strana add 0,10/0,08 → 487/483 k. Jediný „zisk“ 0,15/0,10 →
491 898 (+1,4 k, 6/8, holdout −370; sample_v2 −19 k, 0/8) → REJECT.
**Závěr:** shrink odvíjející strany je cenný (adverzní fill tam nás nechá v horší pozici); fill attribution se do
asymetrie prahů nepřekládá. mm_ml beze změny.

## 2026-11-07 15:05 — improve_stub + park_max 5 na mm_ml bázi: REJECT (home, mozek, R1)

`strategies/exp_ml_istub.py` = mm_ml + port `improve_stub` (stub o tick dovnitř při spreadu ≥ 2 na odvíjející straně),
parita 490 460 ✔, check PASS. Ledger `r1-ml-istub`: improve_stub 20 → round1 478 707 (−2,4 %), sample_v2 217 107 (−15 %);
s park_max 5 → 480 818 / 216 248; park_max 5 samo → round1 492 490 (+2 030, holdout −972), sample_v2 252 323 (−1 738) = šum.
Compare {20, 5}: round1 −9 642 (1/8), sample_v2 −37 813 (0/8) → REJECT.
**Závěr:** co stub dovnitř na mm_park přidával (+0,5 %), ML shrink už bere lépe; „R1 stack“ (thr 0,12/3 + park_max 5 +
improve_stub) tedy nemá smysl — zbývá jen thr 0,12/3 (+1,4 %, režimové). Další zisk musí přijít z kvality signálu.

## 2026-11-07 15:05 — Čistý (neleakující) model potvrzuje mm_ml; runtime; horizont 1 s REJECT (home, mozek, R1)

**Leak:** ridge v mm_ml byl fitnut na celém round1. Čistý fit (čas ≤ 75 %, purge 1 s, stejný formát) → `strategies/exp_ml_clean.py`
(`research/reports/round1_ml_clean.md`): IC train 0,383 / holdout 0,375 (žádný přefit). round1 490 905 (leaky 490 460),
holdout +8 242 nad mm_park (leaky +8 405), **8/8 párových chunků**; sample_v2 253 400 (+2 164, 4/8) → **ACCEPT, zisk ML je
skutečný out-of-sample**. Pro finální soubor je fit na všech veřejných datech správný (leak se týkal jen našeho odhadu).
**Runtime** (`ml_every`, `ml_on_touch_change`, check --rows 0 pod zátěží): every 1 → 24,4 µs/ev, 4,2 min / 8,8 M, 490 905;
touch_change 1 → 18,4 µs, 3,3 min, 490 809 (−96); every 2 → 3,1 min, −0,3 %; every 4 → 2,5 min, −0,7 %. Doporučení:
`ml_on_touch_change` 1 bez ztráty; `ml_every` 2 jen při nedostatku rezervy.
**H16 horizont 1 s** (`strategies/exp_ml_h1000.py`, ledger `r1-ml-h1000`, model 1 s fit na 75 %, prahy sladěné na stejnou
četnost): 0,39 → 490 440, 0,49 → 484 026, 0,6 → 467 615; compare {1000, 0,49}: round1 −6 434 (1/8), v2 −1 198 → REJECT.
Pro shrink rozhoduje načasování 100 ms signálu, ne velikost 1 s pohybu.

## 2026-11-07 15:05 — MLP místo ridge pro ML shrink: vyšší IC, nižší PnL → REJECT (home, mozek, R1)

`strategies/exp_ml_mlp.py` (`ml_model` ridge/mlp, parita 490 460 ✔, check PASS, 34,6 µs/ev, 5,7 min / 8,8 M),
report `research/reports/round1_mlp_signal.md`. MLP 23-16-1 na GPU (train 55 % / valid 20 % / test 25 % času): **test IC
0,447 vs ridge 0,375**, horní 1 % edge +0,50/−0,49 ticku (ridge +0,36/−0,41). Všechny architektury 16–32-32 do ±0,005 IC.
Ale ve strategii: thr 0,10 → 469 064, 0,125 → 466 972, 0,15 → 464 022, 0,25 → 458 567 (ridge 0,15: 490 460); compare
{mlp, 0,10}: round1 −21 396 (0/8), sample_v2 −55 531 (0/8) → REJECT.
**Závěr:** IC navíc sedí v „těle“ rozdělení; shrink odměňuje vzácné, jisté predikce přejetí, které lineární kombinace
toku a imbalance zachytí stejně dobře. Přenos na v2 slabý (trénink jen round1). Případný další pokus: trénink na
round1 + sample_v2, nebo cíl zarovnaný se ztrátou (adverzní pohyb podmíněný fillem) místo MSE na 100 ms pohybu.

## 2026-11-07 15:10 — Features v2 pro ridge: 12 nových kandidátů skoro nic, krychle imbalance +0,03 IC (home, mozek)

`research/features_v2.py` (12 kandidátů z npz: vyprazdňování touch 100/500 ms, cancel/insert čistý tok 200 ms, počet
příkazů, avgsz_imb, L2 imbalance, počet změn midu za 1 s, poslední IOC, podíl posledního obchodu, trvání spreadu,
tok/hloubka), report `research/reports/round1_features_v2.md`. Walk-forward ridge 100 ms round1: baseline 0,380; +všech 12
→ 0,386 (nic); na sample_v2 pomáhají ins/cxl/dep (+0,03–0,06), cross-market se nepřenáší. **Nelinearita je větší páka:**
HGB 0,452; **krychle imb1, imb2, micro, cnt_imb1 → ridge 0,410 (+0,03)**, top-1 % edge 0,449 → 0,543 ticku, cross-market IC
0,278 → 0,293 (robustní). Cena ~0,3 µs. Běží test v strategii (`exp_ml_cubes.py`: refit s krychlemi, práh na stejnou
četnost, compare vs mm_ml); po lekci s MLP (IC 0,447, ale PnL −4 %) rozhodne jen research.run.

## 2026-11-07 15:15 — Nástroj `research.refit_ml` pro refit modelu na nové kolo (home, mozek)

`uv run python -X utf8 -m research.refit_ml data/<kolo>.csv --template strategies/mm_ml.py --out strategies/mm_ml_<kolo>.py`
(docs/refit_ml.md): postaví features, fitne ridge přesně jako `--export-linear` (alpha 10, stride, 6 cifer), nahradí v šabloně
jen LINEAR_MODEL a číselný `ml_thr` (= 0,81-kvantil |pred| na nové pásce → stejná četnost shrinku jako 0,15 na round1),
vypíše IC/std/kvantily a pustí check. ~10 s. **Validace:** refit na round1 je byte-identický s mm_ml.py, 490 460 ✔.
**Generálka R2 (`strategies/mm_ml_v2fit.py`, fit na sample_v2):** škála predikce 0,064 vs 0,11 na round1 → auto práh 0,086;
sample_v2 250 588 (champion s round1 modelem 254 061, −1,4 %, bez follow-up gridu); **model z v2 na round1: 294 677 (−40 %)**
→ model se mezi trhy přenáší špatně, na R2 je refit na skutečné pásce kola nutný a práh musí jít z kvantilu.

## 2026-11-07 15:20 — H17 odstupňovaný shrink: malý zisk na round1 (+0,8 %, 7/8), na v2 mírně záporný (home, mozek)

`strategies/exp_ml_graded.py` = mm_ml + `ml_thr_lo` / `ml_frac_lo` (částečný cap mezi dvěma prahy), parita 490 460 ✔, check PASS.
Ledger `r1-ml-graded`: round1 nejlépe 0,10/0,25 → 496 881, 0,12/0,25 → 496 217 (holdout 128 408), 0,10/0,5 → 494 534;
sample_v2 jen 0,12/0,5 a 0,10/0,5 drží ±0 (253 143 / 252 731), agresivnější frakce 0,25 ztrácí až −5 %.
Compare {0,10, 0,5}: round1 **+4 074 (7/8), holdout +1 305**; sample_v2 −1 330 (1/8, holdout +168) → nástroj ACCEPT, ale
datasety nesouhlasí ve znaménku. **Závěr:** reprodukuje trade-off prahu 0,10–0,12, ne nový edge; kandidát jen do R1
stacku s krychlemi, pokud krychle vyjdou (jeden společný compare, jeden upload).

## 2026-11-07 15:25 — Krychle imbalance v ridge: IC +0,03 drží out-of-sample, PnL na round1 ne → REJECT (home, mozek)

`strategies/exp_ml_cubes.py` (`ml_model` base/cubes/cubes_avg, parita 490 460 ✔, check PASS, +24 µs/ev, 4,1 min / 8,8 M),
report `research/reports/round1_ml_cubes.md`. IC hold 0,375 → 0,406, top-1 % edge 0,384 → 0,476 ticku (drží v holdoutu).
Sweep (ledger `r1-ml-cubes`, prahy 0,13/0,16/0,19 sladěné na četnost): round1 486–488 k (−2 až −12 k), sample_v2 253–258 k
(+3,7 k při 0,16). Compare {cubes, 0,16}: round1 −4 296 (2/8), holdout −1 872; sample_v2 +3 700 (7/8) → REJECT.
Poctivý holdout (váhy jen z 75 %) 124 496 vs mm_ml 126 128 → leak to nevysvětluje.
**Závěr (stejná lekce jako MLP):** na round1 shrink už jistý „ocas“ bere; ostřejší signál přidá filly, ne PnL. Krychle
pomáhají v režimu v2 → nechat jako volbu `ml_model` pro R2, pokud data card připomene v2. **ML shrink na round1 je
nasycen na ~490–497 k; další zisk v této rodině na R1 nečekat.**

## 2026-11-07 15:45 — Předvídatelnost sweepů z pásky: žádná (home, mozek, R1 + v2)

`research/sweep_predict.py`, report `research/reports/round1_sweeps.md`. Velké IOC (≥ 300 lotů: 12 067; ≥ 1 000: 2 095) jsou
**Poissonovské**: CV mezer 0,99–1,02, autokorelace velikostí ±0,01, žádné shlukování na stejné straně (lift 1,0), velikosti
se neopakují, žádný vzor „limit → sweep“ (id jsou striktně +1). Prekurzory (cancely na touch, inzerty, sondovací IOC,
imbalance, vyprazdňování, čas od změny midu) na 50/100/500 ms: lift 0,9–1,1, nejlepší AUC 0,506; walk-forward logistický
skór AUC 0,50. Ridge trigger (|pred| > 0,15) označí 9,7 % sweepů při 9,6 % času → předpovídá drift běžného toku, ne sweepy.
Přidání skóru k 23 features: IC 0,3802 → 0,3812. sample_v2 totéž.
**Závěr:** sweepy nelze předvídat 5 ms dopředu; obrana je jen velikost a pozice ve frontě (mm_park/mm_ml). Hypotéza
„otisky účastníků“ zamítnuta.
## 2026-11-07 16:30 — Ideas on top of mm_ml, all rejected so far (radys)

All on round1, full tape unless noted, vs mm_ml 490,460 (`strategies/exp_overcap.py`, every flag off = mm_ml).
- **Over-cap** (quote 150-300 lots, IOC-cut anything above the 100 band): -133k..-316k on a 25 % slice vs +113k.
  Breaches cost the spread on the cut plus 10 $/ms; far more than the extra fills earn.
- **Holding 900 lots for 1-2 s** (Tomas' question): 2 s round trip averages -1.07 ticks/lot, best case on the whole
  tape +5 ticks; 20k of penalty needs a 22-tick move. Dead.
- **Inventory lockout** (stub the growing side from |I| >= 60): 475k. **Inventory-shifted ML threshold**: 477-479k.
- **Post-fill freeze** (stub a side 20/100 ms after its fill): dropped on the 25 % screen.
- **ML stub only at 1-tick spread**: 492,254 (+1.8k, noise).
- **Gap jump** (post inside a 2-tick spread, side by ML pred): 405k (always) .. 492,783 (thr 0.4). The more it jumps the
  worse: orders inside a fresh gap get hit by the sweep's continuation.
- **Why it matters:** the price range is only 105 ticks in 3.3 h; profit is purely spread capture under a 100-lot budget.
  Next: queue-aware stub (`ahead_stub`), park re-tune on the ML base; heavy levers for Honza in radys-findings s.8.

## 2026-11-07 15:50 — H18 cíl zarovnaný se ztrátou: varianta B = první robustní zlepšení nad mm_ml (home, mozek)

`research/fit_target.py`, `strategies/exp_ml_target.py` (`ml_target` base/A/B/C, parita 490 460 ✔, check PASS), report
`research/reports/round1_ml_target.md`. Stejných 23 features, jiný cíl: A = MSE vážené zásahem takera do 100 ms, **B = pohyb
midu po prvním zásahu × [zásah]** (0 bez zásahu), C = skór adverzního fillu. IC proti vlastnímu cíli se liší jen o ≤ 0,005;
lineární model nese totéž. Ale **B při prahu 0,122 (stejná četnost): round1 494 990 (+4 530, 6/8, holdout +1 528),
sample_v2 258 239 (+4 178, 7/8, holdout +764) → ACCEPT na obou datasetech**, bez runtime nákladu (jen jiný dict).
A: +2,2 k / +2,1 k; C: −8 k. Nižší práh B 0,097 → round1 499 397 (v2 256 572).
**Závěr:** malé (+0,9 % / +1,6 %), ale stejné znaménko na obou trzích → kandidát na novou bázi (R2) a na R1 upload.
Běží dolaďování B (thr 0,09–0,105 × stub 3/5) proti mm_ml_r1tuned (497 292).

## 2026-11-07 15:55 — `mm_ml_b`: cíl B + práh 0,105 / stub 3 → round1 503 148, ACCEPT na obou datasetech (home, mozek)

Dolaďování B (ledger `r1-ml-targetB-tune`, `-tune2`, round1 / sample_v2): thr 0,09 → 494,8 k / 255,2 k; 0,097 → 500,9 k / 256,5 k;
**0,105 → 503 148 / 258 042**; 0,11 → 501,7 k / 259,6 k; 0,115 → 499,1 k / 259,1 k; 0,122 → 495,0 k / 258,2 k → plató
0,097–0,115, střed 0,105; stub 3 ≈ 5. **Compare mm_ml vs {B, 0,105, 3}: round1 +12 688 (+2,6 %), holdout +3 397,
7/8 párových chunků; sample_v2 +3 981, holdout +1 686, 6/8 → ACCEPT.** Zapečeno do `strategies/mm_ml_b.py`
(= exp_ml_target s B/0,105/3 jako výchozí), check s runtime běží. Doporučení: nahrát jako final R1 místo mm_ml_r1tuned
(497 292, na v2 −4 %) — víc PnL i robustnější.

## 2026-11-07 15:58 — H19 kratší horizont modelu (50 / 20 ms): REJECT (home, mozek)

`strategies/exp_ml_h50.py`, `exp_ml_h20.py` (přes `refit_ml`, prahy na stejnou četnost), report `round1_ml_short_horizon.md`,
ledger `r1-ml-short`. h50 @0,080: round1 487 319 (−3,1 k, 1/8), sample_v2 244 176 (−9,9 k, 0/8); h20: −3 % / −9 %.
Osa horizontu je plochá až záporná oběma směry (1 s −6 k, 50 ms −3 k, 20 ms −16 k) → 100 ms zůstává.

## 2026-11-07 16:10 — Orákulum: strop rodiny „shrink podle toxicity“ na round1 (home, mozek)

`strategies/diag_oracle_ml.py` (DIAGNOSTIKA, lookahead z pásky, nikdy nesubmitovat), report `research/reports/round1_oracle_ceiling.md`,
ledger `r1-oracle`. Místo ridge skutečný budoucí pohyb midu. Výsledky (round1, mm_park bez ML 452 651, mm_ml 490 460, mm_ml_b 503 148):
- **Dokonalá znalost 100 ms:** shrink na jakýkoli pohyb 539 181; **jen na pohyb ≥ 0,5 ticku 601 310**; stub 0 (úplný pull)
  545 760, stub 20 520 324. → strop 100ms-rodiny ≈ 540–600 k; ridge už bere ~40–50 % z něj.
- **Dokonalá znalost 1 s:** 808 123 (jakýkoli pohyb), 886 570 (≥ 0,5 ticku), stub 0 846 217 — horní mez, 1 s pohyb
  nelze predikovat s potřebným IC (ridge 1 s měl ve strategii méně než 100 ms).
- Selektivita se vyplácí: shrink jen při velkých pohybech > shrink při každém (+62 k na 100 ms, +78 k na 1 s).
**Závěr:** 650 k lídra nejde udělat 100ms toxicity filtrem ani s dokonalou předpovědí → buď předvídá ~1 s pohyb mnohem
lépe než my, nebo má edge mimo filtrování (fronta, velikost, vrstvy), nebo přefituje veřejná data. Pro nás: cílit shrink
na VELKÉ adverzní pohyby (klasifikační cíl), ne zpřesňovat průměrný 100 ms pohyb.

## 2026-11-07 16:10 — Graded shrink nad mm_ml_b: +0,8 % round1 (7/8, holdout +225), v2 −0,45 % → držet, nepovyšovat (home, mozek)

`strategies/exp_b_graded.py` (parita 503 148 ✔, check PASS), ledger `r1-b-graded`: thr_lo 0,07 → round1 506 979 (+3 831), v2 256 884
(−1 158, holdout −1 418); 0,085 → 505 336 / 257 293. Compare {0,07, 0,5}: ACCEPT nástrojem (7/8), ale holdout jen +225 a v2
záporný → marginální; případně jen do finálního R1 stacku, pokud se sejde víc dílků. park_max 5 nad B: REJECT (holdout −301, v2 −1 k).

## 2026-11-07 16:15 — `refit_ml --target B` hotový; model z round1 je lepší i na v2 než model fitnutý na v2 (home, mozek)

`research/refit_ml.py` umí `--target base|B` (+ `--thr-share`, `--ref-data`): B label přes `fit_target.build_targets`, nahradí jen
položku 'B' v LINEAR_MODELS a `ml_thr` = kvantil |pred| na nové pásce při stejném podílu událostí jako mm_ml_b na round1
(27,95 %). Validace: B na round1 → byte-identické s `mm_ml_b.py` (503 148); base → byte-identické s `mm_ml.py`.
**Generálka R2:** B fitnuté na sample_v2 (`strategies/mm_ml_b_v2fit.py`, IC 0,27, práh 0,037): sample_v2 252 091 vs mm_ml_b
(model z round1) **258 042**, round1 207 829. → **Model z round1 je lepší i na druhém trhu než lokální fit na 1,6h pásce.**
Na R2 kickoffu tedy porovnat obojí: mm_ml_b beze změny vs refit na R2 pásce (compare na R2 + round1 + v2), ne refit slepě.

## 2026-11-07 16:20 — H21 klasifikační cíl „velký pohyb“: REJECT (home, mozek)

`research/fit_bigmove.py`, `strategies/exp_ml_bigmove.py` (D1/D4 v LINEAR_MODELS, parita 503 148 ✔), report `round1_ml_bigmove.md`,
ledger `r1-ml-bigmove`. Precision horních 5/10/20 % na pohyb ≥ 0,5 ticku: B 0,415/0,362/0,278, D1 0,418/0,367/0,287, D4
0,422/0,368/0,286 → zlepšení ≤ 0,009 na základu 0,135. Ve strategii při stejné četnosti −2 až −12 k na round1, až −20 k na v2;
compare D4 @0,17: round1 −6 548 (2/8), v2 −20 230 (0/8) → REJECT. 23-feature lineární prostor je na hranici; B vyhrává, protože je
podmíněné skutečným zásahem takera. Lepší výběr velkých pohybů potřebuje stav fronty, ne jiný lineární cíl.

## 2026-11-07 16:25 — H20 fill-EV model s privátními features: model dobrý, jako shrink pravidlo REJECT (home, mozek)

`research/fill_ev.py`, `strategies/exp_fill_ev.py` (ev_thr/ev_mode, parita 490 460 ✔, check PASS), report `round1_fill_ev.md`, ledger
`r1-fill-ev`. Dataset: 63 428 fillů mm_ml na round1 (markout 1 s 0,224 $/lot), 15 679 na v2. Walk-forward ridge na markout 1 s:
pred sám IC 0,28 → tržní features 0,31 → **+ privátní (pozice ve frontě, tloušťka, |I|, snižuje-|I|) 0,36–0,39**, v2 0,13 → 0,62;
HGB nic navíc; spodní 3 decily nesou −101 k $ markoutu. Ale ve strategii: PnL klesá monotónně s prahem, nejlepší buňka
(ev_thr −0,5, or) round1 488 065 (−2,4 k, 2/8), v2 253 189 → REJECT.
**Proč:** zmenšení na stub stojí objem i na lotech, které jsou jen mírně záporné; mm_ml pred už bere hlavní ocas; queue_ahead
v tréninku je podmíněn sweepem (ne live). Konzistentní s orákulem (strop 100ms filtru 540–600 k).
**Nezkoušeno:** trénink na každé události s vahou pravděpodobnosti fillu; velikost shrinku úměrná EV; EV jen pro extrémní ocas.

## 2026-11-07 16:30 — H22 OR-trigger z 1 s modelu nad mm_ml_b: šum (home, mozek) → výzkum R1 uzavřen, finál mm_ml_b

`strategies/exp_b_or1s.py` (thr_1s, or_stub_1s; parita 503 148 ✔, check PASS), ledger `r1-b-or1s`. Nejlepší 0,85 / úplný pull:
round1 +417 (7/8), v2 +43 (4/8) → < 0,1 %, šum; práh 0,6 pomůže round1 (+1,4 k), uškodí v2 (−1,6 k). Velké 1 s pohyby, které ridge
umí, už 100ms pravidlo B pokrývá; limitem je kvalita 1 s predikce. Nepovyšovat.
**Bilance R1 (home):** ~30 mechanik/modelů, ~400 běhů. Nad mm_v2 vyhrály 4 věci: plná kapacita (+33 %), parkování stubů
(Tomáš, +6 %), ML shrink přes modify (Tomáš, +8 %), cíl B (+2,6 %). **Finál R1 = `strategies/mm_ml_b.py` 503 148.** Orákulum:
strop 100ms toxicity filtru 540–600 k; 650 k lídra touto rodinou nejde.
## 2026-11-07 16:55 — Queue-aware stub rejected, handoff (radys)

- `ahead_stub` (shrink the touch order while > X market lots are ahead): X=1200 476,032, 800 462k, 500 440k, 300 402k,
  150 343k vs 492,254 off. Orders deep in the queue advance to the front; shrinking them throws that away.
- Honza's home PC: `mm_ml_r1tuned.py` (thr 0.12 / stub 3) 497,292 is the current best; sweeps are unpredictable.
- Full state and ranked next steps: `docs/radys-findings.md` section 9. Next: fill diagnostic `research/diag/queue_fills2.py`
  on mm_ml_r1tuned, then capacity/inventory-turnover experiments.

## 2026-11-07 16:45 — R1 stack nad mm_ml_b: tight_only + graded + park_max 5 → round1 510 469 (home, mozek)

Na pokyn Honzy („fit tenhle trh“) naskládáno vše, co na round1 vydělává: `strategies/exp_b_stack.py` = mm_ml_b + graded cap
(`ml_thr_lo` 0,07 / `ml_frac_lo` 0,5) + Tomášův `ml_tight_only` (ML stub jen při spreadu 1) + `park_max` 5. Ledger `r1-b-stack`:
tight_only sám +1 742 (7/8, holdout +683; v2 +1 418, holdout +595); tight + graded 507 905 (+4 757, 8/8, holdout +524; v2 +1 087).
**Compare mm_ml_b vs plný stack: round1 503 148 → 510 469 (+7 321, 7/8, holdout +566); sample_v2 258 042 → 258 573 (+531, 5/8,
holdout +504) → ACCEPT.** Zapečeno do `strategies/mm_r1_final.py`; check s runtime běží. Je to režimové ladění pro round1
(privátní R1 = stejný trh); v jiném režimu znovu naladit park/prahy.

## 2026-11-07 16:50 — Otisk generátoru: páska je SYNTETICKÝ queue-reactive simulátor (home, mozek)

`research/generator_fingerprint.py`, report `research/reports/round1_generator.md`. Signatury: inzerty G i IOC přesně Poisson (Fano 1,00),
celková intenzita inzertů 89,9/s na stranu bez ohledu na stav; **reaguje jen umístění ceny**: podíl inzertů na touch roste s velikostí
touch fronty (21,6/s při < 74 lotech → 64/s při > 2 200), 1 tick za touch klesá; životy cancelů i.i.d. naplánované při vložení
(posunutý lognormal, medián 337 ms, max 7,2 s), nezávislé na stavu; **každý limit dostane cancel i když byl vyplněn** (generátor
filly nesleduje); IOC velikost lognormal(4,106; 1,395) nezávislá na frontě; takeři 4,0/s na stranu; vyprázdnění hladiny je mechanické
(IOC ≥ fronta → hladina pryč v 97–99 %). sample_v2 = stejný generátor s jinými parametry (limit uniform 20–40, IOC lognormal(3,71; 1,40)).
Využitelné pravidelnosti: (1) P(touch pryč do 200 ms) = 0,395 při < 35 lotech, 0,19 při 96–262, 0,014 při > 1 991; podle počtu
příkazů 0,455 (1 příkaz) vs 0,021 (≥ 6); při |imbalance| > 0,89 se tenká strana pohne proti v 34 % vs 1,3 %. (2) Filly jsou
aritmetika: P(IOC ≥ q) = 1 − Φ((ln q − 4,106)/1,395), takeři 4/s, cancel hodiny známé → doba vyprázdnění fronty před námi je
spočitatelná. (3) Rozšíření spreadu se zaplní za medián 22 ms (p90 144) inzertem dovnitř na straně s větší frontou → mid pokračuje
v 70 % epizod (proto P(reverze) 0,46).
**Důsledek:** naše mechaniky (plná kapacita, park, shrink podle imbalance) jsou v souladu s generátorem; lineární ridge už
P(vyprázdnění) zhruba nese (krychle IC +0,03 bez PnL). Pro R2: stejná rodina modelu s jinými parametry → strategie se přenáší,
přeladit prahy/park; „aritmetika fillů“ umožňuje spočítat optimální velikost stubů/bulku z parametrů kola (námět).

## 2026-11-07 17:10 — Věk příkazů (naplánované cancely) jako feature: +0,007 IC na round1 → neportovat (home, mozek)

`research/features_age.py` (numba replay pásky, 6 s; touch množství sedí se simulátorem na 100 % řádků), report
`research/reports/round1_age_feature.md`. Očekávaný cancel objem touch do 200 ms z věkového profilu koreluje 0,68 s realizovaným,
ale velikost fronty sama vysvětlí skoro totéž (R² 0,461 vs 0,474 s věky): cancel čas každého příkazu je jeden i.i.d. los a fronta
o desítkách příkazů to zprůměruje. Walk-forward ridge 100 ms round1: 0,3828 → 0,3900 (+0,007), HGB +0,002. Na sample_v2 +0,022
(100 ms) / +0,076 (1 s) a cross-fit round1 → v2 0,283 → 0,337 — věky jsou podíly, přenášejí se líp. **Závěr:** na round1 nic;
levný robustnostní přídavek pro R2, pokud bude páska jako v2. Port popsán v reportu (ins_t[id] z prvního výskytu v snapshotu).

## 2026-11-07 17:20 — Doladění finálního stacku: zapečené hodnoty jsou optimum (home, mozek)

Ledger `r1-final-retune` (mm_r1_final, round1, thr 0,095/0,105/0,115 × stub 3/5 × frac 0,25/0,5/0,75): **0,105 / 3 / 0,5 = 510 469 je
nejlepší buňka**; sousedé 505–509 k, nejhorší 500,9 k → plató ±1 %, finál beze změny. Nahráno a označeno final (server 510 469,
3 min 2,8 s; 54 229 fillů, 1,97 M lotů = 0,259 $/lot, PnL SE 9 391, |I| kvantily 57/80/100/100).

## 2026-11-07 17:25 — H23 ML-načasovaný IOC unwind: REJECT (home, mozek)

`strategies/exp_ml_unwind.py` (IOC jen při |I| ≥ N a predikci proti pozici; parita 510 469 ✔, check PASS), ledger `r1-ml-unwind`,
12 buněk: všechny ≤ baseline na obou datasetech; nejlepší (95/0,2/20) round1 −505 (2/8), v2 −176; při častějším spouštění
(thr 0,105, qty 50) až −10 k / −7 k. Půlspread IOC + footprint > ušetřený drift; graded shrink už případ „predikce proti
inventáři“ pokrývá. Rychlejší obrat inventáře aktivně nejde ani s načasováním modelem.

## 2026-11-07 17:30 — Strukturální prediktor z pravidel generátoru: IC +0,032 na 100 ms (nelinearita, ne nová informace) (home, mozek)

`research/features_struct.py`, report `research/reports/round1_struct_predictor.md`. Uzavřená forma bez parametrů: P_dep(strana) z Poissona
takerů × lognormal velikosti × cancel člen; E[dm] = 0,5·(P_dep_ask − P_dep_bid). Sám o sobě IC 0,351 (92 % fitnutého ridge 0,380);
ridge 23 + E[dm]: **0,412 (+0,032)** na 100 ms, HGB +0,001 (strom to má), 1 s nic. Cross-fit: nefitnutý E[dm] přenáší líp (0,310 vs
ridge 0,278). Kalibrace P_dep na round1 od 10 % výš dobrá; na v2 2–10× nadhodnoceno (jiné mu). Stejná povaha jako krychle (+0,03 IC,
PnL −1 %) → rozhodne jen strategie: běží port `edm_100` jako 24. feature cíle B (`exp_b_struct.py`), compare do 17:45.

<<<<<<< HEAD
## 2026-11-07 17:40 — Finál R1 = `mm_r1_final2` (514 093); strukturální feature v cíli B: REJECT (home, mozek)

**Grid nad finálem** (`r1-final-grid2`, `-grid3`): ML shrink jako úplné stažení (ml_stub 0) + park 15 / max 8 → **514 093**; compare vs
mm_r1_final: round1 +3 624 (6/8, holdout +1 462), sample_v2 +1 553 (6/8, holdout +205) → ACCEPT; check PASS 4,0 min → zapečeno
`strategies/mm_r1_final2.py` (1bc8ff6), k uploadu jako final. Graded knoby (0,05–0,08 × 0,4/0,6) ±1 k = šum.
**Strukturální E[dm] jako 24. feature cíle B** (`strategies/exp_b_struct.py`, parita 510 469 ✔): holdout IC +0,006 vs cíl B
(+0,025 vs prostý pohyb), ale round1 502 337 (−8 132, 0/8), v2 −1 653 → REJECT. Stejná lekce jako MLP/krychle: lepší globální
rank ≠ lepší výběr horních 28 % |pred|, které shrink používá; B je lepší, protože je podmíněné skutečným zásahem takera.
**Konečná bilance R1 (home):** ~40 mechanik/modelů, ~550 běhů; vítězné: plná kapacita (+33 %), park (Tomáš, +6 %), ML shrink
přes modify (Tomáš, +8 %), cíl B (+2,6 %), stack tight/graded/park (+1,4 %), úplné stažení + park 15/8 (+0,7 %). mm_v2 321 k → 514 k.

## 2026-11-07 17:45 — Poslední grid nad final2: +1,5 k při horším holdoutu → neměnit; R1 uzavřeno na 514 093 (home, mozek)

Ledger `r1-final2-retune` (thr 0,095/0,105/0,115 × thr_lo 0,06/0,07/0,08): 0,115/0,06 → 515 565 (+1 472), ale holdout 130 587 vs
131 716; 0,115/0,07 → 515 098 (holdout −705). Zisk ≤ 0,3 % s horším holdoutem = šum/přefit → **finál R1 zůstává `mm_r1_final2`
(514 093, nahráno, server = lokál, final)**. Tomášův agent chystá strukturální prediktor do shrinku — varování s čísly posláno
(na home −8 132, 0/8); do 18:00 žádné další uploady bez compare ≥ 6/8 proti final2.

## 2026-11-07 17:55 — Audit Tomášova `mm_r1_final3` (quote_size 180, park_max 6): REJECT, final zůstává final2 (home, mozek)

Diff proti final2 jen dvě čísla. Compare final2 vs final3: round1 514 093 → 515 687 (+1 594, 7/8), **holdout −166**; sample_v2 +472 (5/8),
holdout +143 → REJECT protokolem (holdout horší); check PASS. Zisk v šumu (~0,01 bodu), přepínání finalu 20 min před koncem bez
přínosu → doporučeno nenahrávat. **Final R1 = `mm_r1_final2` 514 093.**

## 2026-11-07 18:00 — Optuna (TPE) nad final2: nic nad ruční optimum; R1 definitivně 514 093 (home, mozek)

`research/optuna_tune.py` (TPE přes `research.run --set` v paralelních dávkách, cíl = train PnL, holdout mimo hledání; `uv run --with
optuna`). 24 trialů na round1 přes 9 knobů (thr, thr_lo, frac, stub, park_size/max/depth, quote_size, tight_only; ledger `r1-optuna-*`):
nejlepší train 382 319 / hold 130 067 / pnl 512 386 (thr 0,109, thr_lo 0,08, frac 0,58, stub 3, park 16/7, quote 160) — **pod final2
(train 382 377, hold 131 716, pnl 514 093)**. Povrch je plochý; ruční gridy už optimum našly. Žádný nový soubor; nástroj zůstává pro R2
(přeladění po refitu za ~15 min místo hodinového gridu).

## 2026-11-07 18:15 — Privátní výsledky R1: 12. místo z 61, 511 422; pole vyrostlo o ~25 %, my ne (home, mozek)

Privátní leaderboard R1: GammaStrike 792 696, Olexa namichej 701 374, Podbranč 667 357, Southeast Fortress 657 359, Codehorses
655 444, Mac mec mic 615 391, quantový mouchy 603 501, ThatsMyQuant 602 527, Girls JWHF 576 877, Kakaový nápoj 533 533, FIFO 524 193,
**p<0.05 511 422 (12.)**. Průměr 434 481, SD 138 695, medián 447 797 (60 týmů > 0; Cinkačka −52 = 0). Naše z = 0,55 → **0,44 bodu**;
lídr z 2,58 → 2,07; top 5 ≥ 1,27 bodu.
**Klíčové pozorování:** veřejný průměr pole byl 344 k (15:30), privátní 434 k → pole +25 %; GammaStrike 650 → 793 k (+22 %);
**my 514 093 → 511 422 (±0 %)**. Privátní páska je tedy „bohatší“ (víc objemu/volatility, pravděpodobně jiné parametry
generátoru, jako sample_v2 vs round1), a náš edge se na ni nepřenesl, zatímco edge ostatních ano. Hypotéza: vrstvy laděné na
veřejnou pásku (pevný práh shrinku 0,105 na škále predikce, graded 0,07, park 15/8, tight_only) jsou křehké vůči posunu parametrů
(viz citlivost prahu ±5 % a jiné optimum na v2); plain mm_park by při +25 % pole měl ~565 k. Moje hypotéza „650 k = přefit“ byla špatně.
**Další krok:** (1) zjistit privátní PnL našich pěti submisí (mm_park 452 k, mm_ml 490 k, r1tuned 497 k, final 510 k, final2 514 k
veřejně) → která vrstva se přenesla; (2) adaptivní práh shrinku (klouzavý kvantil |pred|, konstantní podíl spouštění) a velikosti
relativní k rozdělení IOC → robustnost vůči parametrům generátoru; (3) na R2 preferovat robustní variantu před veřejně doladěnou.

## 2026-11-07 18:55 — H24 adaptivní práh shrinku (klouzavý kvantil |pred|): funguje mechanicky, zisk žádný (home, mozek)

`strategies/exp_adaptive.py` (ml_adaptive, ml_share 0,28 / ml_share_lo 0,49, histogram 400 binů, O(1); parita 514 093 ✔, check PASS),
ledger `r1-adaptive`. Prahy sledují škálu predikce (round1 0,105/0,069 = baked; v2 0,0915/0,0495). round1 514 160 (±0), sample_v2
258 293 (−1,8 k) při stejném podílu; podíl 0,22 → +1,0 k / +0,8 k (šum). Compare {0,28} REJECT (3/8). **Závěr:** pojistka proti posunu
škály predikce na novém kole (pro R2 užitečné, pokud nebudeme refitovat), ne zdroj PnL. Privátní R1 navíc ukázal, že se final2 přenesl 1:1,
takže křehkost prahů nebyla příčinou rozdílu k lídrům.

## 2026-11-07 19:00 — H25 hluboký ladder na prázdných hladinách: REJECT (home, mozek)

`strategies/exp_ladder_deep.py` (ladder_start 4/6, depth 3/5, size 20/50, reserve 0; parita 452 651 ✔, check PASS), ledger `r1-ladder-deep`:
všechny buňky 10–81 k (−82 až −98 %), fills −62 až −96 %. Příčky ve 4–9 ticích hloubky zaberou 100–500 lotů invariantu a skoro se
neplní → touch kotace vyhladoví. Kvalitu hlubokých fillů to neměří, ale platí totéž co u mělkého ladderu: kapacita je vázaná a nečinná
příčka stojí filly na touch. „Být první z konstrukce“ v hloubce nejde zaplatit kapacitou. Hypotéza o konstrukci lídrů tímhle padá.
=======
## 2026-11-07 17:55 — Round 1 closed (radys)

- Final: team 511,422, 12th of 21+; leader 792,696. Our local best stack 503,745 (mm_ml_r1tuned + ml_tight_only + park_max 10 + park_size 15).
- Fresh-size gate rejected (-35k..-212k). Every size reduction lost all round; every size increase won.
- What went wrong: too much time on small filters around one architecture, and declaring ceilings instead of searching
  structure. Round 2 plan: radys-findings section 9c.

## 2026-11-07 18:40 — Capacity value and the round-2 prompt (radys)

- `research/diag/nolimit.py` (penalty off): mm_ml_r1tuned at max_position 100/150/200/300 = 497,292 / 598,403 /
  634,727 / 677,830; ~7,900 breach episodes and 4,600-6,700 s over 100 when allowed. Capacity is worth +100..+180k,
  but breaching to get it would cost millions.
- IOC unwind with `unwind_slack` 40 (the old unwind never fired because the reducing side is full): best 452,492 (-45k).
  Rejected: crossing costs more than the freed room earns.
- Conclusion: passive inventory turnover + higher edge per lot. Workflow for Honza's Claude: `docs/prompt-fable-round2.md`.
>>>>>>> origin/main

## 2026-11-07 19:25 — Orákulum priority ve frontě: strop 1,3–1,4 M na round1 → konstrukce lídrů = priorita (home, mozek)

`research/diag/sim_frontqueue.py` (monkey-patch OrderBook.insert: náš příkaz se zařadí před všechny tržní na své ceně; simulátor nedotčen;
s vypnutým patchem reprodukuje 514 093), report `research/reports/round1_front_oracle.md`. S dokonalou prioritou: mm_r1_final2
**1 395 933** (fills 72 k, 3,84 M lotů, 0,364 $/lot), exp_full_capacity 1 301 334, mm_park 1 301 703, final2 bez ML 1 311 712;
priorita jen na svěžích hladinách (< 100 lotů při vložení) 546 305 (+32 k). Chunky stabilní (177–188 k každý).
**Závěr:** strop priority je 2,7× náš výsledek a 2× lídr → mezera 650–790 k je o pozici ve frontě, ne o signálu (ML filtr pod
prioritou +85 k, normálně +51 k). ~850 k hodnoty je ve skoku *hlubokých* front (návrat na touch po ústupu), ne ve svěžích hladinách.
V FIFO legitimně jen stářím příkazu: nechat kapacitu zrát na hladině místo re-kotace při každém pohybu (2,6×/s). Dřívější „trpělivé“
varianty (requote_tolerance 1 −27 % na mm_v2, move_min −1 %, hold_inside −8 %) byly špatně postavené nebo na jiné bázi → R2 výzkum
= jak legitimně získat stáří/prioritu: velký park (běží `r1-stay-big`), zpožděné následování zlepšení, relokace místo shrinku.

## 2026-11-07 20:05 — Round 2 kickoff: trendový režim, všechny makery záporné; pojistka = flow-pull maker (home, mozek)

**Data card round2** (2,19 M řádků, 6 000 s): spread 1 tick 84 %, 2 ticky 12 %; **P(reverze) 0,38, autokorelace změn midu +0,18, VR 60 s
1,27, 1 s std 3,79 ticku (R1 0,92)**; crash −500 ticků v minutách 60–75, celkem 10 000 → 9 689. mm_edge: čelo +0,07/fill (1 s), konec
fronty −0,57; filtr „kotovat stranu jen když proti ní neteče tok 100 ms“ +0,25 na 70 % fillů. Taker scan nejlépe +19,7 k (slabé).
**Baseliny na round2:** exp_full_capacity −216 739, mm_park −229 893, mm_ml_b −174 560, mm_r1_final2 −170 833 (3/8 chunků); momentum
taker −2,1 až −2,4 M; risk grid makera nejlépe max_position 20 → −11 k; imb_pull −83 k; refit B na round2 −93 k (max_position 20 → +1 k).
**Flow-pull maker** (`exp_flow_pull_home`: nekotovat stranu, proti které teče podepsaný tok posledních N taker příkazů): okno 30, práh 200,
max_position 20 → **round2 +20 151 (8/8, chunk_min +364), zrcadlo +20 151, round1 +48 527 (8/8), sample_v2 +36 073 (8/8)**; plató 150–300 ×
okno 30 × max_pos 20–30 (19–24 k). Zapečeno `strategies/mm_r2_insurance.py`. Běží: trend-gated maker, hybrid maker+taker po toku.

## 2026-11-07 20:20 — R2 anatomy: holes in the book, queue age, where the gate's pnl comes from (home, mozek)

Report `research/reports/round2_anatomy.md`. Round2 = round1 generator with deeper placement and deeper IOC limits:
the book behind the touch has holes, 34 transient jumps (> 20 ticks, snap back in ms) are 63 % of the mid
variance; robust (spread <= 3) 1 s std 3.9 ticks vs 0.92. No consistent predictability at 10-120 s (weak VR
1.2-1.5 at 60-300 s). Queue age (`research/diag/queue_age.py`): an order parked 1 s before its level becomes the
touch has 0 lots ahead 84 % of the time (2 s: 97 %). Front-of-queue oracle: gate maker 84k -> 232k.
`exp_r2_ladder.py` (exp_ladder_home + trend gate with robust drift `gate_max_spread`): parity 84,025 ✔; robust
drift 3 -> 86,333 (8/8, chunk_min +95); parking -20k, ladder 15 +1k. Fill decomposition (`fill_decomp.py`):
spread +0.37/lot at the fill, -0.32/lot adverse within 1 s, +0.10/lot trend over 60 s. Next: cut the first-second
adverse selection, then the trend estimate.

## 2026-11-07 20:30 — R2: architectures other than the gate (home, mozek; Honza: "don't optimise one architecture early")

- **Taker economics** (`research/diag/taker_econ.py`): crossing on flow/return signals is break-even at best; after
  big 1 s moves the price REVERTS (momentum taker -1.5..-4.4 ticks at 1-5 s); 60 s momentum +1.5-2.5 ticks but only
  ~100 independent windows -> a directional book is capped by 100 lots (tens of k). Contrarian after q0.999 spikes
  +3..+12 ticks within 1-5 s, but ~15-60 events -> < 10k. Taking is not the R2 engine.
- **Front-of-queue oracle without the gate:** plain maker -217k -> +170k, R1 champion -171k -> +209k. Queue position is a
  400k swing; the gate is a band-aid for back-of-queue fills.
- **Static grid** (`strategies/exp_grid.py`, never re-prices, orders age to the front): -80k..-250k without gate,
  12-49k with gate. REJECT.
- **Why** (`research/diag/front_fill_age.py`): the front-of-queue fill at a FRESH level (made by an improving insert)
  earns +0.64 ticks/lot at 1 s (< 50 ms old) and +0.1..0.2 later; at a level the touch RETREATED onto it loses
  -0.57/-0.42 in the first 200 ms. Grid/ladder/park orders become the touch by retreat -> adverse.
- **Fresh-level maker** (`strategies/exp_fresh.py`, joins only a touch made by improvement <= fresh_us ago):
  no gate -217k -> -36k (keep_alone 0), with flow_pull 200 -13k -> +40k; with the gate no gain (84k vs 86k).
  Improving ourselves (improve2 / improve_min_spread 3) -300..-600k: our own inside orders get picked off. 

## 2026-11-07 20:50 — R2: hidden-fair filter integrated, mm_r2_fair = 101,933 (home, mozek)

- radys' hidden-fair signal (findings s.11) implemented online in `strategies/exp_r2_fair.py` (new market ids = last
  order of each of the first 3 levels; running side offsets; tau 50 ms). Offline split of our fills by the signal
  (`research/diag/fair_fills.py`): plain maker fills with sig against us -305k at 1 s, with it +106k.
- **Upload candidate `strategies/mm_r2_fair.py`** (gate 30 s / 4 / spread<=3 + fair_thr 0.3): round2 **101,933** (8/8,
  chunk_min +2,067), mirror same, round1 / sample_v2 positive, check PASS (~1.1 min / 8.8M rows). Previous candidate
  `mm_r2_gate.py` 86,333; insurance `mm_r2_insurance.py` 20,151.
- Rejected today (subagents): depletion-EV filter (no gain), trend-layer hysteresis / 2nd window / vol-scaled thr /
  flip IOC (30 s / 4 is a spike; 120 s plateau 84-105k but chunk-unstable), synthetic R2 generator (no holes ->
  inconclusive).
- Running at home now (do not duplicate, radys): T2 fair taker overlay (`exp_r2_fairtake.py`), T4 better fair
  estimator (`exp_r2_fair2.py`), fair x gate robustness map + gate on fair drift (`exp_r2_fairgate.py`), small-order
  grid + fair filter + over-cap beyond the 100 invariant (`exp_r2_grid2.py`, your s.10 idea), ML refit + gate
  (`exp_r2_ml.py`). Free for radys: D1-D7 data questions, anything else.

## 2026-11-07 21:00 — R2 leaderboard: 75th percentile ~400k at 20:31 (median 55k) -> we (102k) miss a big edge (home, mozek)

- Offline pure fair TAKER (`research/diag/fair_taker.py`, non-overlapping, <= 100 lots limited by touch qty, crossing
  both ways): best +85k (tau 50, |sig| > 1.5, hold 300 ms, 7/8 chunks); |sig| > 3: +5/lot but only ~55 lots per trade
  (touch is thin exactly then). Not the 400k by itself.
- Mechanism (radys s.11 + IOC limits): IOC sides are random but their LIMIT prices centre on the hidden fair F, so when
  F > mid only buy IOCs execute -> flow is directional given F. New architecture running: fair-centric quoting
  (`exp_r2_fq.py`: bid F-k / ask F+k, IOC when F is beyond the touch).
- Running at home: exp_r2_fq (fair-centric), exp_r2_grid2 (small orders + fair + over-cap), exp_r2_fairtake (taker
  overlay), exp_r2_fair2 (better F estimator), exp_r2_fairgate (robustness map + gate on F), exp_r2_ml (ML refit).
- radys: please hunt other BIG hypotheses (what gives 400k?): D6 holes (fills far from F during one-sided books),
  IOC partial fills revealing F, anything structural. Note: the penalty is FLAT 10,000/s regardless of the excess
  over 100 (simulator._accrue_breach_penalty), the reject is only potential |pos| > 1000.

## 2026-11-07 21:10 — R2 BREAKTHROUGH: maker + informed taker = 204,054 (home, mozek)

- Online fair signal bug: scanning only 3 levels missed the deep inserts. With fair_levels 20 + mid at any spread
  (`fair_mid_any`) the online signal matches the offline one (corr 0.954; |sig| > 1 4.6 % of the time).
- In R2 the generator's IOC takers are INFORMED (limits centred on hidden F): market makers lose ~-3.9M at 10 s,
  takers win it. So we take too: `strategies/exp_r2_taker.py` = maker (gate + fair filter) + IOC taker on |sig| > thr
  that exits when the signal fades (flattens inventory).
- **Upload candidate `strategies/mm_r2_hybrid.py`**: round2 **204,054** (8/8, chunk_min +6,103, hold 69k), mirror
  same, round1 342k, sample_v2 246k, check PASS 1.4 min. Plateau: take_thr 1.0-1.5 x exit -0.25..0.25 x tau 30-80 x
  gate 6-8 = 180-205k, all 8/8. Pure taker alone 85-89k; maker alone (fair 0.45, gate 6) 119.6k.

## 2026-11-07 21:20 — mm_r2_hybrid2 = 211,186; small-order grid rejected; F estimate is at its information limit (home)

- **Upload candidate `strategies/mm_r2_hybrid2.py`**: round2 **211,186** (8/8, chunk_min +8.6k), mirror same, round1 430k,
  sample_v2 248k, check PASS 1.4 min. = hybrid + gate 20 s / thr 8 + fair-filter hysteresis 0.5 (exp_r2_fair2 idea).
- Rejected: small-order grid + fair + over-cap (`exp_r2_grid2.py`, best 72k: in R2 every extra fill/queue position
  lowers pnl), taker exit of own position only (-14.5k), inv_stop (-10..-40k), smaller quote_size / requote tolerance.
- F estimator study (`research/diag/fair_model.py`, `exp_r2_fair2.py`): bids = floor(F - D), D ~ 1 tick + geometric
  tail 0.25/tick, clamped at the opposite touch -1 (46 % on the clamp); EWMA tau 20-50 ms is at the information limit
  (Kalman/Tobit +0.01 corr, worse online). Hysteresis on the maker filter +7.6k.
- Hybrid internals: |I| >= 90 for 73 % of the time; maker fills 587k lots (+0.12/lot at 1 s, +1.04/lot at 60 s = trend),
  taker 77k lots (+1.26/lot at 1 s). Running: fair-centric quoter, ML shrink stacked on the hybrid, taker sizing/exits.

## 2026-11-07 21:30 — R2: aggressive fair taker = 341,105 -> upload candidate `strategies/mm_r2_fq.py` (home)

- Subagent's fair-centric quoter (`strategies/exp_r2_fq.py`) found that the winner is an AGGRESSIVE IOC taker: cross
  100 lots at the touch whenever F_hat = mid + fair_sig is beyond the opposite touch by 0.25 ticks (|sig| >= 0.75 at a
  1-tick spread), re-enter on every update, plus touch quotes only on the side F favours, 30 s gate thr 6, fair over
  20 levels / any-spread mid, tau 50.
- **`strategies/mm_r2_fq.py`** (baked): round2 **341,105** (8/8, chunk_min +14.6k, hold 107k), mirror same, round1 377k,
  sample_v2 253k, breach 0, check PASS 1.3 min. take_k 0.1 / 0.25 / 0.5 = 298k / 335k / 291k; cooldown 0-30 ms 320-341k;
  gate off 277k; tau 20 185k; fair-centric (k 1) quotes under the same taker 304k.
- Also today: flip taker (hold +-100 until the opposite signal) + maker 229-234k; ML shrink on the hybrid rejected
  (gain only in its fit window); fade-exit taker inside the maker 204-211k (mm_r2_hybrid / hybrid2).

## 2026-11-07 21:45 — mm_r2_fq2 = 349,077 (upload candidate); queue position is the remaining 244k (home)

- **`strategies/mm_r2_fq2.py`**: mm_r2_fq + tau 70 ms, fq_take 200 (one IOC can reverse a full position), take_k 0.2,
  gate_thr 8. round2 **349,077** (8/8, chunk_min +15.8k, hold 102k), mirror same, round1 444k, sample_v2 260k, breach 0,
  check PASS 1.2 min. Plateau tau 50-70 x take_k 0.15-0.25 x gate 6-8 = 335-349k.
- Front-of-queue oracle on mm_r2_fq: 341k -> **585k** (+244k): the remaining gap to the top teams is queue position.
  Realistic attempts so far all lose: sticky quotes (keep while F only mildly against) -6..-56k, inside-spread quotes
  -55..-68k, capacity release at neutral signal -57..-350k, inv_stop (maker only reduces) -1..-23k.
- In mm_r2_fq the inventory sits at +-100 71 % of the time and 98 % of the taker's wanted crosses are room-blocked,
  but releasing room costs more than it earns (holding with F is itself the edge).
- Rejected on the fq base: ML shrink (B75 +25k only in its fit window, holdout -3k, sample_v2 -60k; B1 +5.9k 5/8),
  deeper sweeps, size by signal strength, over-cap (penalty 52-59k), passive exits, skew.

## 2026-11-07 22:05 — Oracle-F: the strategy is worth 935k with a perfect 150 ms predictor; realistic predictors fail (home)

- `research/diag/oracle_fair.py` (diagnostic, look-ahead): mm_r2_fq2 with fair_sig replaced by the real future robust
  mid move: lead 50 / 150 / 500 / 1000 ms = 759k / 935k / 884k / 685k; clipped at +-2 ticks 916k; 50 % blend 915k;
  25 % blend 822k. The value is everyday precision at the taker's decision points, not the big F jumps.
- Realistic predictors: fair taus 20/70/150 + fair-vs-queue interactions raise the offline holdout corr with the
  150 ms move 0.44 -> 0.53 but the strategy drops to 244-273k (`exp_r2_pred2.py`); F velocity (dF over 0.5-2 s) is not
  better than the F level offline. In round2 F - mid predicts the mid over 1 / 5 / 10 s with corr 0.32 / 0.33 / 0.33
  (round1: 0.37 / 0.19 / 0.14): the mid keeps moving toward F for seconds.
- Hidden F jumps exist (e.g. -88 ticks at 3873.18 s: a bid inserted 83 ticks below the book), but jumps >= 20 ticks are
  only ~6-10 episodes; most single-insert "5-tick jumps" are the geometric placement tail.
- PnL curve of mm_r2_fq: no catastrophic windows (worst 10 s -4.7k); seconds with |dmid| > 8 earn +115k.
- Running: ridge predictor with build_matrix features (exp_r2_pred.py), oracle-labelled taker entry filter
  (exp_r2_tkfilt.py).

## 2026-11-07 22:10 — Leaderboard: mm_r2_fq2 = 740,067 (local 349k); mm_r2_pred = 371,961 local -> next upload (home)

- Honza uploaded `mm_r2_fq2.py`: leaderboard **+740,067** (22,521 filled orders, 2.25M lots, end position -100,
  mark-out price 10,482, penalty 0, sim 1 min 5 s). The leaderboard tape is a different, ~2x longer/more active one
  (local round2: 10.4k fills, ends ~9,690) -> local gains scale ~x2.1. 75th percentile ~700k at that time.
- **Next upload `strategies/mm_r2_pred.py`** (subagent): mm_r2_fq2 + ridge predictor of the 150 ms move (fair taus,
  queue/spread/flow features, fit on round2's first 75 %) used ONLY by the IOC taker, blended 50/50 with fair_sig.
  round2 **371,961** (8/8; compare ACCEPT: +22.9k, 6/8 paired, hold +4.3k), mirror 369.7k, round1 441.6k, sample_v2
  265.9k, breach 0, check PASS 1.4 min.
- Centered-F oracle (`research/diag/oracle_centf.py`, F from inserts within +-50/100 ms, no future trades): 586k / 684k
  local -> a lag-free low-noise F estimate is worth +240-335k, but every causal attempt fails: longer tau (90-220 ms)
  208-312k, CUSUM level reset 16-225k, surprise-weighted inserts 215-339k, Tobit for the taker 356.7k (hold +0.6k,
  noise), oracle-labelled taker entry filter (no out-of-sample power at the holding horizon), higher flip threshold.

## 2026-11-07 22:20 — Leaderboard mm_r2_pred = 797,904; next: mm_r2_pred2 (gate 20 s / thr 10) (home)

- Leaderboard: `mm_r2_pred` **797,904** (+57.8k over mm_r2_fq2's 740,067; sim 1 min 30 s). Local -> leaderboard ~x2.1-2.15.
- **`strategies/mm_r2_pred2.py`** = mm_r2_pred + gate window 20 s, gate_thr 10 (robust across regimes): round2 378,507
  (+6.5k), mirror 374,093, round1 480,854 (+39k), sample_v2 267,440 (+1.5k), check PASS 1.4 min. (gate 6 / take_k 0.15
  was +3k more on round2 but -64k on round1: rejected as regime-fragile.)
- Rejected since 22:05: slow F-mid gate (sg), taker-only faster tau, Tobit for the taker (+7.6k, hold +0.6k), over-cap
  for extreme signals only (k 4-9: -7..-53k, penalty 4-17k), direction model at 1-10 s (F-mid corr at 1-10 s is an
  artefact of a few big jumps: by quarter +0.21/+0.13/+0.07/-0.01 at 1 s).

## 2026-11-07 22:30 — R2 frontier check: everything below rejected; mm_r2_pred2 stays the upload (home)

Tested on mm_r2_pred / mm_r2_pred2 / fq2 (round2 local): asymmetric maker threshold for adds (-9..-32k), trend-gated
taker (-20..-120k), over-cap only for extreme signals (-7..-53k), CUSUM level F (16-225k), surprise-weighted inserts,
longer tau (90-220 ms), taker-only faster tau, Tobit for the taker (+7.6k noise), tail-region predictors (pred3: +1.3k,
hold worse), oracle-labelled entry filter, liquidity replacement after sweeps (+2.4k per tape, noise), slow F-mid gate,
cancel flow as an F reading (no incremental power). Maker fills that ADD to |I| are -0.18/lot at 1 s but +0.02 at
10 s and removing them hurts. On the leaderboard our inventory is at +-100 for >= 99.7 % of the time.

## 2026-11-07 22:35 — Leaderboard: mm_r2_pred2 = 787,414 (WORSE than mm_r2_pred 797,904) -> keep mm_r2_pred as final (home)

- `mm_r2_pred2` (gate 20 s / thr 10) on the leaderboard: 787,414 (26,071 filled orders vs 22,519, 2.57M lots vs 2.25M):
  the shorter, higher gate trades more and loses 10.5k on the (trending, mark-out 10,482) leaderboard tape although it
  was +6.5k locally. Local gains below ~2-3 % do not transfer; the final defaults to the LAST upload -> re-upload
  `strategies/mm_r2_pred.py` (797,904). Round 1 is no longer a criterion (user).

## 2026-11-07 22:45 — Flip execution fix: quote_size 100 on mm_r2_pred = 383,300 (leaderboard test candidate) (home)

- Subagent B (`research/diag/flip_exec.py`, `strategies/exp_r2_new_b.py`): when the taker condition turns against the
  inventory, the passive quote on the favoured side holds all 200 lots of room, so the flip IOC waits for its CANCELED;
  25 % of these runs end with no IOC, inventory crosses 0 after a median 18 ms (mean 314 ms). Leaving room free
  (rsv / quote_size 100) makes the first IOC immediate and halves the wrong-way time.
- On mm_r2_pred (gate 8 / 30 s): quote_size 100 / 125 / 150 / 200 = 383,300 / 382,841 / 381,448 / 371,961 on round2,
  mirror 383,245 / 380,267 / 378,663 / 369,664, all 8/8. **`strategies/mm_r2_pred_qs.py`** (quote_size 100) baked as the
  next leaderboard test.

## 2026-11-07 22:50 — Leaderboard g6 786,205 (worse); queue-priority ceiling on mm_r2_pred_qs = 622k (home)

- Leaderboard: mm_r2_pred_g6 (gate 6 / 30 s) 786,205, mm_r2_pred2 787,414, mm_r2_pred 797,904 (final). Parameter
  changes worth < ~3-4 % locally behave like +-10k noise on the leaderboard tape.
- Flatten zone (IOC to flat when the signal is mildly against the inventory): -47..-257k. Rejected.
- Front-of-queue oracle on mm_r2_pred_qs: 383k -> 622k (+239k); 'fresh' mode (front only on levels with < 200 market
  lots) 410k: the value is in being early in THICK existing queues -> F-guarded early-placed ladder + taker
  (`exp_r2_ladderF.py`, subagent running).

## 2026-11-07 22:55 — Zero-delay take (subagent A): mm_r2_zd = 392,436 round2, better than mm_r2_pred on all 4 tapes (home)

- The flip IOC waited for the CANCELED of our own passive quote that held the room (subagent B measured it; 25 % of
  wrong-way take runs got no IOC). Subagent A's fix (`strategies/exp_r2_new_a.py`): zd_take counts the room of the
  quote being cancelled in the same callback, so the IOC and the cancel land together; a guard IOC trims any |I| > 100
  at once (penalty ~1k per tape, already inside the pnl); zd_quote 2 = zero-delay requote only toward the market.
- **`strategies/mm_r2_zd.py`** (mm_r2_pred gate 8 / 30 s + zd_take + zd_quote 2, quote_size 200): round2 **392,436**
  (+20.5k), mirror 390,836 (+21.2k), round1 457,289 (+15.7k), sample_v2 267,405 (+1.5k), 8/8 everywhere, breach
  penalty 50-1,057, inv q99.7 100, check_submission OK. Next leaderboard test (expected ~ +40k).
- mm_r2_zd100 (quote_size 100): round2 394,998 but round1 -50k / sample_v2 -29k -> not recommended.

## 2026-11-07 23:10 — Leaderboard mm_r2_zd = 849,747 (final); ladder and order-statistic F rejected (home)

- Leaderboard: `mm_r2_zd` **849,747** (+51.8k over mm_r2_pred; penalty 2,117, 23,239 filled orders) -> **final**.
  Public chart at 22:53: 75th percentile ~860k (+-10k), median 288k; the band's top edge has been stepping with our
  own uploads (~798k at 22:20 = mm_r2_pred, ~860k at 22:48 = mm_r2_zd), so we sit right on the top-25 % line.
- F-guarded small-order ladder (subagent, `strategies/exp_r2_ladderF.py`, `exp_r2_ladderF_zd.py`, ledger r2-ladF):
  no gain on either base (best +80 / -730 on round2 / mirror vs mm_r2_pred_qs; zd port -336 / -190). Deep rungs fill
  only when an informed sweep goes through the touch (-0.75 ticks/lot at 1 s for every guard margin); older touch
  orders earn +0.06..0.12/lot but only by using room the taker and touch quote need. Queue age is a dead end here.
- Order-statistic F (`strategies/exp_r2_os.py`): L = max(bid insert) + 1, U = min(ask insert) - 1 over 20-30 ms.
  Offline the 50/50 blend with the 70 ms EWMA lifts holdout corr with the 150 ms move 0.385 -> 0.412, but in the
  strategy every weight loses: round2 + mirror 783k (off) -> 748-755k (taker 0.25), 529-610k (taker 0.5),
  699-739k (quotes 0.3-0.5). Better bulk correlation, worse tails at the taker's decision points (as pred2 earlier).

## 2026-11-07 23:30 — Retreat pull: mm_r2_rtq = 406,182 round2 / 404,769 mirror (+13.7k / +13.9k over mm_r2_zd) -> upload (home)

- **`strategies/mm_r2_rtq.py`** = mm_r2_zd + rt_ms 20 (no passive quote for 20 ms after its side's touch retreats: market
  fills on a level the touch retreated onto mark out -0.57 ticks within 50 ms) + qs_k 0.45 / qs_weak 100 (the touch quote
  is the full 200-lot flip only while F_hat is >= 0.45 ticks beyond it, else 100 lots). round2 406,182, mirror 404,769
  (sum 810,951 vs 783,271), check_submission OK, research.check PASS 1.5 min.
- `strategies/mm_r2_rt.py` (rt only): sum 804,459 (+21.2k, rt 50 / 120 ms 803.8k / 802.1k: a plateau); round1 453,896
  (-3.4k), sample_v2 266,806 (-0.6k): neutral off-R2, so the gain is R2-mechanism specific, not a fit.
- Stacking: flow pull adds nothing on top of the retreat pull (both pull after adverse events); qs adds +6.5k.
- Rejected this half hour: jump over-cap (inserts beyond the 20-tick snapshot are invisible), sweep follow/fade, taker
  slip, AND-taker, order-statistic F, deeper taker.

## 2026-11-07 23:36 — Leaderboard mm_r2_rtq = 891,734 (+42k); next upload mm_r2_rts = 420,715 / 419,434 (home)

- Leaderboard: `mm_r2_rtq` **891,734.52** (24,723 filled orders, penalty 2,100, end -100): +42.0k over mm_r2_zd for
  +13.8k per local tape (x3.0).
- **`strategies/mm_r2_rts.py`** = mm_r2_rtq + fq_k_sp2 0.7 (subagent tkst: at a spread >= 2 a passive quote needs F_hat
  >= 0.7 ticks inside it; the base quoted both sides of a wide spread). round2 **420,715**, mirror 419,434 (+28.3k /
  +28.6k over mm_r2_zd, +14.5k / +14.7k over mm_r2_rtq), round1 450,845 (-6.4k), sample_v2 232,192 (-35k: another
  generator), check_submission OK, research.check PASS 1.6 min. Stack on round2+mirror: sp2 alone 826.1k, + rt 836.4k,
  + qs 840.1k (base 783.3k).
- Subagent predb (ridge with sweep / queue / clipped-fs70 features): no gain (offline corr 0.432 -> 0.462 but fewer
  triggers; the taker's pnl follows trigger count x edge, not corr). Thick-queue skip, improve-pull: worse.

## 2026-11-07 23:43 — Leaderboard mm_r2_rts = 957,133 (+65k); next upload mm_r2_rth = 430,781 / 429,296 (home)

- Leaderboard: `mm_r2_rts` **957,133.26** (22,630 filled orders, penalty 1,821): +65.4k over mm_r2_rtq for +14.6k per
  local tape (x4.5: the passive-side filters transfer even better than the taker changes).
- **`strategies/mm_r2_rth.py`** = mm_r2_rts + hole filter (hk_gap 3 / hk_k 0.8: when the next market level behind our
  side's touch is >= 3 ticks away, the touch quote needs F_hat >= 0.8 ticks inside it; sweeps through a level with a hole
  behind mark out -1.5/lot). round2 **430,781** (+10.1k), mirror 429,296 (+9.9k), check PASS 2.2 min.
- Rejected on rts: own-fill pull (758k), imbalance pull (693-766k), quote inside the spread (778k), fq_k 0.15 / 0.0,
  separate spread-3 threshold (+0.9k), rt / qs retune (+-2k).

## 2026-11-07 23:50 — Leaderboard mm_r2_rth = 970,101 (+13k); filter search on rth (home)

- Leaderboard: `mm_r2_rth` **970,101.26** (22,590 filled orders, penalty 1,821) -> final unless beaten. Two submissions left.
- Tried on rth (round2 + mirror sum, base 860,077), all within noise or worse: thin touch qty (761-829k), shallow depth
  1-2 ticks behind (829-850k), few orders at the touch (822-861k), mid-jump filter (850-860k), fresh-level looser join
  (859-861k), taker threshold over a hole (857-861k), fq_k_sp2 0.55 / 0.85 and rt 30 (849-860k).
- Subagents on rts: taker retune (fq_take_k, pred_blend, quote_size, fair_tau, fq_k) - nothing on both tapes; passive
  filters (pull after a trade at our touch -65..-70k, queue-ahead pull, spread-change lean, stale-signal lean) - nothing.
  Lesson: pulling a RESTING quote loses its queue place; filters only win where there is no queue place to lose
  (retreated touch, wide spread, a hole behind the touch).

## 2026-11-07 23:57 — Last candidates: mm_r2_rth2 (uploaded), mm_r2_rth3 (last submission) (home)

- `strategies/mm_r2_rth2.py` = mm_r2_rth + qs_k 0.3 + fq_k_sp3 0.3: round2 432,309, mirror 431,616 (+1.5k / +2.3k).
- `strategies/mm_r2_rth3.py` = mm_r2_rth2 + pred_blend 0.6: round2 436,496, mirror 439,487 (+4.2k / +7.9k), 8/8 chunks,
  holdout -0.5k / +0.4k (gain inside the ridge fit window), check PASS 1.5 min: a leaderboard gamble.
- Subagents on rth2: fine grid (qs_weak, qs_k, sp2/sp3, hk) - no gain; merged small filters (taker-over-hole, few orders,
  fresh level) +0.9k; taker retune: fq_take_k 0.15-0.17 and pred_blend 0.55-0.6 up on both tapes (mirror more).
- Mark as final whichever of mm_r2_rth (970,101), mm_r2_rth2, mm_r2_rth3 scores highest on the leaderboard.

## 2026-11-08 00:05 — Leaderboard mm_r2_rth3 = 954,077 (worse); last upload mm_r2_final (home)

- `mm_r2_rth3` (rth2 + pred_blend 0.6) scored **954,077** on the leaderboard, -16k under mm_r2_rth: the taker gain sat
  in the ridge's fit window (holdout was flat), as feared.
- **`strategies/mm_r2_final.py`** = mm_r2_rth + passive refinements only (fq_k_sp2 0.75, fq_k_sp3 0.15, hk_k 0.6, qs_k 0.3),
  pred_blend back to 0.5: round2 434,477, mirror 432,324 (+3.7k / +3.0k over mm_r2_rth), round1 450,065 (+0.4k),
  sample_v2 221,596 (-10.6k, another generator), check PASS 1.5 min. Last upload.
- Final choice: the highest leaderboard pnl among mm_r2_rth (970,101) and mm_r2_final. Summary for the presentation:
  research/reports/round2_final.md.
- Stability of mm_r2_final (round2 / mirror vs 434,477 / 432,324): fq_k_sp2 0.72 433,512 / 432,998, 0.78 433,565 / 432,085;
  hk_gap 2 433,351 / 432,142, hk_gap 4 identical; rt_ms 15 434,063 / 432,144, rt_ms 30 431,696 / 429,039. A plateau.
- mm_r2_final vs mm_r2_rth by chunk: round2 5/8 better (gain mostly chunk 7, +2.6k), mirror 6/8. Modest, consistent sign.
  Final pick rule stays: the higher leaderboard pnl of mm_r2_rth (970,101) and mm_r2_final.

## 2026-11-08 00:08 — Leaderboard mm_r2_final = 974,031 (best); last upload mm_r2_final2 (home)

- `mm_r2_final` scored **974,031.26** (22,486 filled orders, penalty 1,921): +3.9k over mm_r2_rth for +3.35k per local tape.
- Last upload `strategies/mm_r2_final2.py` = mm_r2_final with fq_k_sp2 0.77 (local +0.5k on the two tapes: a coin flip),
  check PASS 1.5 min. Final = the higher of mm_r2_final (974,031) and mm_r2_final2.
- Round 2 path on the leaderboard: 740,067 -> 797,904 -> 849,747 -> 891,735 -> 957,133 -> 970,101 -> 974,031.
- mm_r2_final2 local: round2 434,304, mirror 433,044 (vs final 434,477 / 432,324). fq_k_sp2 0.8: 432,801 / 431,517
  (qs_k 0.35: 433,108 / 431,547) - the 0.75-0.77 peak holds.
- Why mm_r2_final loses on sample_v2 (221,596 vs 267,405 for mm_r2_zd): the whole -45k is fq_k_sp2 (off: 266,625); the hole
  filter never fires there, rt / qs are neutral (220-222k). The wide-spread filter is R2-generator specific (+65k on the
  R2 leaderboard), so it fits the round-2 private tape but would hurt a sample_v2-like market.
