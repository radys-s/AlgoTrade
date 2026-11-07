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
| 2026-11-07 12:10 | herzic | Pipeline kola `research.round` + taker scan + MM edge; generálka na `sample_v2`: momentum -2,44 M, MM +28 025 (8/8), trh mean-reverting |

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
