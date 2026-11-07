# Mozek týmu (research lead)

Session, která rozhoduje, co zkoumat, navrhuje experimenty, řídí subagenty, interpretuje výsledky a
doporučuje submise. Běží na **domácím PC** (20 workerů, GPU) s nejsilnějším dostupným modelem.
Honza ji řídí přes Remote Control, Tomáš submituje, Honzův notebook dělá provoz (data, publikace,
merge do `main`, příprava submisí). Spodní část souboru si mozek sám udržuje aktuální.

## Startovní prompt (vložit do nové session na domácím PC, ve složce repa)

```text
You are the RESEARCH LEAD ("brain") of team p<0.05 in the Qminers Algo Trade Arena hackathon.
Rounds: R1 13-18, R2 19-24, R3 01-06, R4 07-12 (weights 0.8/1/1/1.2). Round score = z-score of PnL
across teams on PRIVATE data of the same market; PnL <= 0 counts as 0. Steady positive PnL in every
round beats one big hit. You run on the home PC (20 CPU workers, RTX GPU, fast internet). Honza drives
you via Remote Control and approves prompts. Tomáš (laptop, branch radys) uploads submissions. Honza's
laptop session does operations: downloads round data, publishes it as release data-<round>, merges into
main, prepares submission files.

YOUR JOB: decide what to research, design experiments, get them executed by subagents and CPU tools,
interpret results, and recommend what to submit. Keep your own context for thinking and decisions:
delegate implementation, sweeps and report reading to subagents (Agent tool in background, or the
Workflow tool with scriptPath research/workflows/research_wave.js). Do trivial checks yourself.

READ FIRST: docs/brain.md (your own decision log and beliefs: continue it), docs/workflow.md
(protocol), STATUS.md, the end of docs/progress.md, submission/README.md, the skills in .claude/skills
(arena-market-mechanics, run-experiment, round-kickoff, signal-research, pre-submit), and the newest
research/reports/*_kickoff.md with its data card, taker scan and mm_edge reports.

TOOLS (CPU, no tokens): research.round (kickoff: --fetch <round>), research.run (sweeps, --halving,
--workers), research.compare (paired-chunk verdict), research.check (sandbox rules + runtime),
research.ledger (all runs by everyone), research.signals, research.taker_scan, research.mm_edge,
research.features, research.train_mlp (GPU env %USERPROFILE%\torch-env).

LOOP (every 45-60 min of a round):
1. State: git fetch; merge origin/main into home (merge, never rebase); read new ledger rows and reports.
2. Decide: list hypotheses with expected value and evidence; pick at most 6; for each fix the file,
   the flag, the grid, the datasets and the acceptance test before running anything.
3. Execute: subagents or a research wave, e.g. args {data, prior, champion: "submission/best_strategy.py",
   wave, workers: 3, models: {experiments: "sonnet", propose: "sonnet"}}. Keep the total of worker
   processes across everything you run at <= 20.
4. Interpret: paired chunks, holdout, robustness on earlier real datasets, parameter plateaus; distrust
   single big wins and anything that only works in one chunk. Check runtime and penalty every time.
5. Recommend: when research.compare says ACCEPT and research.check passes, tell Honza exactly what to
   submit (file, baked parameters, expected public PnL, why, risk). Humans promote and submit.
6. Record: docs/progress.md entry per experiment (append-only, template at its end); update the
   "Aktuální přesvědčení" and "Rozhodnutí" sections of docs/brain.md; commit + push to home at least
   every 30 minutes.

PRINCIPLES: identify the regime first (momentum vs reversion, spread, depth); robustness beats the
peak; the +-100 position penalty (10,000/s) must be impossible by construction; runtime under 7 min
per 8.8M rows; plan the 10 submissions per round (insurance early, final explicitly set);
quota: Claude Max is shared with Honza's laptop session, so use the strongest model only for your own
reasoning and for skeptic/integration steps, sonnet for implementation and sweeps, waves of <= 6
agents, and plain CPU sweeps whenever an agent is not needed.

HARD RULES: never trigger UAC/admin prompts, never change Windows settings, never submit or enter
passwords, work on branch home only, never rebase/squash/force-push, never edit
submission/best_strategy.py (propose; humans promote).

REPORTING: after each wave or decision, 5-10 lines to Honza: what was tested, numbers, what we learned,
what runs next, and at most one yes/no question.
```

## Aktuální přesvědčení (mozek přepisuje, stav 2026-11-07 12:30)

- **Realistická data (`sample_v2`, 2 M řádků, 1,6 h):** trh se na úrovni ticku vrací (P(reverze) 0,63),
  spread medián 2 ticky (1 tick jen 41 %), 1s volatilita 0,5 ticku.
- **Takeři prodělávají:** taker scan 1 152 konfigurací, žádná kladná; momentum champion -2,44 M.
- **Pasivní market making vydělává:** maker na čele fronty +0,77 ticku/fill (8/8 chunků); `mm_v2`
  (quote 100, max_pos 100, skew 0) veřejně 237 306, na serveru 240 995 → practice round 3. místo
  (vedoucí ~291 k). Server ≈ lokál (±1,5 %).
- **Páky:** velikost kotace dominuje; microprice skew (`alpha_imbalance` 1,0) +11 % při quote 20;
  zlepšování ceny uvnitř spreadu mírně škodí; `max_position` 100 = bez rezervy a 100 lotů do markoutu.
- **Edge je jen na touch (12:45).** Cokoliv, co nás posune dovnitř spreadu, prodělává, i když
  získá víc fillů: `improve` −0,5 %, `skew` −33 %, `hold_inside` (zůstat o tick uvnitř po ústupu
  trhu) −11 % při +27 % fillů, `improve_unwind` (odvíjet o tick uvnitř při stropu) −16 %. Filly
  uvnitř spreadu jsou adverzní: cena jde skrz nás. mm_edge „front of queue +0,77“ platí pro touch.
- **Páka na objem = velikost na touch, ne cena.** Invariant (B)/(S) dovoluje straně, která |I|
  snižuje, až M + |I| lotů; `quote_size` 100 to kapoval. `quote_size` 200 (= vše volné):
  `sample_v2` 237 306 → **263 618 (+11 %)**, holdout +12,7 %, párové chunky 8/8 (+2,1 až +2,8 k
  každý), breach 0, runtime stejný (68 s / 8,8 M). 200 = 300 (saturace). Soubor
  `strategies/exp_full_capacity.py` (champion + jedno číslo), check PASS.
- **Round 1 (13:30):** spread 1 tick 93 % času, náhodná procházka, mírné momentum po obchodech.
  `exp_full_capacity` 425 708 vs champion 321 024 (**+33 %**, 8/8 párových chunků, holdout +34 %);
  Tomáš ho povýšil do `submission/best_strategy.py` (main 1ab3b4d). Efekt větší než na sample_v2,
  protože R1 má víc IOC toku.
- **Na R1 nic dalšího nepomáhá (wave 1, ~80 běhů):** imb_pull 0,5–0,95 (−5 až −55 %), alpha 1,0
  (−44 %), min_requote 100 ms (−14 %), max_position 90/80 (−4/−9 %), hold_inside (−8 %), jump_imb,
  flow_pull (0/8 chunků), queue_join (vstup jen do tenkých hladin −2 až −44 %, pull při hluboké
  frontě −9 až −61 %). Vzorec: každý filtr ubírá filly s kladnou hodnotou a pull ztrácí místo ve frontě.
- **Fill attribution kandidáta na R1** (`research/reports/round1_fill_attribution.md`): markout 1 s
  ≈ PnL (431 k vs 426 k); edge při fillu 0,505 $/lot, markout 0,16 → 0,34 ticku driftu proti nám.
  Filly z konce fronty (201+ lotů před nami) jsou **kladné** (+0,12 $/lot; sweepy ≥ 100 lotů
  +0,10), protože příkazy před nami se ruší (medián života 0,34 s) – mm_edge statická pozice to
  podhodnocuje. Jediná záporná kategorie: imbalance proti straně **a zároveň** > 200 lotů před nami
  (−46 k na 276 k lotů); s imbalance pro nás vydělává každá pozice (+0,37 až +0,51 $/lot).
  Čas bez živé kotace 26 % na stranu: strop 12–15 %, in-flight 0,5 %, **„jiné“ 10–14 %** v úsecích
  100–800 ms při |I| 70–89 → podezření na kódovou cestu (replacement až po potvrzení cancelu?).
- **Parkování stubů (Tomáš, 14:45 → `mm_park`, v submission):** když trh zlepší cenu a kotace se musí
  posunout, starý příkaz se `modify_order` zmenší na 20 lotů místo cancelu → drží místo ve frontě
  (simulátor při zmenšení pořadí nemění). round1 **452 651** (+6,3 %, 8/8 chunků, holdout +6,9 %),
  sample_v2 −4,7 % (0/8). Ověřeno nezávisle na home i na notebooku (shoda na jednotku), check PASS,
  54 s / 8,8 M. Jemný grid (45 buněk × 2 datasety, `r1-park-grid`): **depth 1 jednoznačně**, depth
  2–3 horší všude; round1 plató size 15–30 × max 3–5 (453–455 k, ±0,6 %), baked 20/1/3 je v jeho
  středu; sample_v2 chce menší stuby (10/1/2 = 259 k, bez parkování 264 k). Součet r1+v2 je plochý
  (15/1/5 ≈ 20/1/3 ≈ 20/1/5) → **neměnit**. Pro R2 v mean-reverting režimu zvážit park 10 / max 2
  nebo vypnout (rozhodne 4-buňkový test na kickoffu).
- **ML taker overlay (H10, `exp_ml_taker`, lineární ridge 1 s fit na 75 % round1):** IOC při silném
  signálu nad makerem, odvíjení pasivně. Všech 12 buněk hluboko pod kandidátem (−9 % při prahu 1,2 /
  20 lotů, −32 až −37 % s 50 loty), monotónně s intenzitou. Taking v tomto simulátoru neplatí ani
  s pasivním výstupem → GPU/HGB verze nemá smysl.
- **Standing ladder (H11):** neutrální (+0,1 %, 5/8), když má touch přednost; destruktivní (−63 %),
  když příčky berou kapacitu touch. Kapacita na touch je cennější než cokoli v hloubce.
- **Fill attribution mm_park (14:45):** stuby = naše nejlepší filly (0,39 $/lot vs 0,18 touch; 60 % lotů
  s 0 před nami), cena parkování = vytlačený bulk (−18 % lotů touch); na v2 je bulk 3× cennější
  (0,49 $/lot) → parkování tam stojí 4,7 %. Parkování půlí čas na stropu (50 → 26 %). 37 % zlepšení
  flickeruje zpět do 500 ms, přesto se je vyplatí sledovat hned (H13, min_requote prodělaly).
- **Koordinace:** Tomáš nezávisle testoval ladder a back-pull (= moje H11, H7) → před spuštěním agenta
  vždy číst `docs/radys-findings.md` a jeho ledger tagy.
- **mm_ml (Tomáš, 15:40 → submission, main 1769eaa): round1 490 460 (+8,4 % nad mm_park), 8/8 chunků,
  holdout 126 291.** Ridge 100 ms (IC 0,38) → když |pred| > 0,15 ticku proti straně, zmenšit její kotaci
  přes modify na 5-lotový stub (drží místo), po odeznění signálu dokotovat. Plató thr 0,1–0,15, stub 5–10.
- **Sebekritika (14:50):** drift 0,33 ticku po fillu byl největší číslo na stole a já ho odepsal, protože
  cancel-filtry prodělávaly a „shrink ≈ cancel“. Špatně: shrink přes modify drží čelo fronty a vrací loty
  až po odeznění signálu, cancel ne. Správný postup byl testovat každou páku i ve verzi „přes modify“.
  Obecně: každé pravidlo „kdy nekotovat“ implementovat jako zmenšení, nikdy jako cancel.
- **Plán na R2 (19:00):** kickoff (startovní relace) → data card: režim spreadu (1 vs 2 ticky), reverze
  vs momentum po obchodech. Hned CPU grid championa mm_park na R2 + round1 + sample_v2:
  `--grid '{"park_size": [0, 10, 20], "park_max": [2, 3, 5], "park_depth": [1]}'` (0 = park off) a
  `--grid '{"quote_size": [100, 200]}'` pro jistotu; compare vítěze vs champion → doporučení do T+20 min.
  Mechaniky z R1 netestovat znovu, pokud režim nepřinese nový důvod (viz tabulka zamítnutých).
- **Otevřené otázky:** `max_position` 100 je monotónně nejlepší, trendující privátní páska je riziko
  sdílené s championem; příčky místo parkovaných stubů (netestováno, nízká priorita).

## Rozhodnutí (append-only)

| Čas | Rozhodnutí | Proč |
|---|---|---|
| 12:10 | Syntetický `sample.csv` vyřazen z robustnostní sady | jiný svět; penalizoval by vše, co funguje na realistických datech |
| 12:15 | Champion = `mm_v2` (Tomáš) | `sample_v2` +237 k, 8/8 chunků; momentum -2,44 M |
| 12:44 | Kandidát na pojistnou submisi R1 = `strategies/exp_full_capacity.py` (`quote_size` 200), pokud `research.compare` na R1 public řekne ACCEPT; jinak champion | +11 % na sample_v2, 8/8 párových chunků, žádná nová mechanika (jen víc lotů na odvíjející straně v mezích invariantu) |
| 12:44 | H3 `improve_unwind` a H4 `hold_inside` zamítnuty, flagy zůstávají v `strategies/exp_*.py` vypnuté | −16 % / −11 % na sample_v2; filly uvnitř spreadu jsou adverzní |
| 12:45 | Wave 1 na R1 = CPU sweep championa (quote_size × imb_pull × alpha_imbalance × min_requote_us, + max_position 80/90), žádný nový kód, dokud nepřečtu data card R1 | nejdřív režim, pak mechaniky; tokeny šetřit na interpretaci |
| 13:05 | Doporučení: `exp_full_capacity` jako submise R1 (ACCEPT na round1 + sample_v2, check PASS) | +33 %, 8/8 chunků, žádná nová mechanika |
| 13:08 | Honza: nespěchat se submisí (10 slotů, server = lokál); submitovat až promyšleně | slot je vzácnější než čas |
| 13:30 | Wave 1 uzavřena bez změny kandidáta; wave 2 = H7 `deep_pull` + rozbor času bez kotace (kódová cesta) místo dalších filtrů | všech ~80 běhů filtrů prodělává; jediný záporný segment fillů je imbalance proti × hluboká fronta |
| 13:55 | **R1 finál = `submission/best_strategy.py` = Tomášův `mm_park` (452 651), parametry 20/1/3 nechat** | ověřeno nezávisle (home + notebook), ACCEPT 8/8 na round1; grid: plató ploché, baked hodnoty v jeho středu; −4,7 % na v2 je pro R1 irelevantní (privátní data = trh round1) |
| 13:55 | ML taker overlay zamítnut i v levné lineární verzi; GPU/HGB verze se nestaví | −9 až −37 % ve všech 12 buňkách |
| 14:50 | Nový champion mm_ml (Tomáš). Home staví nad ním: jemný grid thr × stub, asymetrické prahy (přidávající vs odvíjející strana), MLP místo ridge na GPU; R1-tuned stuby (park_max 5 + improve_stub 20) přidat až po nich | shrink přes modify je páka na drift; signál s vyšším IC a správně mířený shrink jsou přímé násobky zisku |
| 14:25 | R1 submise hotová: Tomáš nahrál mm_park, server 452 651 = lokál, final. Výzkum R1 mechanik ukončen (H12, H13 REJECT, attribution bez nové páky); do R2 klid + příprava kickoff gridu | 13 mechanik, ~230 běhů; zbývající nápady jsou ±1 % a režimově závislé |
| 13:50 | (dříve) R1 finál = `submission/best_strategy.py` (exp_full_capacity, 425 708). Výzkum R1 mechanik zastaven (H7–H9 také REJECT, ~110 běhů); nahrát nejpozději ~17:00 a nastavit final; tokeny šetřit na R2 | žádná z 9 mechanik ani 5 os nepřekonala kandidáta; strop pozice drží 50 % času a mimo strop je kotace živá |
