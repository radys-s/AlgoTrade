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
