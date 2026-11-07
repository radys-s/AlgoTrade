# Poznatky a proč — předávka pro Honzu (a jeho Claude)

Stav k 2026-11-07 11:20, autor: okno `radys`. Co víme, proč to tak je a jak navázat.
Podrobná čísla a historie: [progress.md](progress.md). Pravidla soutěže: [kontext.md](kontext.md).

## TL;DR

1. **Lokální simulátor = server, přesně.** Stejný soubor dal lokálně i na dashboardu stejné pnl
   (44 116 i 50 965), stejné filly i stejný objem. Lokální skóre na veřejných datech je tedy totéž,
   co ukáže dashboard. Submisi nepotřebujeme na zjišťování skóre, jen na ověření, že kód projde sandboxem.
2. **Playground = syntetický sample ze starteru** (bajtově stejný jako `data/sample.csv.gz`).
   Starter README říká, že kola jsou nahraná reálná data a vypadají jinak. Závěry o trhu níže platí jen
   pro sample. Přenosné jsou nástroje a postup.
3. **Aktuální final v playgroundu:** `strategies/momentum_hold.py`, pnl **50 965** (reference `trend_follower` 1 878).
4. Na sample datech **vyhrává momentum taker a market making prodělává.** Proč, viz níže.

## Nástroje (ve větvi `radys`, teď i v `main`)

| Soubor | Co dělá | Proč existuje |
|---|---|---|
| `research/analyze_tape.py <csv>` | Statistiky pásky (rychlost, G/IOC, velikosti, životnost příkazů), knihy (spread, objem na touch, volatilita, autokorelace) a korelace signálů s budoucím midem na 10 ms až 5 s; pohyb midu po obchodu | Na nových datech kola **jako první krok**: za 3 s řekne, jestli je trh momentový nebo reverzní a jestli má cenu market making. Book staví stejným simulátorem (pasivní recorder), takže vidí přesně to, co strategie. |
| `research/grid.py <strategie> <csv> P=a,b ...` | Paralelní grid parametrů (atributy třídy), výsledky do `research/results/*.csv`, PnL rozpadlé na čtvrtiny času | Ladění bez plýtvání submisemi (máme jich 10 na kolo). Čtvrtiny kontrolují, že zisk není z jednoho úseku (overfitting). 40 kombinací = 7 s. |
| `research/check_submission.py <file>` | AST kontrola pravidel sandboxu: importy, `open/eval/getattr/type`, dunder atributy, `.format`, 1 MiB, jedna podtřída `BaseStrategy` bez argumentů | Neúspěšná submise stojí slot. Je to přibližná kontrola, autorita je server. |

Spuštění: `uv run python -X utf8 research/<skript>.py ...` z kořene repa (`-X utf8` kvůli češtině ve Windows konzoli).

## Proč na samplu funguje momentum a ne market making

Výstup `analyze_tape.py` na `sample.csv`:
- **Spread je vždy 2 ticky** a na každé straně touch stojí typicky jediný příkaz ~25 lotů. Imbalance knihy
  proto nenese informaci (korelace s budoucím midem +0,03 až +0,08).
- **Tok obchodů je silný signál:** EWMA znaménkového objemu koreluje s budoucí změnou midu +0,65 na 1 s.
  Po obchodu se mid posune ve směru agresora o +0,5 ticku za 100 ms, +3,7 za 1 s a +10 za 5 s.
- **Market maker** bere na fillu půl-spread, tedy 1 tick, ale v dalších sekundách ztratí ~3,7 ticku
  (adverse selection). To je čistá ztráta.
- **Taker**, který jde s tokem, zaplatí 1 tick (půl-spread) a 5 ms latenci, ale do 5 s získá ~10 ticků.

## `momentum_hold.py`: jak funguje a proč tyto parametry

- **Vstup:** v callbacku s nenulovým čistým tokem (`ask` = koupil agresor → +) pošle IOC po směru,
  cena touch ± `SLIP_TICKS`, velikost `QTY`, jen když nejsme v protisměrné pozici.
- **Výstup:** po `HOLD_US` od prvního fillu IOC celé pozice. Při toku proti pozici zavírá dřív.
- **Bezpečnost:**
  - Inventář se počítá **jen z fillů**.
  - Vždy jen jeden příkaz v letu (`in_flight`), takže 5ms latence nikdy neudělá dvojitou pozici.
  - Strop `MAX_POS = 80`, rezerva pod pokutovým pásmem 100. Pokuta je 10 000 $/s, takže jedna sekunda nad
    limitem smaže několik hodin zisku.
- **Grid na samplu (40 kombinací):** QTY 50, HOLD 5 s, SLIP 3 → 50 965, pokuta 0, |I| q99,7 = 45,
  čtvrtiny 12,9k / 11,5k / 13,9k / 12,6k (rovnoměrné).
  - HOLD 5 s odpovídá horizontu, kde je momentum nejsilnější. Při HOLD 0,5 s je výsledek výrazně horší.
  - QTY nad ~25 skoro nepomáhá, protože víc likvidity na touch není.

**Známé slabiny (před Round 1 opravit):**
- `FLATTEN_AT_US = 3 590 s` je natvrdo pro hodinový sample. U jiné délky kola je potřeba konec dat
  odhadnout jinak, nebo přepočítat z délky veřejného CSV kola (privátní data mají nejspíš podobnou délku,
  neověřeno).
- Na reálných datech momentum nemusí existovat (může jít o reverzi). Rozhodne `analyze_tape.py` na datech kola.

## Doporučený postup v každém kole

1. Stáhnout veřejná data → `data/`, rozbalit.
2. `analyze_tape.py`: momentum, nebo reverze? Spread 1 nebo víc ticků? Má imbalance signál?
3. Podle toho vybrat modul (momentum taker / market making se skew / hybrid) a pustit `grid.py`.
4. Robustnost: preferovat parametry, které jsou dobré ve **všech čtvrtinách** a v okolí optima
   (plochý vrchol), ne jediný ostrý vrchol. Hodnotí se na privátních datech.
5. `check_submission.py` → submise → ověřit na dashboardu, že pnl = lokální.
6. Zapsat do `progress.md` i neúspěšné pokusy. Mechaniky nemazat, nechávat za parametrem.
