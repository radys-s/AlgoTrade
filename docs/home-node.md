# Domácí PC jako výpočetní uzel

Domácí PC (i7 Ultra, 32 GB RAM, RTX 5060 Ti 16 GB) dělá sweepy a trénink sítí. Ovládá se
přes Claude Code s Remote Control; člověk u PC kliká jen na začátku.

## Co udělá člověk u PC (~5 kliknutí)

1. Zapnout PC a přihlásit se do Windows.
2. Otevřít aplikaci Claude (není-li, stáhnout z claude.ai/download a přihlásit se Honzovým
   účtem — kód z e-mailu nadiktuje Honza).
3. Záložka **Code** → nová session v libovolné složce → zapnout **Remote Control**.
4. (Volitelně) Nastavení → Systém → Napájení → režim spánku **Nikdy**. Jinak to vyřeší
   nastavení keep-awake v aplikaci Claude, které Honza schválí na dálku.

Pak už nic. Pozor jen na okno UAC („Chcete povolit této aplikaci…“): to se dá odkliknout
jen fyzicky u PC, proto bootstrap níže instaluje vše bez administrátorských práv.

## Bootstrap zpráva (Honza ji vloží do vzdálené session)

```text
You are the team's home compute node for the Qminers Algo Trade Arena hackathon. Nobody is at
this PC; I (Honza) approve your permission prompts remotely. Never trigger a UAC/admin prompt:
install only per-user (no MSI/machine-wide installers), and if something needs admin, stop and
tell me. Don't change Windows system settings.

1. Report: CPU model and core/thread count (Get-CimInstance Win32_Processor), RAM, GPU
   (nvidia-smi if present), free disk on C:, and whether git, gh and uv are already on PATH.
2. Missing tools, per-user only:
   - uv: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   - git (only if missing): MinGit 64-bit zip from the official git-for-windows GitHub releases,
     extracted to %LOCALAPPDATA%\Programs\MinGit, its cmd\ folder added to the user PATH.
   - GitHub CLI: the windows_amd64 .zip from the official cli/cli GitHub releases, extracted to
     %LOCALAPPDATA%\Programs\gh, its bin\ folder added to the user PATH.
3. GitHub login without a password at this PC: run `gh auth login --hostname github.com
   --git-protocol https --web` and show me the one-time code; I confirm it on my phone at
   https://github.com/login/device. Then `gh auth setup-git`.
4. Clone: git clone https://github.com/radys-s/AlgoTrade.git %USERPROFILE%\QuantHack, then in it:
   git checkout -b home origin/main (our ledger branch), uv sync.
5. Smoke test: `uv run python -m gzip -dk data/sample.csv.gz` (keep the .gz) then
   `uv run python -m research.run --strategy strategies/trend_follower.py --data data/sample.csv --no-ledger`
   must print pnl 1878. Then measure parallel throughput: run research.run on strategies/mm.py
   with --grid '{"quote_size": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]}' --workers 16 --no-ledger
   and report the total wall time and the per-run seconds.
6. GPU torch for MLP training (separate from the project env, ~3 GB download):
   uv run --with torch --index https://download.pytorch.org/whl/cu128 python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
7. Turn on keep-awake in the Claude app settings (ask me to approve), so the PC doesn't sleep.
8. Report back: everything above, plus the folder path. Then wait for sweep jobs. Every job's
   results go to research/results/*.csv: commit them on branch `home` and push (never rebase,
   squash or force-push).
```

## Jak se mu posílá práce

- Data kola: Honza je stáhne z arény na notebooku a nahraje `.gz` jako asset do privátního
  GitHub release (`gh release create data-<kolo> data/<kolo>.csv.gz`); domácí uzel ho stáhne
  `gh release download data-<kolo> -D data` a rozbalí.
- Strategie: přes git (větev `herzic` → `git fetch` + checkout souboru).
- Výsledky: ledger soubory na větvi `home`; hlavní session je mergne. Soubory se nepřepisují,
  takže konflikty nevznikají.
- Sdílený limit: obě session jedou na jednom účtu Claude Max. Domácí uzel má pálit CPU, ne
  tokeny, takže velké vlny subagentů běží jen na jednom stroji.
