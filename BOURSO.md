# Bourso CLI

CLI non-officiel pour BoursoBank, installe depuis [azerpas/bourso-api](https://github.com/azerpas/bourso-api).

## Installation

Compile depuis source (Rust) le 2026-06-17 :
```bash
cd /tmp && git clone --depth 1 https://github.com/azerpas/bourso-api.git
cd bourso-api && cargo build --release
cp target/release/bourso-cli ~/.local/bin/
```

Binaire : `~/.local/bin/bourso-cli` (v0.5.3)

Pour mettre a jour :
```bash
cd /tmp/bourso-api && git pull && cargo build --release && cp target/release/bourso-cli ~/.local/bin/
```

## Configuration

```bash
bourso-cli config --username <CUSTOMER_ID>
```

Le mot de passe est demande a chaque execution (jamais stocke).

**Attention** : une connexion depuis une IP inhabituelle declenche le MFA BoursoBank.

## Commandes

### Lister les comptes
```bash
bourso-cli accounts
```
Retourne les IDs de compte necessaires pour les autres commandes.

### Passer un ordre
```bash
bourso-cli trade order new --side buy --symbol 1rTCW8 --account <ACCOUNT_ID> --quantity 4
```
- `--side` : `buy` ou `sell`
- `--symbol` : ID Boursorama du tracker (visible dans l'URL, ex: `https://www.boursorama.com/bourse/trackers/cours/1rTCW8/`)
- `--quantity` : nombre de parts

### Cotation (sans authentification)
```bash
bourso-cli quote --symbol 1rTCW8 --length 30 last
```
Sous-commandes : `highest`, `lowest`, `average`, `volume`, `last`

Periodes (`--length`) : 1, 5, 30, 90, 180, 365, 1825, 3650 jours.

### Transfert entre comptes
```bash
bourso-cli transfer --account <FROM_ID> --to <TO_ID> --amount 100
```
Montant minimum : 10 EUR.

### Export de transactions
```bash
bourso-cli export transactions --account <ACCOUNT_ID> --start-date 01/01/2025 --end-date 17/06/2026 --format csv --output transactions.csv
```
Formats : `csv`, `json`. Sans `--output`, ecrit sur stdout.

## Symboles utiles

| ETF | Symbole Bourso | ISIN |
|-----|---------------|------|
| Amundi PEA Nasdaq-100 (PUST) | 1rTPUST | FR0011871110 |
| Amundi Nasdaq-100 2x (LQQ) | 1rTLQQ | FR0010342592 |
| Amundi MSCI World (CW8) | 1rTCW8 | LU1681043599 |
| Amundi PEA S&P 500 (PE500) | 1rTPE500 | FR0013412285 |

---

# Execution automatique PEA

## Architecture

```
22:30  cron backtest (run.py QQQ)  →  outputs/qqq_strategy/signal.json
09:05  cron PEA (real_bourso.py)   →  lit signal, execute PUST sur Euronext
```

- **ETF**: PUST (Amundi PEA Nasdaq-100), symbole Bourso `1rTPUST`
- **Levier**: x1 uniquement (LQQ x2 trop cher par part a 2000+ EUR)
- **Compte**: PEA DESCAMPS

## Cron

Le crontab **doit** contenir :

```cron
PATH=/home/greg/.local/bin:/usr/local/bin:/usr/bin:/bin

30 22 * * 1-5 cd /home/greg/data_local/code/MyQTMv2 && XGBOOST_DEVICE=auto ./venv/bin/python -m src.risk_off_strategy.run QQQ >> logs/cron_backtest.log 2>&1 && XGBOOST_DEVICE=auto ./venv/bin/python -m src.risk_off_strategy.compare_pit QQQ >> logs/cron_backtest.log 2>&1

5 9 * * 1-5 cd /home/greg/data_local/code/MyQTMv2 && ./venv/bin/python -m src.real_bourso --execute >> logs/cron_pea.log 2>&1
```

### Pieges cron rencontres

1. **`PATH` manquant** — `bourso-cli` est installe dans `~/.local/bin/` qui n'est pas dans le PATH par defaut du cron. Sans la ligne `PATH=...` en tete du crontab, `real_bourso.py` echoue avec `No such file or directory: 'bourso-cli'`. **Incident du 19 juin 2026** : le premier cron matin a echoue pour cette raison.

2. **`cd` obligatoire** — les commandes cron s'executent depuis `$HOME`, pas depuis le repertoire du projet. Sans `cd /home/greg/data_local/code/MyQTMv2 &&` devant chaque commande, les imports Python et les chemins relatifs (`logs/`, `outputs/`) echouent.

3. **Contention GPU** — le backtest du soir peut tourner en parallele d'autres workloads GPU. `XGBOOST_DEVICE=auto` laisse `_detect_device` (dans `backtest.py`) utiliser le GPU quand il est libre et basculer sur CPU quand un autre process l'occupe (verifie via `nvidia-smi` : process compute residents + utilisation). Forcer avec `XGBOOST_DEVICE=cpu` ou `=cuda` si besoin.

## Signal (signal.json)

Le backtest ecrit `outputs/qqq_strategy/signal.json` :

```json
{
  "status": "ok",
  "ticker": "QQQ",
  "date": "2026-06-18",
  "probability": 0.85,
  "allocation": 1.0,
  "timestamp": "2026-06-18T22:35:00"
}
```

- `status`: `"running"` au debut du backtest, `"ok"` a la fin. Si crash, reste `"running"` et le matin refuse d'executer.
- `allocation`: 0.0 (cash) a 1.0 (full invest), calcule via `(prob - 0.70) / (0.75 - 0.70)`.
- Le script du matin verifie la fraicheur du signal : age max `MAX_SIGNAL_AGE_HOURS=90h`. Assez large pour tolerer les week-ends/feries (lundi matin = signal du vendredi soir ~58h ; long week-end jeu. soir → mar. matin ~82h). Au-dela = le backtest du soir s'est arrete → refus.

## Frais et seuils

- **Achat**: 0% (ETF gratuit sur Bourso PEA)
- **Vente**: 0.5% → vente seulement si delta allocation >= 20% (`SELL_THRESHOLD`)
- **Emergency OFF**: creer `logs/emergency_off.json` avec `{"active": true}` pour forcer allocation a 0%

## Scripts

| Script | Role |
|---|---|
| `src/real_bourso.py` | Execution matin: lit signal, PEA prepare, achat/vente PUST |
| `src/bourso/prepare.py` | Dry-run: affiche prix, cash, capacite d'achat |
| `src/bourso/execute.py` | Execution manuelle interactive (PEA ou CTO) |
| `src/bourso/list_accounts.py` | Liste tous les comptes et soldes |

## Logs

- `logs/cron_backtest.log` — sortie du backtest du soir
- `logs/cron_pea.log` — sortie de l'execution matin
- `logs/trades.jsonl` — historique des ordres (lu par la webapp)

## Execution manuelle

```bash
# Dry-run (voir ce qui serait fait)
python -m src.real_bourso

# Execution reelle
python -m src.real_bourso --execute

# Ordre manuel interactif
python -m src.bourso.execute pea PUST buy 4
```
