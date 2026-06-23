# Bourso CLI

CLI non-officiel pour BoursoBank, base sur [azerpas/bourso-api](https://github.com/azerpas/bourso-api),
avec un **fork maison** ([gdescamps/bourso-api](https://github.com/gdescamps/bourso-api)) qui ajoute
la sous-commande `trade summary`.

## Installation (fork maison + patch `trade summary`)

`azerpas/bourso-api` n'expose **pas** de commande CLI pour lire l'etat d'un compte
trading (cash/positions) — la fonction existe dans la lib (`get_trading_summary`)
mais n'est pas cablee au CLI. Or `src/bourso/prepare.py` (recap du soir + execution
du matin) en a besoin. On maintient donc un **fork** avec un patch minimal.

Le submodule `external/bourso-api` pointe sur le fork, branche **`myqtm`** =
tag upstream (actuellement **v0.5.3**) + 1 commit `feat: add 'trade summary' CLI`.
Deux remotes dans le submodule :

| Remote | URL | Usage |
|---|---|---|
| `origin` | `git@github.com:gdescamps/bourso-api.git` | notre fork (branche `myqtm`, push) |
| `upstream` | `https://github.com/azerpas/bourso-api.git` | tags + observation de `main` (lecture) |

Apres un clone du depot principal :
```bash
git submodule update --init external/bourso-api
./2_bourso_cli_update.sh            # compile le commit epingle (myqtm) + installe dans ~/.local/bin
```

Binaire : `~/.local/bin/bourso-cli` (v0.5.3 + patch `trade summary`)

Quand un nouveau tag azerpas sort (le cron de 20h alerte), MAJ manuelle :
```bash
./2_bourso_cli_update.sh --pull     # rebase myqtm sur le dernier tag azerpas, build, installe
cd external/bourso-api && git push origin myqtm --force-with-lease && cd -
git add external/bourso-api && git commit -m "chore: bump bourso-cli a vX.Y.Z"
```

Le script `2_bourso_cli_update.sh` :
1. initialise le submodule s'il manque ;
2. (`--pull`) `fetch upstream --tags` puis **rebase `myqtm` sur le dernier tag** azerpas ;
3. compile (`cargo build --release`) le commit epingle ;
4. copie le binaire dans `~/.local/bin/` ;
5. verifie que la version installee == version du `Cargo.toml`.

> Le patch lui-meme (`trade summary`) expose la fonction officielle
> `BoursoWebClient::get_trading_summary` ; voir le commit sur la branche `myqtm`
> du fork. `src/bourso/prepare.py` parse son JSON (cash, valuation, positions).

## Tests + verification quotidienne

`tests/` contient des tests pytest qui valident bourso-cli a deux niveaux :

- **Build dry-run** (sans auth ni ordre reel) : presence du binaire, version ==
  pin du submodule, sous-commandes attendues (`accounts/config/trade/quote/transfer`),
  **presence du patch `trade summary`**, aide de chaque sous-commande, **absence de
  panic Rust**, parsing JSON des wrappers.
- **Connexion compte reel** (`test_prepare_live`, marker `live`) : equivalent de
  `prepare.py` — un vrai `bourso-cli trade summary` sur le PEA (auth + lecture
  cash/valuation/positions, aucun ordre). **Desactive par defaut**, actif seulement si
  `BOURSO_LIVE_TESTS=1`, pour ne pas authentifier le compte a chaque `pytest`.

```bash
pytest tests/                                  # build dry-run (prepare_live skippe)
BOURSO_LIVE_TESTS=1 pytest tests/ -k prepare_live   # check compte reel a la demande
```

Un cron quotidien (**tous les jours 20:00**) lance `src.bourso.check_cli`, qui **alerte** :
1. **build casse** — les tests dry-run echouent (sujet `BUILD CASSE`) ;
2. **connexion compte KO** — le `trade summary` reel sur le PEA echoue, p.ex.
   auth/reseau/CLI cassee (sujet `CONNEXION COMPTE KO`) ;
3. **nouveau tag azerpas** plus recent que notre tag de base — le mail inclut le
   **changelog complet** du tag (sujet `nouveau tag vX.Y.Z`), pour decider du rebase ;
4. sinon, rapport `OK`. (Statut aussi affiche sur stdout / log.)

```cron
0 20 * * * cd $DIR && ./venv/bin/python -m src.bourso.check_cli >> logs/cron_bourso_check.log 2>&1
```

```bash
python -m src.bourso.check_cli            # tests + check upstream + email
python -m src.bourso.check_cli --no-email # rapport affiche sans envoi
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

### Lire l'etat du compte (`trade summary` — patch maison)
```bash
bourso-cli trade summary --account <ACCOUNT_ID>
```
Renvoie en JSON le resume du compte trading : `cash`, `valuation`, `total`, et les
`positions` (par symbole : `quantity`, `last` prix, `amount`...). Read-only, ne passe
aucun ordre. C'est la commande sur laquelle s'appuie `src/bourso/prepare.py`.

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
20:00  cron check (check_cli.py)   →  tests dry-run + maj upstream + email
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

0 20 * * * cd /home/greg/data_local/code/MyQTMv2 && ./venv/bin/python -m src.bourso.check_cli >> logs/cron_bourso_check.log 2>&1
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
| `src/bourso/prepare.py` | Lecture etat PEA (cash/positions/cours) via `trade summary` |
| `src/bourso/execute.py` | Execution manuelle interactive (PEA ou CTO) |
| `src/bourso/list_accounts.py` | Liste tous les comptes et soldes |
| `src/bourso/check_cli.py` | Cron 20h: tests dry-run + check commits upstream + email |
| `2_bourso_cli_update.sh` | Build/install bourso-cli depuis le submodule `external/bourso-api` |

## Logs

- `logs/cron_backtest.log` — sortie du backtest du soir
- `logs/cron_pea.log` — sortie de l'execution matin
- `logs/trades.jsonl` — historique des ordres (lu par la webapp)
- `logs/cron_bourso_check.log` — sortie du check quotidien bourso-cli (20h)

## Execution manuelle

```bash
# Dry-run (voir ce qui serait fait)
python -m src.real_bourso

# Execution reelle
python -m src.real_bourso --execute

# Ordre manuel interactif
python -m src.bourso.execute pea PUST buy 4
```
