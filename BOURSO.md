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
tag upstream (actuellement **v0.5.4**) + 4 commits maison : `trade summary` CLI,
`realGL` en f64, les options d'ordre (`--order-type ATP|LIM`, `--tolerance`,
`--limit`, `--validity`) et le rejet d'une tolerance LIM negative.
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

Binaire : `~/.local/bin/bourso-cli` (v0.5.4 + patchs maison : `trade summary`, options d'ordre)

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

Le patch utilise `position=INSTANT` (position **temps reel**, qui inclut les ordres
executes non encore regles) et **non** `ACCOUNTING` (comptable, fige au reglement J+2
— qui afficherait p.ex. 4 parts au lieu de 6 le jour d'un achat). Indispensable pour
une strategie quotidienne : le nombre de parts et le cash sont a jour immediatement.

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
22:30  cron backtest (run.py QQQ)  →  outputs/qqq_strategy/signal.json (allocation, exposure, rsi14)
09:05  cron PEA (real_bourso.py)   →  lit signal, execute PUST et/ou LQQ sur Euronext
20:00  cron check (check_cli.py)   →  tests dry-run + maj upstream + email
```

- **ETF**: PUST (Amundi PEA Nasdaq-100 x1, `1rTPUST`) **et** LQQ (Amundi Nasdaq-100 2x, `1rTLQQ`), detenus ensemble. `TRADE_INSTRUMENT` n'est plus lu ; `TRADE_E_MAX` (optionnel) surcharge le plafond d'exposition (`E_MAX`=1.7 dans `strategy.py` ; `1.0` = PUST seul).
- **Comptes**: jusqu'a 4 logins Bourso (voir ci-dessous), chacun avec son PEA

## Strategie deployee : PUST + LQQ, exposition plafonnee x1.7

- Exposition cible `E = min(2 x allocation, 1.7)`, realisee **drag-minimale** : `E <= 1` → PUST = E, cash = 1−E ; `E > 1` → PUST = 2−E, LQQ = E−1, **cash = 0** (LQQ ne porte que la part > 100%, tout le cash travaille). Exposition reelle = (PUST + 2 x LQQ) / equity geree (PUST + LQQ + cash − reserve DCA ; les autres lignes du PEA sont ignorees).
- **Migration depuis l'ancien "LQQ + cash"** : automatique et immediate au premier run LIVE — le compte est levier (E > 1) avec du cash oisif (>= 5%) → restructuration **a exposition constante** (plafonnee 1.7) : vente de l'excedent de LQQ (~17% du compte, frais ~0.5% du vendu), achat de PUST avec tout le cash. Ne se redeclenche pas ensuite (plus de cash quand E > 1). Dry-run pour voir le plan : `python -m src.real_bourso --account 1`.
- **Ventes puis achats** : les achats attendent que le produit des ventes soit credite (`trade summary` est temps reel : des que l'ordre est execute). Backoff ~15 min, puis relance horaire jusqu'a 17h ; si le cash n'est toujours pas la, l'achat est ecrit dans `logs/pending_orders.json` et repris **le lendemain matin sans bande** (mail "ACHAT DIFFERE").
- **Apport de capital** : detecte le matin par ecart entre le cash lu et le cash attendu (cash de la veille ± flux des ordres ; seuil de bruit max(100 EUR, 0.5% equity, 5% du notionnel traite la veille)). L'apport est **reserve** puis libere par tranches **hebdomadaires quand le RSI14 du QQQ < 50** : tranche = 20% + 2% x (50 − RSI) (RSI 40 → 40%, RSI 30 → 60%), premiere tranche le jour de la detection, tout libere au plus tard apres 26 semaines. Le jour d'une tranche les achats se font sans bande. Etat dans `logs/capital.json` ; mail "APPORT +X EUR" et panneau "Capital a allouer detecte" dans la webapp. Un retrait reduit d'abord la reserve. **Ne pas ajouter de capital le jour de la mise en prod** (la premiere lecture initialise seulement la reference de cash).

## Multicompte

Les comptes sont declares dans `.env` par slots (`src/bourso/accounts.py`) :

```
BOURSO_ID_1="..."      BOURSO_CODE_1="..."      BOURSO_MAIL_1="proprietaire@..."
BOURSO_ID_2=""         BOURSO_CODE_2=""         BOURSO_MAIL_2=""      # vide -> ignore
BOURSO_ID_3=""         ...
BOURSO_ID_4=""         ...
```

- Un slot n'est **gere** que si `BOURSO_ID_n` ET `BOURSO_CODE_n` sont renseignes ; un slot vide n'est pas traite.
- `BOURSO_MAIL_n` = destinataire des mails de CE compte (rapport du matin, recap du soir, alerte split). Vide -> `MAILING_LIST` de `notify.py`.
- L'ancienne paire `BOURSO_ID`/`BOURSO_CODE` reste acceptee comme slot 1 si `BOURSO_ID_1` est absent.
- Chaque login Bourso a son propre PEA : l'id (hexa 32 car.) est **decouvert une fois** via `bourso-cli accounts --trading` (compte de trading nomme `PEA ...`, hors `PEA-PME`) et mis en cache dans `logs/accounts.json` (nom + id + mail, jamais d'identifiant client — lu par la webapp). Forcer un id avec `BOURSO_PEA_ID_n`. Verifier avec `python -m src.bourso.accounts`.
- Le **meme signal est replique** sur tous les comptes geres : chaque compte est lu, decide (bande de non-action sur SA propre allocation reelle) et execute a son tour. Un compte injoignable est relance toutes les heures jusqu'a 17h **sans bloquer les autres**, puis recoit un mail `CONNEXION KO` et une ligne `connection_error` dans `trades.jsonl`. Code de sortie 1 si au moins un compte a echoue.
- **Un mail par compte chaque matin** a `BOURSO_MAIL_n` : connexion (OK/ECHEC), allocation conseillee vs reelle (avant/apres), action du jour, especes/titres/total, mode LIVE/DRY-RUN. Le recap du soir (`notify --recap`) envoie de meme un mail par compte avec l'apercu de l'action du lendemain sur ce compte.
- `trades.jsonl` : champs `account` (slot) et `account_name` ; les anciennes lignes sans `account` = slot 1. `logs/last_price.json` : cles `INSTRUMENT#slot` (l'ancienne cle `INSTRUMENT` = slot 1).
- `python -m src.real_bourso --account 2` limite un run a un slot ; `python -m src.bourso.execute pea PUST buy 4 --account 2` pour un ordre manuel sur un autre compte ; `python -m src.bourso.list_accounts` liste tous les PEA geres.
- `check_cli` (cron 20h) teste la connexion de **chaque** compte et alerte `CONNEXION COMPTE KO` si l'un echoue.
- Webapp : onglet **Comptes** (une carte de statut par compte : exposition cible/reelle, repartition PUST / LQQ / cash, capital a allouer detecte, achat differe) ; Gain reel / Trades / Allocations sont affiches par compte.

## Cron

Le crontab **doit** contenir :

```cron
PATH=/home/greg/.local/bin:/usr/local/bin:/usr/bin:/bin

30 22 * * 1-5 cd /home/greg/data_local/code/MyQTMv2 && XGBOOST_DEVICE=auto ./venv/bin/python -m src.risk_off_strategy.run QQQ >> logs/cron_backtest.log 2>&1

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
  "exposure": 1.7,
  "e_max": 1.7,
  "rsi14": 64.4,
  "macro_off": false,
  "timestamp": "2026-06-18T22:35:00"
}
```

- `status`: `"running"` au debut du backtest, `"ok"` a la fin. Si crash, reste `"running"` et le matin refuse d'executer.
- `allocation`: 0.0 (cash) a 1.0 (full invest) ; `exposure` = min(2 x allocation, `e_max`) ; `rsi14` = RSI de Wilder du QQQ (DCA des apports).
- Le script du matin verifie la fraicheur du signal : age max `MAX_SIGNAL_AGE_HOURS=90h`. Assez large pour tolerer les week-ends/feries (lundi matin = signal du vendredi soir ~58h ; long week-end jeu. soir → mar. matin ~82h). Au-dela = le backtest du soir s'est arrete → refus.

## Frais et seuils

- **Achat**: 0% (ETF gratuit sur Bourso PEA)
- **Bande de non-action asymetrique en EXPOSITION** (importee de `strategy.py`, meme calibrage que le backtest net `simulate_net(e_max=1.7)`) :
  - **Achat** (gratuit) : seulement si exposition cible − reelle >= +0.50 (`BUY_THR_E` = 0.25 x 2)
  - **Vente** (0.5%) : seulement si reelle − cible >= 1.00 (`SELL_THR_E` = 0.50 x 2) → on ne DE-lève que par grands pas
  - **Force cash** : un passage a 0% (garde-fous macro / emergency) liquide TOUJOURS (les deux ETF), meme sous le seuil de vente ; une premiere entree depuis le cash total s'execute meme sous le seuil d'achat.
  - **Restructuration** (hors bande) : compte levier avec cash oisif >= 5% → composition drag-minimale a expo constante (cf. migration ci-dessus).
- **Garde-fou split** (par instrument, cles `PUST#slot` / `LQQ#slot`) : si le prix saute d'un facteur >= 1.5 (x ou /) vs la seance precedente (`logs/last_price.json`, maj a chaque run LIVE) = signature d'un split (ex: LQQ /200) ou d'une incoherence d'affichage broker -> **aucune position prise ce jour-la**, email d'alerte "SPLIT detecte", reprise a la seance suivante (reference = prix post-split). Evite d'acheter/vendre sur un prix fausse le jour du split.
- **Type d'ordre** : LIMITE avec tolerance `LIMIT_TOLERANCE_PCT`=3% (limite = cours ±3%, achat +, vente -), ventes d'abord puis achats. Tampon le gap d'ouverture -> remplissage fiable tout en bornant le prix (un ordre limite pile au cours n'avait pas rempli le 07-07 quand le cours s'est ecarte).
- **Emergency OFF**: creer `logs/emergency_off.json` avec `{"active": true}` pour forcer allocation a 0%

## Scripts

| Script | Role |
|---|---|
| `src/real_bourso.py` | Execution matin: lit signal, puis pour CHAQUE compte gere : etat PEA (PUST + LQQ + cash), apport/DCA, ventes puis achats, mail |
| `src/bourso/capital.py` | Detection d'apport (cash lu vs attendu) + DCA hebdo pilote par le RSI (`logs/capital.json`) |
| `src/bourso/accounts.py` | Slots multicompte du `.env` + decouverte/cache du PEA de chaque login |
| `src/bourso/prepare.py` | Lecture etat PEA (cash/positions/cours) via `trade summary` (`creds=` par compte) |
| `src/bourso/execute.py` | Execution manuelle interactive (PEA ou CTO, `--account N`) |
| `src/bourso/list_accounts.py` | Liste les PEA de tous les comptes geres et leurs soldes |
| `src/bourso/check_cli.py` | Cron 20h: tests dry-run + check commits upstream + email |
| `2_bourso_cli_update.sh` | Build/install bourso-cli depuis le submodule `external/bourso-api` |

## Logs

- `logs/cron_backtest.log` — sortie du backtest du soir
- `logs/cron_pea.log` — sortie de l'execution matin
- `logs/trades.jsonl` — historique des ordres, une ligne par compte et par jour (lu par la webapp)
- `logs/accounts.json` — cache des PEA decouverts par slot (nom, id, mail)
- `logs/capital.json` — cash attendu, apports detectes et DCA en cours, par slot
- `logs/pending_orders.json` — achats reportes (produit des ventes non credite avant 17h)
- `logs/last_price.json` — prix de reference du garde-fou split (`PUST#slot`, `LQQ#slot`)
- `logs/cron_bourso_check.log` — sortie du check quotidien bourso-cli (20h)

## Execution manuelle

```bash
# Dry-run (voir ce qui serait fait)
python -m src.real_bourso

# Execution reelle (tous les comptes geres)
python -m src.real_bourso --execute

# Un seul compte
python -m src.real_bourso --account 2

# Ordre manuel interactif (slot 1 par defaut, --account N pour un autre)
python -m src.bourso.execute pea PUST buy 4 --account 2

# Comptes geres + PEA decouverts
python -m src.bourso.accounts
python -m src.bourso.list_accounts
```
