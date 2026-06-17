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
| Amundi PEA Nasdaq-100 (PANX) | a chercher | FR0013412269 |
| Amundi Nasdaq-100 2x (LQQ) | a chercher | FR0010342592 |
| Amundi MSCI World (CW8) | 1rTCW8 | LU1681043599 |

Pour trouver un symbole : aller sur `boursorama.com/bourse/trackers/cours/<SYMBOLE>/`.
