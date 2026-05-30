# energy-prices

Servizio interno (dashboard + forecasting) per i prezzi **GME** dell'energia
elettrica e del gas in Italia: prezzi aggiornati per zona + previsioni
probabilistiche del prezzo futuro.

> ⚠️ **Riforma di mercato 2025 (importante).** Dal **1° gen 2025** il PUN non è più
> il prezzo di settlement: il mercato del giorno prima (MGP) si chiude sui **7
> prezzi zonali** (NORD, CNOR, CSUD, SUD, CALA, SICI, SARD); il "PUN Index GME" è
> ora un indice di riferimento ex-post. Dal **1° ott 2025** la risoluzione è di
> **15 minuti** (96 valori/giorno) invece che oraria. Il codice gestisce entrambi
> i regimi e prevede i **prezzi zonali**, non il PUN.

## Architettura

```
ENTSO-E (primario, gratis) ─┐
GME API (ufficiale, operatore) ─┤
TTF yfinance / gas fundamentals ─┼─► ingestion ─► PostgreSQL+TimescaleDB ─► forecasting ─► dashboard (Streamlit)
                                 │                  (system-of-record)        (LEAR+LightGBM,    grafici + bande
                                 │                                             probabilistico,    di confidenza)
                                 └─ AGSI+ / ENTSOG (feature esogene)           pre-calcolato su DB)
```

- **Dati**: ENTSO-E è la fonte automatica primaria (gratuita); l'**API GME**
  ufficiale fornisce PUN Index, intraday e gas (richiede credenziali operatore).
- **Modelli**: ensemble robusto **LEAR** (`epftoolbox`) + **LightGBM** con quantili
  (forecast probabilistico). Metriche: rMAE, CRPS, pinball, coverage.
- **Storage**: Postgres+TimescaleDB in produzione; SQLite a zero-setup in locale.
- **Forecast pre-calcolati** su DB: la dashboard fa solo `SELECT` (veloce, uguale
  per tutti, backtest gratuito su ogni run archiviata).

## Quick start (locale, zero-setup, SQLite + dati demo)

```powershell
# 1. Ambiente
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# 2. Config
Copy-Item .env.example .env     # demo_mode=true di default

# 3. DB + dati demo sintetici + forecast + dashboard
energy init-db
energy seed-demo
energy forecast --market elec
energy dashboard                # http://localhost:8501
```

La dashboard parte subito con dati **sintetici realistici** (chiaramente
etichettati come demo) così la vedi funzionare prima di collegare le fonti reali.

## Dati reali

1. **ENTSO-E (gratis, ~3 giorni lavorativi)** — registrati su
   <https://transparency.entsoe.eu>, poi invia una mail a
   `transparency@entsoe.eu` (oggetto "RESTful API access"). Metti il token in
   `.env` (`ENERGY_ENTSOE_API_TOKEN`).
2. **GME (avete già accesso operatore)** — inserisci username/password API in
   `.env` (`ENERGY_GME_API_USERNAME` / `ENERGY_GME_API_PASSWORD`).
3. **(Opzionali, gratis)** chiave AGSI+ (storage gas) ed EIA (Henry Hub).

Poi:

```powershell
# (.env) ENERGY_DEMO_MODE=false
energy ingest --source all      # popola lo storico
energy forecast --market elec
energy forecast --market gas
```

## Deploy condiviso per i colleghi (Postgres + Timescale, solo locale)

Passaggio dal file SQLite (dev) al system-of-record **Postgres+TimescaleDB**,
condiviso in rete interna. **Nessun deploy cloud** — sharing via VPN.

```powershell
# 1. Config: compila i secret e cambia le password di default
Copy-Item .env.example .env
#   in .env: POSTGRES_PASSWORD=<forte>, ENERGY_DASHBOARD_PASSWORD=<condivisa>,
#            ENERGY_DEMO_MODE=false

# 2. Avvia lo stack (db Timescale + scheduler ingest + dashboard)
docker compose up -d --build

# 3. (Una tantum) migra lo storico SQLite -> Postgres
pip install -e ".[postgres]"
python scripts/migrate_sqlite_to_postgres.py `
    --source sqlite:///./data/energy_prices.db `
    --dest postgresql+psycopg2://energy:<password>@localhost:5432/energy
```

- **DB non esposto in LAN:** la porta Postgres è vincolata a `127.0.0.1`. I
  colleghi accedono solo alla **dashboard**, non al database.
- **Auth dashboard:** `ENERGY_DASHBOARD_PASSWORD` attiva un gate a password
  condivisa (difesa in profondità; per auth per-utente usare Streamlit OIDC).
- **VPN (sharing):** esponi la dashboard ai colleghi via **Tailscale** (rete
  privata, zero-config) — es. `tailscale up` sulla macchina host e condividi
  l'IP `100.x.y.z:8501`. Niente porte aperte su Internet, niente cloud.
- Lo scheduler fa il ciclo giornaliero **ingest + forecast + alert** ~13:30 CET
  (orario configurabile via `ENERGY_SCHEDULER_*`).

## Alert prezzo → canale reale

Le regole di alert (`energy alerts`) possono essere **consegnate** a un canale:

```powershell
energy alerts --notify            # valuta + invia ai canali configurati
energy scheduler --once           # giro a vuoto: ingest+forecast+alert una volta
```

- **Webhook n8n (consigliato):** imposta `ENERGY_ALERT_WEBHOOK_URL` (cloud o
  self-hosted). Gli alert vengono inviati come JSON; n8n fa il fan-out
  (email/Slack/Telegram). Senza canale configurato gira in **stub** (logga il
  payload che verrebbe inviato).
- **Email SMTP diretta:** in alternativa imposta `ENERGY_SMTP_*` +
  `ENERGY_ALERT_EMAIL_TO`.

## Struttura del progetto

```
src/energy_prices/
  config/        settings (env) + enums (mercati, zone, codici EIC)
  storage/       db.py, models.py (ORM), repositories.py  ← contratti dati
  ingestion/     entsoe_client, gme_client, ttf_client, gie_client, entsog_client,
                 demo.py (dati sintetici), scheduler.py
  features/      calendar.py, build.py (lag/rolling leak-safe)
  models/        base.py (interfaccia), baseline.py, lear.py, lgbm.py, gas_sarimax.py, ensemble.py
  forecasting/   runner.py (batch → tabella forecasts), evaluation.py (rMAE/CRPS/DM)
  dashboard/     app.py (+ pages/)
  cli.py         comandi: init-db, seed-demo, ingest, forecast, backtest, dashboard, scheduler
```

## Comandi CLI

| Comando | Cosa fa |
|---|---|
| `energy init-db` | crea le tabelle (e hypertable Timescale su Postgres) |
| `energy seed-demo` | genera dati sintetici per la demo |
| `energy ingest --source all` | scarica i dati reali (ENTSO-E/GME/TTF/fondamentali) |
| `energy backfill --from 2015-01-01` | backfill storico a blocchi (rispetta i limiti API) |
| `energy gme-inspect` | scarica un campione GME e mostra i campi reali (valida il parser) |
| `energy forecast --market elec\|gas [--zone NORD] [--calibrate]` | calcola e salva i forecast (`--calibrate` = intervalli CQR onesti) |
| `energy backtest --market elec --zone NORD [--calibrate]` | walk-forward + metriche (rMAE/CRPS/coverage) |
| `energy alerts` | valuta le regole di alert prezzo sui forecast più recenti |
| `energy dashboard` | avvia la dashboard Streamlit |
| `energy scheduler` | loop schedulato (ingest + forecast giornalieri) |

## ⚖️ Note legali (leggere prima dell'uso esteso)

I dati GME sono di proprietà GME: le Condizioni Generali consentono solo uso
"informativo e privato" e vietano uso/ridistribuzione commerciale senza consenso
scritto; richiesta l'attribuzione **"Fonte: Gestore dei Mercati Energetici S.p.A."**.
Per l'uso interno a supporto di decisioni di trading verificare che il contratto
di partecipazione/operatore lo copra. yfinance (TTF) è un proxy non ufficiale,
adatto a MVP/sanity-check, non come system-of-record per un prodotto commerciale.

## Stato

MVP in costruzione. Fonti dati e approccio verificati al 2026-05-30 — vedi
la memoria di progetto `market-data-architecture`.
