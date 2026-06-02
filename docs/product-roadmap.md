# Product Roadmap — Visualizzazione, Scenari, Mobile

> **Scopo di questo documento.** È la guida operativa per costruire la prossima
> generazione di feature della dashboard (storico zonale + mappa, scenari
> what-if, layer eventi/news, notifiche e widget iOS). È scritto per essere letto
> da un'istanza Claude futura che riprende il lavoro a freddo: leggi prima
> "Stato attuale" e "Architettura target", poi esegui le fasi in ordine. Ogni
> fase ha _obiettivo → step concreti con percorsi file → dipendenze → criteri di
> accettazione → trappole_. Aggiorna le checkbox `[ ]→[x]` man mano che procedi e
> tieni allineata la sezione "Stato attuale".

Ultimo aggiornamento: 2026-06-02.

---

## 1. Stato attuale (snapshot)

**Stack**: Python 3.11 · SQLAlchemy 2 + Timescale/SQLite · Streamlit + Plotly ·
APScheduler (job giornaliero 13:30 Europe/Rome) · n8n per il fan-out alert.

**Dati nel DB** (`price_observations`, `exogenous_observations`, `forecasts`):
- PUN + 7 zone (`NORD, CNOR, CSUD, SUD, CALA, SICI, SARD`) — orario 2021-01→2025-09-30, 15-min 2025-10-01→oggi.
- Gas day-ahead GME (daily) + TTF (yfinance, daily, dal 2020).
- Forecast probabilistici (quantili q0.1..q0.9) per `(market, zone, model_name, run_at)`.

**Moduli chiave da riusare** (NON reinventare):
- `storage/repositories.py`: `PriceRepository.get_prices(market, zone, start, end, source)`,
  `.distinct_zones(market)`, `.latest_delivery(market, zone)`;
  `ForecastRepository.get_forecasts(market, zone, model_name, run_at, latest)` → pivot wide quantili;
  `ExogenousRepository.get_series(name, zone, start, end)`.
- `forecasting/runner.py`: `run_forecasts(...)` — pipeline forecast completo (carica storia, esogene, fitta, persiste).
- `ingestion/scheduler.py`: `run_once(notify=...)` — un ciclo ingest+forecast+alert; `start_scheduler()` blocking.
- `notifications.py`: `build_payload(alerts)`, `_post_webhook` (n8n), `_send_email` (SMTP).
- `dashboard/app.py`: `load_prices`, `load_forecast`, `_zone_options`, `_section`, `_style_fig`,
  `_observed_chart`, `_forecast_chart`, `_sidebar`, `main`. Tema dark "energy desk" già impostato.
- `config/settings.py`: tutte le config via env `ENERGY_*` (pydantic-settings, legge `.env`).

**Vincoli appresi noti (impari da qui, non a tue spese):**
- **GME rate-limit**: ~1 richiesta ogni 30-40s sostenuti. La retry-logic a 6 tentativi
  martella durante il cooldown ed è controproducente per backfill lunghi. Per pull storici:
  chunk grandi (orario regge 90gg/req), `--skip-gas`, pausa lunga tra richieste.
- **Risoluzione mista**: orario (≤2025-09-30) e 15-min (≥2025-10-01) NON vanno mischiati nella
  stessa serie di backtest (`walk_forward` slicea posizionalmente). Usa `--start/--end`.
- **Zone post-riforma 2021**: CALA si è staccata da SUD. Il GeoJSON deve riflettere le 7 zone
  di **mercato** (bidding zones), non le regioni amministrative.

---

## 2. Architettura target

```
                        +-----------------------------+
                        |   DB (Timescale / SQLite)    |
                        | prices . exog . forecasts    |
                        +--------------+--------------+
                                       | repositories.py (UNICO accesso dati)
            +--------------------------+---------------------------+
            |                          |                            |
   +--------v--------+      +----------v----------+      +----------v---------+
   | Streamlit dash  |      |  FastAPI read-API   |      | Scheduler 13:30    |
   | (desktop, ricca)|      | /latest /forecast   |      | ingest+forecast+   |
   |                 |      | /zonal /scenario    |      | alert -> n8n/push  |
   +-----------------+      +----------+----------+      +--------------------+
                                       | HTTPS JSON
                            +----------v----------+
                            | Scriptable widget   |  + push (ntfy/Pushover/Telegram)
                            | (iPhone home screen)|
                            +---------------------+
```

**Decisione strutturale chiave**: introdurre una **read-API FastAPI** sopra i
repository esistenti. È il backbone che sblocca widget, notifiche e (volendo) una
futura SPA. La dashboard Streamlit resta per l'uso desktop ricco. Nessuna logica
di accesso dati duplicata: l'API chiama gli stessi `*Repository`.

---

## 3. Fase 0 — Prerequisiti e decisioni

- [ ] **GeoJSON zone di mercato**: procurare/derivare il poligono delle 7 bidding zones
  (NORD, CNOR, CSUD, SUD, CALA, SICI, SARD). Sorgenti possibili: dataset Terna/GME,
  oppure aggregare GeoJSON regioni ISTAT in macro-zone. Salvare in
  `src/energy_prices/dashboard/assets/zones_it.geojson` con proprietà `zone` = codice zona.
  ATTENZIONE: validare che ogni regione sia mappata alla zona giusta post-2021.
- [ ] **Scelta canale push**: ntfy (self-host/gratis, zero account) vs Pushover (1 euro una tantum,
  affidabile) vs Telegram bot (gratis, richiede chat_id). _Raccomandato: ntfy_ per semplicità.
- [ ] Confermare con l'utente: l'API gira sullo stesso host della dashboard? Auth (token statico ok per uso personale)?

---

## 4. Fase 1 — Dashboard ricca: storico zonale + mappa Italia

**Obiettivo**: vedere storico PUN/PSV, confronto zonale, e mappa interattiva con
prezzo live/medio per zona.

- [ ] **1.1 Vista multi-zona** in `dashboard/app.py`: nuovo `_section("Confronto zonale")`
  con multiselect zone (riusa `_zone_options()`), overlay delle serie via `load_prices`
  per ogni zona, palette per zona. Aggiungere toggle PUN vs zona.
- [ ] **1.2 Spread zonali**: grafico dello spread `zona - PUN` (mostra dove l'energia costa di piu).
- [ ] **1.3 Mappa choropleth**: nuova funzione `_zonal_map(snapshot: dict[zone,float], accent)` ->
  `plotly.express.choropleth_mapbox` o `go.Choroplethmapbox` con il GeoJSON di Fase 0.
  Colore = ultimo prezzo (o media giornaliera) per zona. Hover = zona + EUR/MWh + delta vs PUN.
  Snapshot dati: per ogni zona `PriceRepository.latest_delivery` + valore a quel timestamp.
- [ ] **1.4 PSV/gas**: sezione storico gas day-ahead + overlay TTF (riusa `_render_ttf_overlay`).
- [ ] **1.5 Cache**: avvolgere i loader pesanti con `@st.cache_data(ttl=...)` (gia parzialmente fatto, verificare).

**Dipendenze nuove**: nessuna obbligatoria (Plotly basta per choropleth_mapbox; mapbox
token NON necessario con stile "carto-positron").

**Criteri di accettazione**: la mappa colora correttamente le 7 zone con il prezzo corrente;
selezione multi-zona disegna le serie sovrapposte; spread calcolato sul timestamp comune.

**Trappole**: allineare i timestamp tra zone (i 15-min/orari devono combaciare prima dello spread —
usare join sull'indice, non posizionale). La zona "PUN" e uno pseudo-codice nazionale, non un poligono:
escluderla dalla mappa.

---

## 5. Fase 2 — Read-API (FastAPI) + notifiche push

**Obiettivo**: un servizio HTTP leggero che espone i dati per widget/notifiche, e una
notifica push quando esce il nuovo prezzo/forecast.

- [ ] **2.1 Modulo API**: nuovo package `src/energy_prices/api/` con `app.py` (FastAPI) e `schemas.py` (pydantic).
  Endpoints minimi:
  - `GET /health`
  - `GET /latest?market=elec&zone=PUN` -> ultimo prezzo + timestamp + freshness.
  - `GET /forecast?market=elec&zone=PUN&model=...` -> quantili prossimo giorno (riusa `get_forecasts`).
  - `GET /zonal` -> snapshot {zona: prezzo} per la mappa/widget.
  - `GET /history?market&zone&start&end` -> serie (downsampled, cap punti).
  Ogni handler apre `session_scope()` e usa i repository. Auth: header `X-API-Key` confrontato con `settings.api_key`.
- [ ] **2.2 Config**: aggiungere a `settings.py` i campi `api_key: str | None`, `push_*` (vedi 2.4).
- [ ] **2.3 CLI**: comando `energy api` in `cli.py` che lancia `uvicorn` (come fa `dashboard`).
- [ ] **2.4 Push channel**: nuova funzione in `notifications.py`, es. `_push_ntfy(payload, settings)` (POST a `https://ntfy.sh/<topic>`).
  Integrarla nel fan-out alert accanto a webhook/email. Config: `push_provider`, `push_topic`/`push_token`.
- [ ] **2.5 Trigger "nuovo prezzo"**: nello `scheduler.run_once`, dopo l'ingest, se `latest_delivery` e
  avanzato rispetto all'ultimo notificato, inviare push "Nuovo PUN pubblicato: X EUR/MWh". Persistere
  l'ultimo timestamp notificato (tabella `ingestion_runs` o un piccolo state file/kv) per evitare doppioni.
- [ ] **2.6 Docker**: aggiungere il servizio `api` a `docker-compose.yml` (porta dedicata).

**Dipendenze nuove**: `fastapi`, `uvicorn[standard]`. (httpx per i test API.)

**Criteri di accettazione**: `GET /latest` risponde con l'ultimo prezzo reale; una nuova ingest
genera esattamente UNA push; l'API rifiuta richieste senza `X-API-Key` corretta.

**Trappole**: NON duplicare query SQL nell'API — sempre via repository. Timezone: l'API restituisce
UTC ISO8601; la conversione a Europe/Rome e responsabilita del client. Rate/timeout: la dashboard e
il widget devono tollerare l'API spenta (fallback leggibile).

---

## 6. Fase 3 — Scenari what-if sul forecast

**Obiettivo**: l'utente sposta un driver (TTF/gas, domanda, eolico+solare) e vede come
reagisce la previsione — valore decisionale da trader.

- [ ] **3.1 Forecast parametrico**: in `forecasting/runner.py` aggiungere un percorso che accetta
  esogene "override" (shift/scala su `load_forecast`, `wind_solar_forecast`, prezzo gas) prima del `fit_predict`.
  NON toccare il path di produzione: nuova funzione `run_scenario(...)` che ritorna i quantili senza persistere.
- [ ] **3.2 UI scenari** in `dashboard/app.py`: `_section("Scenari")` con slider (es. "TTF +20%",
  "Domanda -5%", "Eolico+Solare +30%") -> chiama `run_scenario` -> sovrappone la curva scenario al baseline.
- [ ] **3.3 (opz.) Endpoint** `POST /scenario` nell'API per esporre lo stesso calcolo.

**Criteri di accettazione**: muovendo lo slider TTF la curva gas/PSV si sposta in modo monotono e
sensato; il baseline resta visibile per confronto.

**Trappole**: gli scenari sono _condizionali_, non probabilita reali — etichettarli chiaramente come
"simulazione, non previsione". Validare che gli override restino nel dominio plausibile dei modelli.

---

## 7. Fase 4 — Layer eventi geopolitici / news (versione onesta)

**Obiettivo**: contesto leggibile sui grafici, SENZA claim causali fasulli.

- [ ] **4.1 Tabella eventi**: nuova tabella `market_events(ts, title, category, severity, url, source)`
  in `storage/models.py` + repository. Seed manuale con eventi storici chiave (24/02/2022 invasione
  Ucraina, stop Nord Stream, picchi TTF, ondate di freddo).
- [ ] **4.2 Annotazioni sul grafico**: in `_observed_chart`/`_forecast_chart` aggiungere marker/`add_vline`
  con hover = titolo evento. Toggle on/off in sidebar.
- [ ] **4.3 (opz.) News digest LLM**: job che pesca news energy (RSS/API) e produce un riassunto giornaliero
  ("cosa muove i mercati oggi") mostrato in un pannello. Esplicitamente _qualitativo_.
- [ ] **4.4 NON fare**: nessun modello "news->impatto prezzo" come feature predittiva senza validazione
  rigorosa (rischio overfitting/spurio). Se richiesto, trattarlo come progetto di ricerca separato.

**Criteri di accettazione**: gli eventi appaiono come annotazioni allineate alla data corretta; il
digest (se fatto) e chiaramente marcato come contesto non predittivo.

---

## 8. Fase 5 — Widget iPhone (Scriptable)

**Obiettivo**: prezzo + forecast in home screen + push, senza App Store/Xcode.

- [ ] **5.1 Script Scriptable** (JS): file `mobile/energy-widget.js` (nel repo come riferimento) che fa
  `Request` a `GET /latest` e `/forecast`, e disegna un `ListWidget` (prezzo corrente, delta%, sparkline forecast).
  Config: URL API + `X-API-Key` salvati nei parametri del widget.
- [ ] **5.2 Notifiche**: il push di Fase 2 arriva gia sull'iPhone (ntfy ha app iOS, o Telegram/Pushover).
- [ ] **5.3 Documentare** in `mobile/README.md` i passi di installazione (incolla script in Scriptable,
  aggiungi widget alla home, imposta URL/chiave).

**Criteri di accettazione**: il widget mostra il prezzo reale aggiornato; tocco -> apre la dashboard.

**Trappole**: iOS non consente un "vero" widget nativo senza app; Scriptable e la via pragmatica per
uso personale. Se in futuro serve distribuzione, valutare una PWA o app nativa (progetto a se).

---

## 9. Cross-cutting

- **Test**: ogni fase aggiunge test in `tests/` (API con `TestClient`/httpx; map/scenari con unit sui data-builder).
  Mantenere la CI verde: `pytest -q` + ruff + mypy come da pipeline esistente.
- **Config**: ogni nuovo segreto/flag va in `settings.py` con default sicuro e documentato in `.env.example`.
- **File < 500 righe** (regola repo): se `app.py` cresce troppo, splittare in `dashboard/sections/*.py`.
- **Sicurezza**: API key obbligatoria; mai loggare chiavi/URL webhook completi (vedi `_redact_url`).
- **Niente duplicazione dati-access**: API e dashboard passano SEMPRE dai repository.

## 10. Ordine consigliato di esecuzione

`Fase 1 (dashboard+mappa)` -> `Fase 2 (API+push)` -> `Fase 5 (widget, sblocca subito valore mobile)` ->
`Fase 3 (scenari)` -> `Fase 4 (eventi/news)`.
Razionale: 1 da valore visivo immediato sui dati gia presenti; 2 e il backbone; 5 e economico una
volta che c'e l'API; 3 e 4 sono incrementi di valore analitico.

## 11. Domande aperte per l'utente

- Host/deploy dell'API: stesso server della dashboard? Esposta su internet o solo VPN/locale?
- Canale push preferito (ntfy / Pushover / Telegram)?
- GeoJSON zone: ce l'hai gia da Terna/GME o lo derivo dalle regioni ISTAT?
- Le news: ti basta il layer di annotazioni manuali + digest, o vuoi una pipeline news automatica?
