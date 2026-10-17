# Price Tracker

Watches Google Shopping twice a day for two hardware deals, filters out anything that doesn't
meet the spec or comes from a sketchy store, and emails **adaptabyte@gmail.com** when something
matches. A web UI shows every search, every listing found, and exactly why each one was accepted
or rejected. All the rules can be changed from the Settings page.

| Tracker | Accepts | Rejects |
|---|---|---|
| **64GB DDR4 RAM** | DDR4 RDIMM / LRDIMM / UDIMM, **64GB per stick**, sold as 1 stick or a 2-stick kit, DDR4-2666 or faster (DDR4-3200+ tagged *preferred*), **under $300 per stick** including shipping | 2x32GB / 4x16GB / 8x8GB kits, SO-DIMM / laptop memory, DDR3/DDR5, slower than 2666, speed not stated, 4+ stick lots, for-parts / untested |
| **AMD Threadripper PRO 3975WX** | The 3975WX (or its AMD part number), **under $500** including shipping | Listings that say vendor locked / PSB, Lenovo P620 pulls that don't say unlocked, for parts / broken / bent pins / untested, coolers and other accessories, CPU + motherboard bundles, other models |

Both trackers only accept listings from **trusted merchants**: eBay, Walmart, Best Buy, Newegg,
Poshmark, B&H, Amazon, Micro Center, Adorama, CDW, Mercari, Back Market, Crucial, Kingston,
Supermicro, Dell, HP, Lenovo and about 15 other well-known US retailers. Sellers that copy a trusted
store's name but link to a different domain are rejected as likely impostors.

## How it works

```
 scheduler (08:00 + 20:00) ──► Google Shopping search via SerpApi or Serper (one request per query)
                                   │
                                   ▼
                     every result ──► tracker rules ──► merchant check ──► accepted / rejected + reasons
                                   │                                              │
                                   ▼                                              ▼
                          SQLite (runs, queries,                  new matches and price drops ──► one email
                          listings, observations)                        (never the same listing twice)
                                   │
                                   ▼
                             web UI on :8000
```

Google has no public Shopping API and blocks scrapers, so searches go through a SERP API service
that runs the real Google Shopping search and returns the results as JSON. Two are supported:

- **[SerpApi](https://serpapi.com)**: the default provider.
- **[Serper](https://serper.dev)**: an alternative.

A third option, **Demo**, uses built-in sample listings, so you can try the UI with no API key.

The default settings send 6 queries per scan (4 RAM, 2 CPU). At 2 scans a day that's about
**360 searches a month**. The Settings page shows the estimate for your own settings. Make sure
your provider's plan covers it, or remove some queries.

## Setup

You need:

1. **An API key** from SerpApi or Serper.
2. **A Gmail App Password** so the app can send email. Your normal Gmail password won't work.
   Turn on 2-Step Verification, then go to Google Account → Security → 2-Step Verification →
   App passwords.
3. **An always-on machine** to run it: a home server, NAS, Raspberry Pi or small VPS. Scans only
   run while the app is running.

### With Docker (recommended)

```bash
git clone https://github.com/adaptabyte/test1.git pricetracker && cd pricetracker
cp .env.example .env        # fill in SERPAPI_API_KEY, SMTP_USERNAME, SMTP_PASSWORD, TRACKER_UI_PASSWORD
docker compose up -d --build
```

Open `http://<that-machine>:8000`. If you set `TRACKER_UI_PASSWORD`, log in as `admin`. Then:

1. **Settings → Email → Send test email** to check that email works.
2. **Settings → Schedule**: set your time zone. The default is America/New_York, with scans at
   08:00 and 20:00.
3. **Dashboard → Run scan now** to run the first scan without waiting for the schedule.

Data is kept in the `tracker-data` Docker volume.

### Without Docker

Python 3.11 or newer:

```bash
make install                 # creates .venv and installs dependencies
export SERPAPI_API_KEY=...   # or paste the key in Settings later
make run                     # web UI + scheduler on http://localhost:8000
```

The database goes in `./data/tracker.db`.

### Using cron instead of the built-in schedule

Turn off **Settings → Schedule → Run scans automatically**, then run a scan from cron or any
other scheduler:

```bash
python -m pricetracker scan            # exits 1 if the scan failed
python -m pricetracker scan --no-email
```

Use either the built-in schedule or cron, not both. If two scans overlap, the second one is
refused.

## The web UI

| Page | What it's for |
|---|---|
| **Dashboard** | Last and next scan, current matches, a summary per tracker, a setup checklist and a **Run scan now** button |
| **Scans** | Every scan with its trigger, status, result counts and email outcome |
| **Scan detail** | Each query sent (with the exact request), how long it took and what came back. Every listing with its decision, reasons and warnings, filterable by tracker and by rejection reason. The full scan log |
| **Replay** | Re-checks a past scan's stored results against your *current* settings and shows which decisions would change. Uses no API queries and sends no email. Use it to test a settings change |
| **Listings** | Every distinct listing ever seen. Search, filter (accepted / rejected / emailed) and sort by price |
| **Listing detail** | Price and decision history across scans, parsed specs, and the raw JSON from the search API |
| **Merchants** | Every seller seen, whether it's trusted, and how many listings were accepted or rejected. One-click **Trust** / **Block** |
| **Emails** | Every email sent, skipped or failed, with the full message body |
| **Rule tester** | Paste any listing title, price and seller to see exactly how the rules judge it |
| **Settings** | Search provider, schedule, email, both trackers' rules and search queries, trusted and blocked merchants, data retention |

## Matching details

**RAM.** The rules read the listing title plus Google's attribute chips:

- **Stick configuration.** Read from `2x64GB`, `64GB x 2`, `(2 x 64 GB)`, `2pcs 64GB` and
  `lot of 2`. Known part numbers count too, e.g. Samsung `M393A8G40…` and `M386A8K40…`, SK hynix
  `HMAA8GR7…` and Micron `MTA36ASF8G72PZ…` are all 64GB modules. A title like "64GB Kit" with no
  breakdown is rejected, because it's usually 2x32GB. Rank notation such as `2Rx4` is not
  mistaken for a kit.
- **Speed.** Read from `3200MHz`, `DDR4-2933`, `PC4-2666V`, `PC4-25600`, or a part number's
  speed code (`-CWE`, `-XN`, `-3G2`, …). If several speeds are listed, the highest one is used.
- **Type.** Read from RDIMM/LRDIMM/UDIMM, registered, load-reduced, unbuffered, or the JEDEC
  label letter (`-RB2` means RDIMM). SO-DIMM, laptop, 260-pin and iMac listings are always
  rejected.
- **Price.** The total including shipping, divided by the number of sticks. It must be *below*
  the limit. Very low prices (under $40 a stick) are rejected as likely scams.

**CPU vendor lock.** Threadripper PRO chips that were used in a Lenovo ThinkStation P620 are
usually permanently PSB-locked to Lenovo boards.

- **Rejected:** listings that say *vendor locked*, *Lenovo locked*, *locked to …* or mention
  *PSB*, and Lenovo/P620 pulls that don't explicitly say they're unlocked.
- **Accepted:** listings that say *not vendor locked*, *non-locked* or *no PSB*, and new retail
  boxed chips.
- **Lock not mentioned** (most listings): accepted with a warning in the email by default. Switch
  **Vendor lock** to *Strict* in Settings to reject them instead.

**Condition.** These are always rejected: for parts, not working, broken, damaged, defective,
bent or missing pins, no POST, dead. Untested and as-is are rejected by default (you can change
that in Settings). Phrases that deny a defect, like "no bent pins" or "tested working, not for
parts", are not counted against the listing.

**Emails.** You get at most one email per scan, listing every new match:

- A listing that was already emailed is sent again only if its price drops by at least 5%
  (adjustable).
- The same seller relisting the same item under a new ID isn't sent again.
- If an email fails, those matches go in the next scan's email.

## Configuration

Everything is on the Settings page and stored in SQLite. These environment variables are also
read:

| Variable | Purpose |
|---|---|
| `SERPAPI_API_KEY`, `SERPER_API_KEY` | Search API keys. Used when the Settings field is blank |
| `SMTP_USERNAME`, `SMTP_PASSWORD` | Email login. Used when the Settings field is blank |
| `TRACKER_UI_PASSWORD`, `TRACKER_UI_USERNAME` | Turn on HTTP Basic login for the web UI (username defaults to `admin`) |
| `TRACKER_DATA_DIR` / `TRACKER_DB_PATH` | Where the database lives (default `./data/tracker.db`; `/data` in Docker) |
| `TRACKER_HOST`, `TRACKER_PORT` | Listen address (default `0.0.0.0:8000`) |

The UI shows secrets as "saved" or "from environment" and never displays their values. If you
expose the UI beyond your home network, set `TRACKER_UI_PASSWORD` and put it behind HTTPS.

## Limitations

- The rules only see what Google Shopping shows: title, price, seller, shipping and attribute
  chips. They don't see the full listing description, so a seller who mentions the vendor lock or
  a defect only in the description won't be caught. Always read the listing before buying.
- Results cover whatever Google Shopping has indexed. eBay and Walmart are well covered; some
  stores never appear.
- Google sometimes hides the merchant link behind a Google product page. In that case the seller
  name alone decides whether it's trusted.

## Development

```bash
make install   # venv + dev dependencies
make test      # pytest
make lint      # ruff format --check + ruff check
make fmt       # auto-format
```

```
pricetracker/
  rules/ram.py, rules/cpu.py   per-tracker matching rules (pure functions, heavily tested)
  rules/common.py              condition, price and keyword checks shared by both trackers
  merchants.py                 trusted / blocked merchant matching
  providers.py                 SerpApi, Serper and Demo search clients
  scanner.py                   one scan: search → evaluate → store → email
  notifier.py                  email building and SMTP
  scheduler.py                 twice-daily schedule (APScheduler)
  db.py                        SQLite schema and queries
  settings.py                  defaults, validation, Settings form parsing
  web.py, templates/, static/  FastAPI web UI
tests/                         rule tables, provider parsing, scan/email flow, web routes
```
