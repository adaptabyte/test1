# Price Tracker

Searches for two hardware deals twice a day, filters out anything that doesn't meet the spec or
comes from a sketchy store, and emails **adaptabyte@gmail.com** when something matches. A web UI
shows every search, every listing found, and exactly why each one was accepted or rejected. All
the rules can be changed from the Settings page.

Everything runs free, on your own machine: searches go through a bundled, private
[SearXNG](https://docs.searxng.org) instance, with eBay's free official API as an optional extra,
and email goes out through [Resend](https://resend.com)'s free tier.

| Tracker | Accepts | Rejects |
|---|---|---|
| **64GB DDR4 RAM** | DDR4 RDIMM / LRDIMM / UDIMM, **64GB per stick**, sold as 1 stick or a 2-stick kit, DDR4-2666 or faster (DDR4-3200+ tagged *preferred*), **under $300 per stick** including shipping | 2x32GB / 4x16GB / 8x8GB kits, SO-DIMM / laptop memory, DDR3/DDR5, slower than 2666, speed not stated, 4+ stick lots, for-parts / untested |
| **AMD Threadripper PRO 3975WX** | The 3975WX (or its AMD part number), **under $500** including shipping | Listings that say vendor locked / PSB, Lenovo P620 pulls that don't say unlocked, for parts / broken / bent pins / untested, coolers and other accessories, CPU + motherboard bundles, other models |

Every tracker also rejects:

- **Untrusted stores.** Listings must come from eBay, Walmart, Best Buy, Newegg, Poshmark, B&H,
  Amazon, Micro Center, Adorama, CDW, Mercari, Back Market, Crucial, Kingston, Supermicro, Dell,
  HP, Lenovo, or about 15 other well-known US retailers. A seller named like a trusted store but
  linking to a different site is rejected as a likely impostor.
- **Low-feedback eBay sellers**, when seller feedback is known (the eBay API reports it). The
  defaults are below 97% positive, or fewer than 10 ratings.
- **Out-of-stock and ended listings.**

## How it works

```
 scheduler (08:00 + 20:00) ─► each tracker query goes to each search source
                                  │
       ┌──────────────────────────┼─────────────────────────────┬───────────────────────┐
       ▼                          ▼                             ▼                       ▼
 SearXNG eBay engine     SearXNG web search             eBay Browse API          SerpApi / Serper
 (eBay results with      (DuckDuckGo, Brave, Bing,      (optional, free key)     (optional, paid:
  prices)                 Google; trusted stores only)                            Google Shopping)
                                  │
                                  ▼
                     open each listing page and read its price
                     from the store's schema.org product data
       └──────────────────────────┴───────────────┬─────────────┴───────────────────────┘
                                                  ▼
             same item from several sources = one listing ─► rules ─► accepted / rejected + reasons
                                                  │                          │
                                                  ▼                          ▼
                                 SQLite: scans, searches, listings     new matches and price drops
                                 and every observation                 ─► one email (Resend or SMTP)
                                                  │
                                                  ▼
                                            web UI on :8000
```

### Search sources

| Source | Cost | Covers | Notes |
|---|---|---|---|
| **SearXNG · eBay** | free | eBay | Uses SearXNG's eBay engine, which reads eBay's search page. The bundled config fixes the engine's search URL and searches Buy It Now listings only. It can break when eBay redesigns its site. |
| **SearXNG · web** | free | All trusted stores | Web search restricted to the trusted stores. Each listing page is opened to read its price, shipping, condition and stock status. Stores with strong bot protection (often Amazon and Walmart) may not give a price; those listings are rejected with that reason. |
| **eBay Browse API** | free key | eBay | Recommended. eBay's official API returns exact prices, shipping, condition and seller feedback. It allows up to 5,000 calls a day. |
| **SerpApi / Serper** | paid | Google Shopping | Optional. Real Google Shopping results from all stores. |

Google Shopping itself has no free API, and SearXNG has no Google Shopping engine, which is why
the free setup searches eBay and the stores directly instead.

## Setup

On your always-on machine you need Docker with the Compose plugin.

```bash
git clone https://github.com/adaptabyte/test1.git pricetracker && cd pricetracker
cp .env.example .env     # fill in RESEND_API_KEY (and optionally the eBay keys, a UI password)
docker compose up -d --build
```

Open `http://<that-machine>:8000`. If you set `TRACKER_UI_PASSWORD`, log in as `admin`. Then:

1. **Email.** Sign up at [resend.com](https://resend.com) **using adaptabyte@gmail.com** and
   create an API key. Put it in `.env` as `RESEND_API_KEY` or paste it in **Settings → Email**,
   then click **Send test email**.
   - Without your own domain, Resend's shared sender (`onboarding@resend.dev`) can only email the
     address the Resend account was created with. That's all the tracker needs.
   - To email other addresses, verify a domain in Resend and set the From address.
2. **Check the searches.** In **Settings → Search sources**, click **Test SearXNG · eBay** and
   **Test SearXNG · web search**. Each one runs one sample search.
3. **Optional but recommended: the eBay API.**
   1. Create a free account at [developer.ebay.com](https://developer.ebay.com).
   2. Create a **Production** keyset.
   3. eBay asks about marketplace account deletion notifications; choose the exemption for apps
      that don't store eBay user data.
   4. Paste the App ID (client ID) and Cert ID (client secret) into the settings, or set
      `EBAY_CLIENT_ID` / `EBAY_CLIENT_SECRET` in `.env`.
   5. Tick **Use the eBay API**.
4. **Schedule.** In **Settings → Schedule**, set your time zone. The default is America/New_York,
   with scans at 08:00 and 20:00.
5. **Dashboard → Run scan now** to run the first scan without waiting for the schedule.

The stack has two containers: `pricetracker` (web UI and scheduler) and `pricetracker-searxng`.
SearXNG isn't exposed to your network. To use it in a browser too, uncomment its `ports:` lines
in `docker-compose.yml`. Data lives in the `tracker-data` Docker volume.

Update with `git pull && docker compose pull searxng && docker compose up -d --build`.

### Without Docker

Python 3.11 or newer, plus a SearXNG instance with JSON output enabled. You can reuse
[searxng/settings.yml](searxng/settings.yml) for that instance.

```bash
make install
export SEARXNG_URL=http://localhost:8888 RESEND_API_KEY=re_...
make run                     # web UI + scheduler on http://localhost:8000
```

### Using cron instead of the built-in schedule

Turn off **Settings → Schedule → Run scans automatically**, then run scans from cron with
`python -m pricetracker scan` (or `docker exec pricetracker python -m pricetracker scan`).
The command exits with status 1 if the scan failed.

## The web UI

| Page | What it's for |
|---|---|
| **Dashboard** | Last and next scan, current matches, a summary per tracker, a setup checklist and a **Run scan now** button |
| **Scans** | Every scan with its trigger, sources, status, result counts and email outcome |
| **Scan detail** | Each search sent: source, exact request, time taken, results, which engines failed. Every listing with its decision, reasons and warnings, filterable by tracker and by rejection reason. The full scan log |
| **Replay** | Re-checks a past scan's stored results against your *current* settings and shows which decisions would change. No searches are run and no email is sent |
| **Listings** | Every distinct listing ever seen. Search, filter (accepted / rejected / emailed) and sort by price |
| **Listing detail** | Price and decision history, parsed specs, and the raw data from the source, including what was read from the listing page |
| **Merchants** | Every seller and site seen, whether it's trusted, and its accepted / rejected counts. One-click **Trust** / **Block** |
| **Emails** | Every email sent, skipped or failed, with the full message body |
| **Rule tester** | Paste any listing title, price and seller to see exactly how the rules judge it |
| **Settings** | Search sources (with test buttons), schedule, email, both trackers' rules and queries, trusted and blocked merchants, seller-feedback limits, data retention |

## Matching details

**RAM.** The rules read the listing title:

- **Stick configuration.** Read from `2x64GB`, `64GB x 2`, `(2 x 64 GB)`, `2pcs 64GB` and
  `lot of 2`. Known part numbers count too, e.g. Samsung `M393A8G40…` / `M386A8K40…`, SK hynix
  `HMAA8GR7…`, Micron `MTA36ASF8G72PZ…`. A title like "64GB Kit" with no breakdown is rejected,
  because it's usually 2x32GB. Rank notation such as `2Rx4` is not mistaken for a kit.
- **Speed.** Read from `3200MHz`, `DDR4-2933`, `PC4-2666V`, `PC4-25600`, or a part number's speed
  code (`-CWE`, `-XN`, `-3G2`, …).
- **Type.** Read from RDIMM/LRDIMM/UDIMM, registered, load-reduced, unbuffered, or the JEDEC label
  letter (`-RB2` means RDIMM). SO-DIMM and laptop listings are always rejected.
- **Price.** The total including shipping, divided by the number of sticks. It must be *below*
  the limit.

**CPU vendor lock.** Threadripper PRO chips that were used in a Lenovo ThinkStation P620 are
usually permanently PSB-locked to Lenovo boards.

- **Rejected:** listings that say *vendor locked* / *Lenovo locked* / *PSB*, and Lenovo/P620
  pulls that don't say they're unlocked.
- **Accepted:** listings that say *not vendor locked* (or similar), and new retail chips.
- **Lock not mentioned:** accepted with a warning in the email by default. Switch **Vendor lock**
  to *Strict* to reject them instead.

**Condition.** These are always rejected: for parts, not working, broken, damaged, bent pins, no
POST, and eBay's "For parts or not working" condition. Untested and as-is are rejected by default.
Phrases like "no bent pins" or "not for parts" are not counted against the listing.

**Emails.** You get at most one email per scan, listing every new match:

- A listing already emailed is sent again only if its price drops at least 5% (adjustable).
- The same eBay item found by several sources, or relisted under a new ID, isn't sent twice.
- If an email fails, the next scan retries it.

## Configuration

Everything is on the Settings page and stored in SQLite. These environment variables (for
example in `.env`) are used when the matching Settings field is blank:

| Variable | Purpose |
|---|---|
| `RESEND_API_KEY` | Resend key for match emails |
| `EBAY_CLIENT_ID`, `EBAY_CLIENT_SECRET` | eBay Browse API keyset |
| `SEARXNG_URL` | SearXNG address (Docker Compose sets `http://searxng:8080`) |
| `SEARXNG_SECRET` | Secret for the bundled SearXNG instance |
| `TRACKER_UI_PASSWORD`, `TRACKER_UI_USERNAME` | Turn on HTTP Basic login for the web UI (username defaults to `admin`) |
| `SERPAPI_API_KEY`, `SERPER_API_KEY` | Optional paid Google Shopping sources |
| `SMTP_USERNAME`, `SMTP_PASSWORD` | Only if you send email over SMTP instead of Resend |
| `TRACKER_DATA_DIR` / `TRACKER_DB_PATH`, `TRACKER_HOST`, `TRACKER_PORT` | Storage and listen address |

The UI never displays secret values. If the UI is reachable from outside your home network, set
`TRACKER_UI_PASSWORD` and put it behind HTTPS.

## Limitations

- **Engine reliability.** SearXNG's upstream engines (DuckDuckGo, Brave, Bing, Google, eBay)
  sometimes block automated searches or change their pages. Each search records which engines
  failed, and the others still run. The eBay API is the dependable eBay source.
- **Titles only.** The rules see the title, price, seller, shipping, condition and stock status.
  They don't see the full listing description, so a seller who mentions a vendor lock or a defect
  only in the description won't be caught. Always read the listing before buying.
- **Bot-protected stores.** Some stores block page reads; their listings are rejected as "No
  price" with the reason shown.

## Development

```bash
make install   # venv + dev dependencies
make test      # pytest
make lint      # ruff format --check + ruff check
make fmt       # auto-format
```

```
pricetracker/
  rules/                 matching rules per tracker, plus shared condition / price / listing checks
  merchants.py           trusted / blocked merchant matching
  sources/               searxng.py, ebay.py (Browse API), pages.py (listing-page reader),
                         google_shopping.py (SerpApi, Serper), demo.py
  scanner.py             one scan: search every source → evaluate → store → email
  notifier.py            email building; Resend and SMTP delivery
  scheduler.py           twice-daily schedule (APScheduler)
  db.py, settings.py     SQLite storage; defaults, validation, Settings form parsing
  web.py, templates/     FastAPI web UI
searxng/settings.yml     config for the bundled SearXNG (JSON output, eBay engine fixes)
tests/                   rule tables, source parsing, scan/email flow, web routes
```
