"""Command line: `python -m pricetracker` serves the web UI and scheduler; `... scan` runs one scan."""

import argparse
import logging
import sys

import uvicorn

from . import config


def main() -> int:
    parser = argparse.ArgumentParser(prog="pricetracker", description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("serve", help="run the web UI and the scan scheduler (default)")
    scan = commands.add_parser("scan", help="run one scan now and exit (for cron or CI)")
    scan.add_argument("--no-email", action="store_true", help="do not email new matches")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "scan":
        from .db import Database
        from .scanner import start_scan

        db = Database(config.DB_PATH)
        run = db.get_run(start_scan(db, trigger="cli", send_email=not args.no_email))
        print(
            f"Scan #{run['id']}: {run['status']} · {run['accepted_count']} accepted, {run['rejected_count']} rejected, "
            f"{run['new_match_count']} new matches · email: {run['email_status']}"
        )
        if run["error"]:
            print(f"Error: {run['error']}", file=sys.stderr)
        return 0 if run["status"] == "ok" else 1

    from .web import create_app

    if not config.UI_PASSWORD:
        logging.getLogger("pricetracker").warning(
            "TRACKER_UI_PASSWORD is not set: the web UI has no login. Only expose it on a trusted network."
        )
    uvicorn.run(create_app(), host=config.HOST, port=config.PORT, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
