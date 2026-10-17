"""Runs scans at the times configured in Settings (twice a day by default)."""

from __future__ import annotations

import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .db import Database
from .scanner import ScanAlreadyRunning, start_scan
from .settings import load_settings

log = logging.getLogger(__name__)


class ScanScheduler:
    def __init__(self, db: Database):
        self.db = db
        self._scheduler = BackgroundScheduler(timezone="UTC")

    def start(self) -> None:
        self._scheduler.start()
        self.apply(load_settings(self.db))

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)

    def apply(self, settings: dict) -> None:
        """(Re)create the scan jobs from the schedule settings."""
        self._scheduler.remove_all_jobs()
        schedule = settings["schedule"]
        if not schedule["enabled"]:
            log.info("Scheduled scans are disabled")
            return
        for time_of_day in schedule["times"]:
            hour, minute = (int(part) for part in time_of_day.split(":"))
            self._scheduler.add_job(
                self._run,
                CronTrigger(hour=hour, minute=minute, timezone=schedule["timezone"]),
                id=f"scan-{time_of_day}",
                misfire_grace_time=3600,
                coalesce=True,
                max_instances=1,
            )
        log.info("Scheduled scans at %s (%s)", ", ".join(schedule["times"]), schedule["timezone"])

    def next_runs(self) -> list[datetime]:
        times = (getattr(job, "next_run_time", None) for job in self._scheduler.get_jobs())
        return sorted(t for t in times if t)

    def _run(self) -> None:
        try:
            run_id = start_scan(self.db, trigger="scheduled")
            log.info("Scheduled scan finished: run %s", run_id)
        except ScanAlreadyRunning:
            log.warning("Skipped scheduled scan: another scan is still running")
