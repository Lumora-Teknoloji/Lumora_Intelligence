# services/scheduler_service.py
"""
Lumora Intelligence — APScheduler kurulumu.

Zamanlanmış görevler:
  - Her gece 02:00 → nightly_batch()   (rank momentum, category signals, alerts)
  - Her Pazar 03:00 → weekly_retrain() (CatBoost yeniden eğitim)
"""
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

import config

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


def create_scheduler(intelligence_service) -> AsyncIOScheduler:
    """
    Scheduler oluşturur ve görevleri ekler.
    intelligence_service → IntelligenceService singleton

    Returns:
        Yapılandırılmış (henüz başlatılmamış) AsyncIOScheduler
    """
    global _scheduler
    _scheduler = AsyncIOScheduler(timezone="Europe/Istanbul")

    # ─── Nightly Batch — her gece 02:00 ─────────────────────────────────────
    _scheduler.add_job(
        intelligence_service.nightly_batch,
        trigger=CronTrigger(
            hour=config.NIGHTLY_BATCH_HOUR,
            minute=config.NIGHTLY_BATCH_MINUTE,
        ),
        id="nightly_batch",
        name="Lumora Intelligence — Nightly Batch",
        replace_existing=True,
        max_instances=1,
        coalesce=True,  # Birden fazla biriktiyse tek çalıştır
    )

    # ─── Weekly Retrain — her Pazar 03:00 ────────────────────────────────────
    _scheduler.add_job(
        intelligence_service.weekly_retrain,
        trigger=CronTrigger(
            day_of_week=config.WEEKLY_RETRAIN_DAY,
            hour=config.WEEKLY_RETRAIN_HOUR,
            minute=0,
        ),
        id="weekly_retrain",
        name="Lumora Intelligence — Weekly CatBoost Retrain",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    logger.info(
        f"📅 Scheduler yapılandırıldı: "
        f"nightly={config.NIGHTLY_BATCH_HOUR:02d}:00, "
        f"weekly={config.WEEKLY_RETRAIN_DAY} {config.WEEKLY_RETRAIN_HOUR:02d}:00"
    )
    return _scheduler


def get_scheduler() -> AsyncIOScheduler | None:
    return _scheduler
