# api/routes/trigger.py
"""POST /trigger — Manuel analiz tetikleme."""
import asyncio
import uuid
from typing import Optional, Literal
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from services.intelligence_service import intelligence_service

router = APIRouter()


class TriggerRequest(BaseModel):
    scope:    Literal["all", "category"] = Field("all",    description="'all' → tüm kategoriler, 'category' → belirli")
    category: Optional[str]              = Field(None,     description="scope='category' ise hangi kategori")
    priority: Literal["normal", "urgent"] = Field("normal", description="'urgent' → öncelikli çalıştır")


@router.post("/trigger", tags=["Admin"])
async def trigger_analysis(request: TriggerRequest, background_tasks: BackgroundTasks):
    """
    Manuel analiz tetikler (arka planda çalışır).

    Nightly batch'i beklemek zorunda kalmadan anlık güncelleme için kullanın.
    """
    task_id = str(uuid.uuid4())[:8]

    async def _run():
        if request.scope == "category" and request.category:
            intelligence_service.predict(category=request.category, top_n=500)
        else:
            await intelligence_service.nightly_batch()

    background_tasks.add_task(_run)

    return {
        "task_id":  task_id,
        "status":   "queued",
        "scope":    request.scope,
        "category": request.category,
        "priority": request.priority,
        "message":  "Analiz arka planda başlatıldı",
    }
