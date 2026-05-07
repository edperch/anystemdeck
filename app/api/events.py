from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.registry import get as registry_get

router = APIRouter(tags=["events"])

# Close SSE connections that outlive this threshold to prevent zombie
# connections from accumulating when clients disconnect without a TCP RST.
_MAX_SSE_SECONDS = 4 * 3600  # 4 hours


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    job = registry_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    async def stream() -> AsyncIterator[str]:
        last = None
        keepalive_at = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _MAX_SSE_SECONDS
        while loop.time() < deadline:
            snapshot = job.to_state()
            serialized = json.dumps(snapshot)
            if serialized != last:
                yield f"data: {serialized}\n\n"
                last = serialized
                keepalive_at = 0
            if snapshot["status"] in ("done", "error", "cancelled"):
                return
            keepalive_at += 1
            if keepalive_at >= 75:  # ~15s
                yield ": keepalive\n\n"
                keepalive_at = 0
            await asyncio.sleep(0.2)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
