"""Static HTML frontend for DocFinder web UI."""

from __future__ import annotations

from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


def _load_template() -> str:
    template = files("docfinder.web").joinpath("templates", "index.html")
    return template.read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = _load_template()
    return HTMLResponse(content=html)
