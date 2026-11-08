"""Static HTML frontend for DocFinder web UI."""

from __future__ import annotations

from importlib.resources import files

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


def _load_template(name: str) -> str:
    template = files("docfinder.web").joinpath("templates", name)
    return template.read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    response = HTMLResponse(content=_load_template("index.html"))
    # HTML carries the header logo; never let any client cache it stale.
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@router.get("/spotlight", response_class=HTMLResponse)
async def spotlight() -> HTMLResponse:
    response = HTMLResponse(content=_load_template("spotlight.html"))
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response
