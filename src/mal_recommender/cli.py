from __future__ import annotations

import asyncio
import secrets
import webbrowser

import typer

from . import db
from .ingest import ingest_user_list
from .mal import MALClient, make_pkce_pair
from .models import RecommendationMode, RecommendationRequest
from .recommender import recommend
from .traits import label_missing

app = typer.Typer(help="MyAnimeList recommender model lab")
auth_app = typer.Typer(help="MAL OAuth helpers")
ingest_app = typer.Typer(help="Ingestion commands")
label_app = typer.Typer(help="Trait labeling commands")
app.add_typer(auth_app, name="mal-auth")
app.add_typer(ingest_app, name="ingest")
app.add_typer(label_app, name="label")


@app.command("init-db")
def init_database() -> None:
    db.init_db()
    typer.echo("Initialized database")


@auth_app.command("login")
def login(open_browser: bool = True) -> None:
    verifier, challenge = make_pkce_pair()
    state = secrets.token_urlsafe(24)
    client = MALClient()
    url = client.auth_url(challenge, state)
    typer.echo("Open this URL and authorize the app:")
    typer.echo(url)
    typer.echo("")
    typer.echo(f"State: {state}")
    typer.echo(f"Code verifier: {verifier}")
    if open_browser:
        webbrowser.open(url)
    code = typer.prompt("Paste authorization code")
    returned_state = typer.prompt("Paste returned state", default=state)
    if returned_state != state:
        raise typer.BadParameter("OAuth state mismatch")
    asyncio.run(client.exchange_code(code, verifier))
    typer.echo("Saved MAL token")


@ingest_app.command("user-list")
def ingest_user_list_command(content_type: str = typer.Option("anime")) -> None:
    if content_type not in {"anime", "manga"}:
        raise typer.BadParameter("content_type must be anime or manga")
    count = asyncio.run(ingest_user_list(content_type))
    typer.echo(f"Ingested {count} {content_type} list entries")


@label_app.command("traits")
def label_traits(limit: int = 200, use_llm: bool = True) -> None:
    count = asyncio.run(label_missing(limit=limit, use_llm=use_llm))
    typer.echo(f"Labeled {count} items")


@app.command("recommend")
def recommend_command(
    mood: str | None = None,
    mode: RecommendationMode = RecommendationMode.mood_match,
    limit: int = 20,
    include_seen: bool = False,
) -> None:
    request = RecommendationRequest(mood=mood, mode=mode, limit=limit, include_seen=include_seen)
    run_id, results = recommend(request)
    typer.echo(f"Run {run_id}")
    for idx, result in enumerate(results, start=1):
        typer.echo(f"{idx:02d}. [{result.content_type}] {result.title} ({result.score})")
        typer.echo(f"    {', '.join(result.reasons)}")
