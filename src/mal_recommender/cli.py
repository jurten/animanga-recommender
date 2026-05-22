from __future__ import annotations

import asyncio
import secrets
import webbrowser

import typer

from . import db
from .ingest import add_history_event, canonical_id_for_source, ingest_anilist, ingest_user_list
from .mal import MALClient, make_pkce_pair
from .models import RecommendationMode, RecommendationRequest
from .recommender import recommend
from .traits import label_missing
from .vectors import build_vectors, query_vectors
from .works import build_works

app = typer.Typer(help="MyAnimeList recommender model lab")
auth_app = typer.Typer(help="MAL OAuth helpers")
db_app = typer.Typer(help="Database commands")
ingest_app = typer.Typer(help="Ingestion commands")
history_app = typer.Typer(help="Source-agnostic history commands")
label_app = typer.Typer(help="Trait labeling commands")
traits_app = typer.Typer(help="Trait build commands")
vectors_app = typer.Typer(help="Vector embedding commands")
works_app = typer.Typer(help="Work grouping commands")
app.add_typer(auth_app, name="mal-auth")
app.add_typer(db_app, name="db")
app.add_typer(ingest_app, name="ingest")
app.add_typer(history_app, name="history")
app.add_typer(label_app, name="label")
app.add_typer(traits_app, name="traits")
app.add_typer(vectors_app, name="vectors")
app.add_typer(works_app, name="works")


@app.command("init-db")
def init_database() -> None:
    db.init_db()
    typer.echo("Initialized database")


@db_app.command("upgrade")
def db_upgrade() -> None:
    if db.is_postgres_url():
        try:
            from alembic import command
            from alembic.config import Config
        except ImportError as exc:
            raise typer.BadParameter("Install alembic to run Postgres migrations.") from exc
        command.upgrade(Config("alembic.ini"), "head")
    else:
        db.init_db()
    typer.echo("Database schema is up to date")


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


@ingest_app.command("mal-list")
def ingest_user_list_command(content_type: str = typer.Option("anime")) -> None:
    if content_type not in {"anime", "manga"}:
        raise typer.BadParameter("content_type must be anime or manga")
    count = asyncio.run(ingest_user_list(content_type))
    typer.echo(f"Ingested {count} {content_type} list entries")


@ingest_app.command("user-list")
def ingest_user_list_legacy_command(content_type: str = typer.Option("anime")) -> None:
    count = asyncio.run(ingest_user_list(content_type))
    typer.echo(f"Ingested {count} {content_type} list entries")


@ingest_app.command("anilist")
def ingest_anilist_command(content_type: str = typer.Option("anime"), limit: int = 100) -> None:
    if content_type not in {"anime", "manga"}:
        raise typer.BadParameter("content_type must be anime or manga")
    count = asyncio.run(ingest_anilist(content_type, limit=limit))
    typer.echo(f"Ingested {count} AniList {content_type} records")


@history_app.command("add")
def history_add(
    content_type: str = typer.Option("anime"),
    source: str = typer.Option("mal"),
    source_item_id: str = typer.Option(...),
    event_type: str = typer.Option("saved"),
    user_id: int = 1,
    score: int | None = None,
    progress: int | None = None,
) -> None:
    allowed_events = {"watched", "read", "completed", "dropped", "liked", "disliked", "rated", "saved", "skipped"}
    if content_type not in {"anime", "manga"}:
        raise typer.BadParameter("content_type must be anime or manga")
    if event_type not in allowed_events:
        raise typer.BadParameter(f"event_type must be one of: {', '.join(sorted(allowed_events))}")
    with db.session() as conn:
        canonical_item_id = canonical_id_for_source(conn, source, content_type, source_item_id)
        if canonical_item_id is None:
            raise typer.BadParameter("Unknown source item. Ingest metadata before adding history.")
        event_id = add_history_event(
            conn,
            user_id,
            canonical_item_id,
            content_type,
            event_type,
            "manual",
            {"source": source, "source_item_id": source_item_id},
            score=score,
            progress=progress,
        )
    typer.echo(f"Added history event {event_id}")


@label_app.command("traits")
def label_traits(limit: int = 200, use_llm: bool = True) -> None:
    count = asyncio.run(label_missing(limit=limit, use_llm=use_llm))
    typer.echo(f"Labeled {count} items")


@traits_app.command("build")
def build_traits(limit: int = 200, use_llm: bool = True) -> None:
    count = asyncio.run(label_missing(limit=limit, use_llm=use_llm))
    typer.echo(f"Labeled {count} items")


@works_app.command("build")
def works_build(content_type: str = typer.Option("anime")) -> None:
    if content_type not in {"anime", "manga"}:
        raise typer.BadParameter("content_type must be anime or manga")
    result = build_works(content_type=content_type)
    typer.echo(f"Built {result.works} works, linked {result.links} items, read {result.relations} relations")


@vectors_app.command("build")
def vectors_build(
    limit: int | None = None,
    level: str = typer.Option("item"),
    batch_size: int | None = None,
) -> None:
    if level not in {"item", "work"}:
        raise typer.BadParameter("level must be item or work")
    count = build_vectors(limit=limit, level=level, batch_size=batch_size)
    typer.echo(f"Built {count} {level} vectors")


@vectors_app.command("query")
def vectors_query(query: str, limit: int = 20, level: str = typer.Option("item")) -> None:
    if level not in {"item", "work"}:
        raise typer.BadParameter("level must be item or work")
    rows = query_vectors(query, limit=limit, level=level)
    for idx, row in enumerate(rows, start=1):
        source_id = row.get("source_item_id") or row["id"]
        typer.echo(f"{idx:02d}. [{row['content_type']}] {row['title']} ({row['score']:.4f}) source_id={source_id}")


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
