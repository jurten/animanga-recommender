# MAL Recommender

A Python-first model lab for anime and manga recommendations using source-agnostic anime/manga data, Postgres + pgvector, local embeddings, and optional MyAnimeList bootstrapping.

The current data spine stores immutable raw source payloads, canonical items, source links, source-agnostic user events, cached trait labels, local item vectors, recommendation runs, and feedback. MAL remains useful for quick history import, but recommendations are keyed by canonical items rather than MAL list rows.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,llm]"
cp .env.example .env
```

Create a MAL API client at https://myanimelist.net/apiconfig and set `MAL_CLIENT_ID` and `MAL_CLIENT_SECRET` in `.env`.

Runtime storage is now Postgres with pgvector:

```bash
DATABASE_URL=postgresql://recommender:recommender@localhost:55432/recommender
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=32
WORK_EMBEDDING_BATCH_SIZE=2
```

Start the local database with Docker:

```bash
docker compose up -d postgres
mal-rec db upgrade
```

Or with the Makefile. It uses Docker Compose when Docker is available, otherwise it starts the same pgvector image with Podman:

```bash
make db-up
make db-upgrade
```

Useful database commands:

```bash
make db-shell
make db-logs
make db-down
```

## CLI

```bash
mal-rec db upgrade
mal-rec mal-auth login
mal-rec ingest mal-list --content-type anime
mal-rec ingest mal-list --content-type manga
mal-rec ingest anilist --content-type anime --limit 500
mal-rec history add --content-type anime --source mal --source-item-id 1 --event-type saved
mal-rec traits build
mal-rec vectors build
mal-rec vectors build --level work --batch-size 4
mal-rec vectors query "cozy after something heavy"
mal-rec recommend --mood cozy --mode cooldown_after_heavy
```

For work vectors on a memory-constrained machine, start with:

```bash
EMBEDDING_DEVICE=cuda mal-rec vectors build --level work --batch-size 1 --limit 25
```

## API

```bash
uvicorn mal_recommender.api:app --reload
```

Useful endpoints:

- `GET /health`
- `GET /items/{content_type}/{mal_id}`
- `POST /recommendations`
- `POST /feedback`
- `GET /runs/{run_id}`

## Model Strategy

The ranker uses a hybrid flow:

- candidate data from canonical items and source links
- local BGE-M3 vectors in pgvector for semantic retrieval
- TF-IDF fallback for local tests and empty vector stores
- positive taste profile from highly scored, completed, liked, or saved events
- negative signals from dropped, disliked, skipped, or low-scored events
- recent consumption context from list update and finish dates
- explicit mood/mode controls
- filler endurance inferred from completion/drop/progress patterns

The system logs recommendation impressions and feedback so a future trained ranker can learn from accepted, dismissed, saved, and opened recommendations.
