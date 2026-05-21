# MAL Recommender

A Python-first model lab for anime and manga recommendations using the MyAnimeList API v2.

The first version focuses on ingestion, cached trait labels, embeddings, hybrid ranking, feedback capture, and a small FastAPI/CLI surface. It is designed to work without a trained model first, while collecting data that can support a learned re-ranker later.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,llm]"
cp .env.example .env
```

Create a MAL API client at https://myanimelist.net/apiconfig and set `MAL_CLIENT_ID` in `.env`.

## CLI

```bash
mal-rec init-db
mal-rec mal-auth login
mal-rec ingest user-list --content-type anime
mal-rec ingest user-list --content-type manga
mal-rec label traits
mal-rec recommend --mood cozy --mode cooldown_after_heavy
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

V1 uses a hybrid ranker:

- content similarity from TF-IDF embeddings over title, synopsis, genres, media type, and generated traits
- positive taste profile from highly scored or completed items
- negative signals from dropped, on-hold, or low-scored items
- recent consumption context from list update and finish dates
- explicit mood/mode controls
- filler endurance inferred from completion/drop/progress patterns

The system logs recommendation impressions and feedback so a future trained ranker can learn from accepted, dismissed, saved, and opened recommendations.
