.PHONY: db-up db-down db-logs db-shell db-upgrade test

POSTGRES_DB ?= recommender
POSTGRES_USER ?= recommender
POSTGRES_PASSWORD ?= recommender
POSTGRES_PORT ?= 55432
POSTGRES_CONTAINER ?= mal-recommender-postgres
POSTGRES_VOLUME ?= mal-recommender-postgres-data
POSTGRES_IMAGE ?= docker.io/pgvector/pgvector:pg16
PODMAN_ROOT ?= /tmp/mal-recommender-podman-root
PODMAN_RUNROOT ?= /tmp/mal-recommender-podman-run
PODMAN = XDG_RUNTIME_DIR=$(PODMAN_RUNROOT) podman --root $(PODMAN_ROOT) --runroot $(PODMAN_RUNROOT)

db-up:
	@if command -v docker >/dev/null 2>&1; then \
		docker compose up -d postgres; \
	elif command -v podman >/dev/null 2>&1; then \
		mkdir -p $(PODMAN_ROOT) $(PODMAN_RUNROOT); \
		$(PODMAN) volume inspect $(POSTGRES_VOLUME) >/dev/null 2>&1 || $(PODMAN) volume create $(POSTGRES_VOLUME) >/dev/null; \
		$(PODMAN) run -d --replace \
			--name $(POSTGRES_CONTAINER) \
			-e POSTGRES_DB=$(POSTGRES_DB) \
			-e POSTGRES_USER=$(POSTGRES_USER) \
			-e POSTGRES_PASSWORD=$(POSTGRES_PASSWORD) \
			-p $(POSTGRES_PORT):5432 \
			-v $(POSTGRES_VOLUME):/var/lib/postgresql/data \
			$(POSTGRES_IMAGE); \
		until $(PODMAN) exec $(POSTGRES_CONTAINER) pg_isready -U $(POSTGRES_USER) -d $(POSTGRES_DB) >/dev/null 2>&1; do sleep 1; done; \
	else \
		echo "Neither docker nor podman is installed."; \
		exit 1; \
	fi

db-down:
	@if command -v docker >/dev/null 2>&1; then \
		docker compose down; \
	elif command -v podman >/dev/null 2>&1; then \
		mkdir -p $(PODMAN_ROOT) $(PODMAN_RUNROOT); \
		$(PODMAN) rm -f $(POSTGRES_CONTAINER); \
	else \
		echo "Neither docker nor podman is installed."; \
		exit 1; \
	fi

db-logs:
	@if command -v docker >/dev/null 2>&1; then \
		docker compose logs -f postgres; \
	elif command -v podman >/dev/null 2>&1; then \
		mkdir -p $(PODMAN_ROOT) $(PODMAN_RUNROOT); \
		$(PODMAN) logs -f $(POSTGRES_CONTAINER); \
	else \
		echo "Neither docker nor podman is installed."; \
		exit 1; \
	fi

db-shell:
	@if command -v docker >/dev/null 2>&1; then \
		docker compose exec postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB); \
	elif command -v podman >/dev/null 2>&1; then \
		mkdir -p $(PODMAN_ROOT) $(PODMAN_RUNROOT); \
		$(PODMAN) exec -it $(POSTGRES_CONTAINER) psql -U $(POSTGRES_USER) -d $(POSTGRES_DB); \
	else \
		echo "Neither docker nor podman is installed."; \
		exit 1; \
	fi

db-upgrade:
	mal-rec db upgrade

test:
	python -m pytest -q
