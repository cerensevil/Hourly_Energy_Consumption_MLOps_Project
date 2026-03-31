# =========================
# DOCKER MODE
# =========================

up:
	docker compose up -d --build

down:
	docker compose down -v

deploy:
	docker exec -it energy_mlops_container prefect deploy src/training/evaluate_and_promote_flow.py:evaluate_and_promote \
	  --name daily-evaluate-promote --pool default-agent-pool --cron "0 0 * * *"

test:
	curl -s http://localhost:4200/api/health
	curl -s http://localhost:5000
	curl -s http://localhost:8000/health

all: up deploy test


# =========================
# LOCAL MODE (FIXED 🔥)
# =========================

up-local:
	chmod +x run_all.sh
	./run_all.sh

down-local:
	@echo "🛑 Stopping local services..."
	@if [ -f mlops_pids.txt ]; then \
		while read pid; do \
			kill $$pid 2>/dev/null || true; \
		done < mlops_pids.txt; \
		rm mlops_pids.txt; \
	else \
		echo "No PID file found."; \
	fi


# =========================
# QUICK DEV MODE
# =========================

dev:
	uvicorn src.api.app:app --reload


# =========================
# HELP
# =========================

help:
	@echo "Commands:"
	@echo "  make up           -> Start with Docker"
	@echo "  make down         -> Stop Docker"
	@echo "  make up-local     -> Start locally (no Docker)"
	@echo "  make down-local   -> Stop local processes"
	@echo "  make dev          -> Run API in dev mode"