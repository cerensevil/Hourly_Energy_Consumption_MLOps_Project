# =========================
# DOCKER MODE
# make up → tüm servisler + deployment otomatik çalışır
# =========================

up:
	docker compose up -d --build

down:
	docker compose down -v

test:
	curl -s http://localhost:4200/api/health
	curl -s http://localhost:5000
	curl -s http://localhost:8000/health

all: up test


# =========================
# LOCAL MODE
# make up-local → servisleri başlat
# make deploy-local → training bekle → inference başlat
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

deploy-local:
	prefect deploy --all
	@echo "🚀 Running training deployment (waiting for completion)..."
	prefect deployment run 'evaluate-and-promote-multi-state/training-deployment' --watch
	@echo "✅ Training done. Starting inference simulation..."
	prefect deployment run 'inference-simulation-flow/inference-simulation'


# =========================
# QUICK DEV MODE
# =========================

dev:
	uvicorn src.api.app:app --reload


# =========================
# HELP
# =========================

help:
	@echo ""
	@echo "========================================="
	@echo "  DOCKER MODE (önerilen)"
	@echo "========================================="
	@echo "  make up            -> Tüm servisleri başlat (otomatik deploy)"
	@echo "  make down          -> Tüm servisleri durdur"
	@echo "  make test          -> Servis health check"
	@echo ""
	@echo "========================================="
	@echo "  LOCAL MODE"
	@echo "========================================="
	@echo "  make up-local      -> Servisleri başlat"
	@echo "  make down-local    -> Servisleri durdur"
	@echo "  make deploy-local  -> Training → Inference"
	@echo ""
	@echo "========================================="
	@echo "  DEV MODE"
	@echo "========================================="
	@echo "  make dev           -> Sadece API (hot reload)"
	@echo ""