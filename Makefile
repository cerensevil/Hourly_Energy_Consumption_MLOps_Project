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
