#!/bin/bash
set -e

echo "🚀 Starting FULL MLOps system (LOCAL PRO MODE)..."

# =========================
# ACTIVATE VENV
# =========================
echo "🐍 Activating virtual environment..."

if [ -f "mlops_env/bin/activate" ]; then
  source mlops_env/bin/activate
else
  echo "❌ Virtual environment not found!"
  exit 1
fi

# sanity check
if ! command -v mlflow &> /dev/null; then
  echo "❌ mlflow not found in environment!"
  exit 1
fi

# =========================
# CONFIG
# =========================
PID_FILE="mlops_pids.txt"
LOG_DIR="logs"
mkdir -p $LOG_DIR
rm -f $PID_FILE

# =========================
# HELPER FUNCTIONS
# =========================
start_service () {
  NAME=$1
  CMD=$2
  LOG_FILE=$3

  echo "▶️ Starting $NAME..."

  bash -c "$CMD" > $LOG_FILE 2>&1 &
  PID=$!

  echo $PID >> $PID_FILE

  sleep 1

  if ! kill -0 $PID 2>/dev/null; then
    echo "❌ $NAME failed to start. Check $LOG_FILE"
    exit 1
  fi

  echo "✅ $NAME started (PID: $PID)"
}

wait_for_http () {
  NAME=$1
  URL=$2

  echo "⏳ Waiting for $NAME..."

  for i in {1..30}; do
    if curl -s $URL > /dev/null; then
      echo "✅ $NAME ready"
      return
    fi
    sleep 2
  done

  echo "❌ $NAME did not become ready in time"
  exit 1
}

# =========================
# CLEAN PORTS
# =========================
echo "🧹 Cleaning ports..."
for port in 8000 8501 9090 5000 4200; do
  fuser -k ${port}/tcp 2>/dev/null || true
done

sleep 2

# =========================
# FIX MLFLOW STRUCTURE
# =========================
echo "🛠 Fixing MLflow structure..."
rm -rf mlflow.db
mkdir -p mlflow/artifacts

# =========================
# START MLFLOW
# =========================
start_service "MLflow" \
"mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root ./mlflow/artifacts \
--host 0.0.0.0 \
--port 5000" \
"$LOG_DIR/mlflow.log"

wait_for_http "MLflow" "http://localhost:5000"

# =========================
# START PREFECT SERVER
# =========================
start_service "Prefect Server" \
"prefect server start" \
"$LOG_DIR/prefect.log"

# bazı versiyonlarda health endpoint yok → fallback
wait_for_http "Prefect" "http://localhost:4200" || true

# =========================
# START PREFECT WORKER
# =========================
start_service "Prefect Worker" \
"prefect worker start --pool default-agent-pool --type process" \
"$LOG_DIR/worker.log"

# =========================
# START PROMETHEUS
# =========================
mkdir -p prometheus_data

start_service "Prometheus" \
"prometheus \
--config.file=prometheus.yml \
--storage.tsdb.path=./prometheus_data \
--web.enable-lifecycle" \
"$LOG_DIR/prometheus.log"

wait_for_http "Prometheus" "http://localhost:9090/-/ready"

# =========================
# START FASTAPI
# =========================
start_service "FastAPI" \
"uvicorn src.api.app:app --host 0.0.0.0 --port 8000" \
"$LOG_DIR/api.log"

wait_for_http "API" "http://localhost:8000/health"

# =========================
# START STREAMLIT
# =========================
start_service "Streamlit" \
"streamlit run ui/app.py --server.port 8501" \
"$LOG_DIR/streamlit.log"

# =========================
# START SIMULATOR
# =========================
start_service "Simulator" \
"python src/monitoring/live_simulator.py" \
"$LOG_DIR/simulator.log"

# =========================
# DONE
# =========================
echo ""
echo "================================="
echo "🎉 FULL SYSTEM RUNNING"
echo "================================="
echo "API:        http://localhost:8000"
echo "UI:         http://localhost:8501"
echo "MLflow:     http://localhost:5000"
echo "Prefect:    http://localhost:4200"
echo "Prometheus: http://localhost:9090"
echo "================================="

wait