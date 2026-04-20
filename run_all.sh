#!/bin/bash
set -e

echo "🚀 Starting FULL MLOps system (LOCAL PRO MODE)..."

echo "🐍 Activating virtual environment..."

if [ -f "mlops_env/bin/activate" ]; then
    source mlops_env/bin/activate
else
    echo "❌ Virtual environment not found!"
    exit 1
fi

export PYTHONPATH=$(pwd)

# =========================
# CONFIG
# =========================

PID_FILE="mlops_pids.txt"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
rm -f "$PID_FILE"

PROM_PORT=9091
GRAFANA_PORT=3000

# =========================
# HELPERS
# =========================

start_service () {
    NAME="$1"
    CMD="$2"
    LOG_FILE="$3"

    echo "▶️ Starting $NAME..."

    bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    PID=$!

    echo $PID >> "$PID_FILE"

    sleep 5

    if ! kill -0 $PID 2>/dev/null; then
        echo "❌ $NAME failed to start. Check $LOG_FILE"
        exit 1
    fi

    echo "✅ $NAME started (PID: $PID)"
}

wait_for_http () {
    NAME="$1"
    URL="$2"

    echo "⏳ Waiting for $NAME..."

    for i in {1..40}; do
        if curl -s "$URL" > /dev/null; then
            echo "✅ $NAME ready"
            return
        fi
        sleep 2
    done

    echo "❌ $NAME did not become ready in time"
    exit 1
}

# =========================
# CLEANUP
# =========================

echo "🧹 Cleaning processes..."

pkill -f prometheus || true
pkill -f uvicorn || true
pkill -f streamlit || true
pkill -f prefect || true
pkill -f grafana-server || true

sleep 2

echo "🧹 Cleaning ports..."

for port in 8000 8501 9090 9091 5000 4200 3000; do
    fuser -k ${port}/tcp 2>/dev/null || true
done

sleep 2

# =========================
# GRAFANA CHECK
# =========================

if ! command -v grafana-server &> /dev/null; then
    echo "❌ Grafana not installed. Install with: sudo apt install grafana"
    exit 1
fi

# =========================
# 🔥 FIX: GRAFANA CONFIG
# - anonymous admin eklendi (şifre gerekmez)
# - datasource provisioning eklendi (her başlatmada otomatik tanımlanır)
# =========================

echo "🛠 Preparing Grafana config..."

cat > grafana.ini << EOL
[paths]
data = $(pwd)/grafana_data
logs = $(pwd)/logs
plugins = $(pwd)/grafana_plugins

[server]
http_port = $GRAFANA_PORT

[auth.anonymous]
enabled = true
org_role = Admin
EOL

mkdir -p grafana_data grafana_plugins
chmod -R 777 grafana_data

# Prometheus datasource otomatik provisioning
mkdir -p grafana_data/provisioning/datasources
cat > grafana_data/provisioning/datasources/prometheus.yaml << EOL
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    uid: dfivw5abjdp8gf
    url: http://localhost:${PROM_PORT}
    access: proxy
    isDefault: true
    editable: true
    jsonData:
      httpMethod: POST
      timeInterval: "5s"
EOL

echo "✅ Grafana config ready"

# =========================
# MLFLOW
# =========================

echo "🛠 Starting MLflow..."
rm -rf mlflow.db
mkdir -p mlflow/artifacts

start_service "MLflow" "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow/artifacts --host 0.0.0.0 --port 5000" "$LOG_DIR/mlflow.log"
wait_for_http "MLflow" "http://localhost:5000"

# =========================
# PREFECT
# =========================

export PREFECT_API_URL="http://localhost:4200/api"

echo "🛠 Starting Prefect Server..."
start_service "Prefect Server" "prefect server start" "$LOG_DIR/prefect.log"
wait_for_http "Prefect API" "http://localhost:4200/api/health"

sleep 3

echo "🛠 Creating work pool..."
prefect work-pool create default-agent-pool --type process 2>/dev/null || true

# =========================
# WORKER
# =========================

echo "🛠 Starting Prefect Worker..."
start_service "Prefect Worker" "prefect worker start --pool default-agent-pool --type process" "$LOG_DIR/worker.log"

echo "⏳ Waiting for Prefect Worker..."

for i in {1..30}; do
    if grep -q "Listening for flow runs" "$LOG_DIR/worker.log"; then
        echo "✅ Prefect Worker ready"
        break
    fi
    sleep 2
done

# =========================
# 🔥 FIX: FastAPI ÖNCE başlatılıyor
# Orijinalde Prometheus'tan sonra geliyordu.
# Prometheus ilk scrape'i yaptığında API hazır olsun.
# =========================

start_service "FastAPI" "uvicorn src.api.app:app --host 0.0.0.0 --port 8000" "$LOG_DIR/api.log"
wait_for_http "API" "http://localhost:8000/health"

# =========================
# PROMETHEUS
# =========================

echo "🛠 Starting Prometheus..."

mkdir -p prometheus_data
chmod -R 777 prometheus_data

PROM_CONFIG="$(pwd)/prometheus.yml"
PROM_DATA="$(pwd)/prometheus_data"

start_service "Prometheus" "prometheus --config.file=$PROM_CONFIG --storage.tsdb.path=$PROM_DATA --web.enable-lifecycle --web.listen-address=0.0.0.0:$PROM_PORT" "$LOG_DIR/prometheus.log"
wait_for_http "Prometheus" "http://localhost:$PROM_PORT/-/healthy"

# =========================
# GRAFANA
# =========================

echo "🛠 Starting Grafana..."
start_service "Grafana" "grafana-server --homepath=/usr/share/grafana --config=$(pwd)/grafana.ini" "$LOG_DIR/grafana.log"
wait_for_http "Grafana" "http://localhost:$GRAFANA_PORT/api/health"

# =========================
# STREAMLIT
# =========================

start_service "Streamlit" "streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0" "$LOG_DIR/streamlit.log"
wait_for_http "Streamlit" "http://localhost:8501"

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
echo "Prometheus: http://localhost:$PROM_PORT"
echo "Grafana:    http://localhost:$GRAFANA_PORT"
echo "================================="

wait