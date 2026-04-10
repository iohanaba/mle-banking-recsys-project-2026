#!/bin/bash

# Script to set up the full production stack for the banking recommendation system.
# Includes starting MLflow, FastAPI API, and potentially Airflow.

set -euo pipefail # Exit on error, undefined variable, pipe failure

# --- Configuration ---
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE="$PROJECT_ROOT/.env" # Optional .env file for secrets/configs

# Source environment variables if .env file exists
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment variables from $ENV_FILE"
    # Use set -a to export all variables defined subsequently
    set -a
    source "$ENV_FILE"
    set +a
fi

# Paths to executables and logs
MLFLOW_EXEC=$(which mlflow)
UVICORN_EXEC=$(which uvicorn)
PYTHON_EXEC=$(which python)
# DOCKER_COMPOSE_EXEC=$(which docker-compose) # Uncomment if using docker-compose

# Log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Service-specific configurations
MLFLOW_HOST=${MLFLOW_HOST:-"localhost"}
MLFLOW_PORT=${MLFLOW_PORT:-5000}
MLFLOW_BACKEND_STORE_URI=${MLFLOW_BACKEND_STORE_URI:-"$PROJECT_ROOT/mlruns"}
MLFLOW_DEFAULT_ARTIFACT_ROOT=${MLFLOW_DEFAULT_ARTIFACT_ROOT:-"$PROJECT_ROOT/mlruns"}

API_HOST=${API_HOST:-"0.0.0.0"}
API_PORT=${API_PORT:-8000}
API_MODULE="src.api.app:app" # Path to the FastAPI app instance

# Process IDs and PID files
MLFLOW_PID_FILE="$LOG_DIR/mlflow.pid"
API_PID_FILE="$LOG_DIR/api.pid"

# Colors='\033[0;31m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Helper Functions ---

log_info() {
    echo -e "${GREEN}[INFO]$(date '+%Y-%m-%d %H:%M:%S')${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]$(date '+%Y-%m-%d %H:%M:%S')${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]$(date '+%Y-%m-%d %H:%M:%S'){NC} $1"
}

check_pid_running() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null
}

check_service_status() {
    local service_name=$1
    local pid_file=$2
    local exec_check=$3

    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if check_pid_running "$pid"; then
            log_info "$service_name (PID: $pid) is running."
            return 0
        else
            log_warn "PID file for $service_name (PID: $pid) exists but process is not running. Cleaning up..."
            rm -f "$pid_file"
            return 1
        fi
    else
        # Check if process is running without PID file (fallback)
        if pgrep -f "$exec_check" >/dev/null; then
            log_info "$service_name is running (found by pgrep). PID file missing."
            return 0
        else
            log_info "$service_name is NOT running."
            return 1
        fi
    fi
}

start_mlflow() {
    log_info "Checking MLflow status..."
    if check_service_status "MLflow" "$MLFLOW_PID_FILE" "mlflow server"; then
        log_info "MLflow is already running."
        return 0
    fi

    log_info "Starting MLflow server..."
    nohup $MLFLOW_EXEC server \
        --host "$MLFLOW_HOST" \
        --port "$MLFLOW_PORT" \
        --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
        --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
        --workers 1 \
        > "$LOG_DIR/mlflow.log" 2>&1 &
    local mlflow_pid=$!

    # Write PID to file
    echo $mlflow_pid > "$MLFLOW_PID_FILE"

    # Wait a moment for the server to potentially fail on startup
    sleep 5

    if check_pid_running "$mlflow_pid"; then
        log_info "MLflow server started successfully (PID: $mlflow_pid). Logs: $LOG_DIR/mlflow.log"
    else
        log_error "Failed to start MLflow server. Check logs: $LOG_DIR/mlflow.log"
        # Clean up PID file if startup failed
        rm -f "$MLFLOW_PID_FILE"
        return 1
    fi
}

start_api() {
    log_info "Checking API status..."
    if check_service_status "API" "$API_PID_FILE" "uvicorn.*$API_MODULE"; then
        log_info "API is already running."
        return 0
    fi

    log_info "Starting API server (using uvicorn)..."
    # Ensure we run in the project root
    cd "$PROJECT_ROOT"
    nohup $UVICORN_EXEC "$API_MODULE" \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --workers 1 \
        --log-level info \
        --access-log \
        > "$LOG_DIR/api.log" 2>&1 &
    local api_pid=$!

    # Write PID to file
    echo $api_pid > "$API_PID_FILE"

    # Wait a moment for the server to potentially fail on startup
    sleep 5

    if check_pid_running "$api_pid"; then
        log_info "API server started successfully (PID: $api_pid). Logs: $LOG_DIR/api.log"
    else
        log_error "Failed to start API server. Check logs: $LOG_DIR/api.log"
        # Clean up PID file if startup failed
        rm -f "$API_PID_FILE"
        return 1
    fi
}

# --- Main Execution ---

log_info "Starting production setup..."

# Check if required executables are available
if ! command -v mlflow &> /dev/null; then
    log_error "MLflow executable not found. Please install mlflow (pip install mlflow)."
    exit 1
fi

if ! command -v uvicorn &> /dev/null; then
    log_error "Uvicorn executable not found. Please install uvicorn (pip install uvicorn)."
    exit 1
fi

# Start services in order (MLflow often needed for API logging)
start_mlflow
start_api

# Final status check
echo ""
log_info "Final Status:"
check_service_status "MLflow" "$MLFLOW_PID_FILE" "mlflow server"
check_service_status "API" "$API_PID_FILE" "uvicorn.*$API_MODULE"

log_info "Production setup completed. Check logs in $LOG_DIR/ for details."
echo "MLflow UI: http://$MLFLOW_HOST:$MLFLOW_PORT"
echo "API Docs: http://$API_HOST:$API_PORT/docs"

# Optional: Check for Airflow DAGs if Airflow is present
if command -v airflow &> /dev/null; then
    log_info "Airflow found. Ensure scheduler is running separately if needed (e.g., 'airflow scheduler' in another terminal)."
else
    log_info "Airflow not found. DAG scheduling requires Airflow installation."
fi
