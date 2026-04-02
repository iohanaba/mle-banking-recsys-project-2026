#!/bin/bash
# Скрипт запуска MLflow tracking server с локальным бэкендом

set -e

# Параметры сервера
MLFLOW_PORT=5000
BACKEND_STORE="./mlruns"
ARTIFACT_ROOT="./mlruns/artifacts"

echo "Setting up MLflow infrastructure..."

# Создаём директории для хранения
mkdir -p "$BACKEND_STORE" "$ARTIFACT_ROOT"

# Проверяем, не занят ли порт
if lsof -ti:$MLFLOW_PORT; then
    echo "Port $MLFLOW_PORT is already in use. Stopping previous process..."
    kill -9 $(lsof -ti:$MLFLOW_PORT) 2>/dev/null || true
fi

# Запускаем MLflow server в фоновом режиме
echo "Starting MLflow server on port $MLFLOW_PORT..."
mlflow server \
    --backend-store-uri file://$(pwd)/$BACKEND_STORE \
    --default-artifact-root file://$(pwd)/$ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port $MLFLOW_PORT \
    > mlflow_server.log 2>&1 &

# Сохраняем PID процесса
echo $! > mlflow.pid

# Ожидаем запуска сервера
sleep 3

# Проверяем доступность
if curl -s http://localhost:$MLFLOW_PORT/health > /dev/null; then
    echo "MLflow server started successfully."
    echo "Tracking URI: http://localhost:$MLFLOW_PORT"
    echo "PID: $(cat mlflow.pid)"
else
    echo "Warning: MLflow server may not be fully ready. Check mlflow_server.log"
fi