#!/bin/bash
# Скрипт загрузки и распаковки обучающих данных
# Источник: Yandex Disk public link с обработкой редиректов

set -e

DATA_DIR="data/raw"
ARCHIVE_NAME="train_ver2.csv.zip"
OUTPUT_FILE="train_ver2.csv"

echo "Starting data download process..."

# Создаём директорию если не существует
mkdir -p "$DATA_DIR"

# Формируем ссылку для скачивания
# Яндекс.Диск требует извлечения реальной ссылки через мета-API
BASE_URL="https://disk.yandex.com/d/Io0siOESo2RAaA"
META_URL="https://cloud-api.yandex.net/v1/disk/public/resources?public_key=${BASE_URL}&offset=0&limit=100"

echo "Fetching metadata from Yandex Disk API..."

# Получаем метаданные и извлекаем прямую ссылку на скачивание
DOWNLOAD_URL=$(curl -s "$META_URL" | grep -o '"file":"[^"]*"' | head -1 | cut -d'"' -f4)

# Если не удалось получить ссылку через API, пробуем альтернативный метод
if [ -z "$DOWNLOAD_URL" ]; then
    echo "API method failed, trying direct download with headers..."
    # Прямая ссылка с правильными заголовками для обхода проверки браузера
    DOWNLOAD_URL="${BASE_URL}?dl=1"
fi

echo "Downloading archive from: $DOWNLOAD_URL"

# Скачиваем с заголовками, имитирующими браузер
wget -q --show-progress \
    --user-agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36" \
    --header="Accept: */*" \
    -O "${DATA_DIR}/${ARCHIVE_NAME}" \
    "$DOWNLOAD_URL"

# Проверяем размер файла - должен быть больше 100 МБ
FILE_SIZE=$(stat -c%s "${DATA_DIR}/${ARCHIVE_NAME}" 2>/dev/null || echo 0)
MIN_SIZE=100000000  # 100 MB

if [ "$FILE_SIZE" -lt "$MIN_SIZE" ]; then
    echo "Warning: Downloaded file is only $((FILE_SIZE / 1024)) KB, expected > 100 MB"
    echo "This may indicate an HTML error page was downloaded instead of the archive."
    
    # Проверяем, не является ли файл HTML
    if head -c 100 "${DATA_DIR}/${ARCHIVE_NAME}" | grep -q "<!DOCTYPE\|<html"; then
        echo "Error: Downloaded content appears to be HTML, not a ZIP archive."
        echo "Please download the file manually from: $BASE_URL"
        echo "Then place it in data/raw/ and rename to train_ver2.csv.zip"
        exit 1
    fi
fi

# Проверяем успешность загрузки
if [ -f "${DATA_DIR}/${ARCHIVE_NAME}" ]; then
    echo "Archive downloaded successfully. Size: $((FILE_SIZE / 1024 / 1024)) MB"
    
    # Проверяем, является ли файл валидным ZIP-архивом
    if ! unzip -t "${DATA_DIR}/${ARCHIVE_NAME}" > /dev/null 2>&1; then
        echo "Error: File is not a valid ZIP archive."
        echo "Attempting to treat as gzip-compressed CSV..."
        
        # Пробуем распаковать как gzip если это .csv.gz
        if gunzip -c "${DATA_DIR}/${ARCHIVE_NAME}" > "${DATA_DIR}/${OUTPUT_FILE}" 2>/dev/null; then
            echo "Successfully extracted as gzip."
        else
            echo "Error: Cannot extract archive. Please verify the source file."
            exit 1
        fi
    else
        # Распаковываем ZIP-архив
        echo "Extracting ZIP archive..."
        unzip -q -o "${DATA_DIR}/${ARCHIVE_NAME}" -d "$DATA_DIR"
    fi
    
    # Ищем целевой CSV-файл (может быть вложен в архив)
    if [ ! -f "${DATA_DIR}/${OUTPUT_FILE}" ]; then
        echo "Searching for CSV file in extracted contents..."
        FOUND_CSV=$(find "$DATA_DIR" -maxdepth 2 -name "*.csv" -type f | head -1)
        if [ -n "$FOUND_CSV" ]; then
            mv "$FOUND_CSV" "${DATA_DIR}/${OUTPUT_FILE}"
            echo "Found and moved: $FOUND_CSV"
        fi
    fi
    
    # Финальная проверка
    if [ -f "${DATA_DIR}/${OUTPUT_FILE}" ]; then
        echo "Data file ${OUTPUT_FILE} ready."
        echo "File size: $(du -h ${DATA_DIR}/${OUTPUT_FILE} | cut -f1)"
        echo "Row count (sample): $(head -1000 ${DATA_DIR}/${OUTPUT_FILE} | wc -l)"
        
        # Удаляем архив для экономии места
        rm -f "${DATA_DIR}/${ARCHIVE_NAME}"
        echo "Archive removed."
    else
        echo "Error: Target file ${OUTPUT_FILE} not found after extraction."
        echo "Contents of data/raw/:"
        ls -la "$DATA_DIR"
        exit 1
    fi
else
    echo "Error: Failed to download archive."
    exit 1
fi

echo "Data preparation completed."