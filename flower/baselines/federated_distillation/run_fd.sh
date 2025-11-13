#!/bin/bash
# run_fd.sh — запуск Federated Distillation (1 сервер + 3 клиента)

# ✅ Инициализация conda для bash
source ~/miniforge3/etc/profile.d/conda.sh

echo "Starting Federated Distillation experiment..."
conda activate flwr_lab3_fjord

# Папка для логов
LOG_DIR=logs
mkdir -p $LOG_DIR

# Запуск сервера
echo "Starting server..."
python server_fd.py > $LOG_DIR/server.log 2>&1 &
sleep 3

# Запуск трёх клиентов
echo "Starting clients..."
python client_fd.py > $LOG_DIR/client1.log 2>&1 &
python client_fd.py > $LOG_DIR/client2.log 2>&1 &
python client_fd.py > $LOG_DIR/client3.log 2>&1 &

echo "All processes started!"
echo "Check logs in $LOG_DIR/"
