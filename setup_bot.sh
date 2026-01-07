#!/bin/bash

# Clear screen (optional, mimicking @echo off + cls somewhat, though typically just running commands is fine)
# clear

echo "========================================"
echo "SOL/USDT Trading Bot Setup"
echo "========================================"
echo ""

echo "[1/3] Installing required libraries..."
pip install ccxt pandas numpy

echo ""
echo "[2/3] Checking if bot_backtest.py exists..."
if [ ! -f bot_backtest.py ]; then
    echo "ERROR: bot_backtest.py not found!"
    echo ""
    echo "Please create bot_backtest.py in this folder first."
    echo "Copy the code from Claude and save it as bot_backtest.py"
    echo ""
    read -p "Press any key to continue..."
    exit 1
fi

echo ""
echo "[3/3] Running backtest and optimization..."
echo "This will take 5-10 minutes..."
echo ""
python3 bot_backtest.py

echo ""
echo "========================================"
echo "Process Complete!"
echo "========================================"
read -p "Press any key to close..."
