#!/bin/bash
# Convenient launcher for PulmoNet AI Web Application
cd "$(dirname "$0")"

echo "=========================================================="
echo "🫁 PulmoNet AI - COVID-19 Chest X-ray Detection Suite"
echo "=========================================================="

if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "Using Python: $PYTHON_EXEC"
echo "Starting web server at http://127.0.0.1:8001..."
echo "Press Ctrl+C to stop."
echo "=========================================================="

$PYTHON_EXEC main.py web --host 127.0.0.1 --port 8001

