#!/bin/bash
set -e

BOLD="\033[1m"
GREEN="\033[32m"
NC="\033[0m"

echo "Running Check 1/3: openenv validate..."
python -m uv run openenv validate

echo "Running Check 2/3: docker build..."
docker build -t supportdesk_env .

echo "Running Check 3/3: Python syntax check..."
python -m uv run python -m py_compile server.py

printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
