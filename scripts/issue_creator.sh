#!/usr/bin/env bash
set -euo pipefail

TEMPLATE_PATH="${1:-docs/templates/issue_creator.template.json}"
MODE="${2:---dry-run}"

python3 scripts/issue_creator.py --template "$TEMPLATE_PATH" "$MODE"
