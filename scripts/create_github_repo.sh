#!/usr/bin/env bash
# Run only in an authenticated GitHub CLI environment. Creates a PRIVATE repo.
set -euo pipefail
PMR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PMR_ROOT"
command -v gh >/dev/null || { echo 'GitHub CLI gh is required to create the repository.' >&2; exit 2; }
ACCOUNT="$(gh api user --jq .login)"
[[ "$ACCOUNT" == XintongYang1010 || "$ACCOUNT" == xintongyang1010 ]] || { echo "Wrong GitHub account: $ACCOUNT" >&2; exit 2; }
if [[ ! -d .git ]]; then
  git init -b main
fi
git config user.name XintongYang1010
git config user.email xintongyang1010@outlook.com
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  git add .
  git commit -m 'Implement week-one predictive memory diagnosis and controller training'
fi
if git remote get-url origin >/dev/null 2>&1; then
  echo 'origin already exists; refusing to overwrite it' >&2
  exit 2
fi
gh repo create XintongYang1010/predictive-memory-lab --private --source=. --remote=origin --push \
  --description 'Week-one RealMotion predictive-memory diagnosis and controller training'
