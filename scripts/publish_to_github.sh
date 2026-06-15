#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: sh scripts/publish_to_github.sh OWNER/REPO" >&2
  echo "example: sh scripts/publish_to_github.sh andyzhou4451/NWP-Benchmark" >&2
  exit 2
fi

REPO="$1"
BRANCH="${BRANCH:-codex/cluster-weight-downloads}"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh is required. Install GitHub CLI or push manually with git." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run: gh auth login" >&2
  exit 1
fi

if gh repo view "$REPO" >/dev/null 2>&1; then
  if git remote get-url github >/dev/null 2>&1; then
    git remote set-url github "https://github.com/$REPO.git"
  else
    git remote add github "https://github.com/$REPO.git"
  fi
else
  gh repo create "$REPO" --public --source . --remote github
fi

git push github HEAD:"$BRANCH"

echo "Pushed to https://github.com/$REPO/tree/$BRANCH"
