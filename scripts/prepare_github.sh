#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -f .env.agent ]]; then
  echo "Refusing to stage .env.agent because it may contain an API key." >&2
  echo "Keep only .env.agent.example in the repository." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed." >&2
  exit 1
fi
if ! git lfs version >/dev/null 2>&1; then
  echo "Git LFS is required before staging epoch699.ckpt." >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  git init -b main
fi
git lfs install --local
git lfs track "*.ckpt" "macemodel/*.model"
git add .gitattributes
git add .

echo
echo "Repository staged. Verify before committing:"
echo "  git status --short"
echo "  git lfs ls-files"
echo "  git diff --cached --stat"
echo
echo "Then commit and publish:"
echo "  git commit -m 'Initial SymmBand-Agent monorepo'"
echo "  gh repo create SymmBand-Agent --source=. --private --push"

