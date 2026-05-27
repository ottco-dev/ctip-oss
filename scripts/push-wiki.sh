#!/usr/bin/env bash
# Push docs/github-wiki/ content to the GitHub wiki repository.
#
# Usage:
#   ./scripts/push-wiki.sh <PAT>
#
# Example:
#   ./scripts/push-wiki.sh github_pat_xxxxx
#
set -euo pipefail

PAT="${1:-}"
if [[ -z "$PAT" ]]; then
  echo "Usage: $0 <github_pat>"
  exit 1
fi

REPO="ottco-dev/ctip-oss"
WIKI_URL="https://${PAT}@github.com/${REPO}.wiki.git"
WIKI_DIR="$(mktemp -d)"
PAGES_DIR="$(dirname "$0")/../docs/github-wiki"

echo "→ Regenerating wiki pages from source..."
node "$(dirname "$0")/export-wiki.mjs"

echo "→ Cloning wiki repo..."
git clone "$WIKI_URL" "$WIKI_DIR" 2>/dev/null || {
  # Wiki doesn't exist yet — init it
  mkdir -p "$WIKI_DIR"
  git -C "$WIKI_DIR" init
  git -C "$WIKI_DIR" remote add origin "$WIKI_URL"
}

echo "→ Copying pages..."
cp "$PAGES_DIR"/*.md "$WIKI_DIR"/

echo "→ Committing..."
git -C "$WIKI_DIR" add -A
git -C "$WIKI_DIR" \
  -c user.name="ottco-dev" \
  -c user.email="ottco-dev@users.noreply.github.com" \
  commit -m "docs: update wiki from CTIP source" 2>/dev/null || echo "  (nothing changed)"

echo "→ Pushing..."
git -C "$WIKI_DIR" push origin HEAD:master --force

rm -rf "$WIKI_DIR"
echo "✓ Wiki pushed to https://github.com/${REPO}/wiki"
