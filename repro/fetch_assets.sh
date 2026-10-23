#!/usr/bin/env bash
# Fetch files the official release imports but does not ship, and verify them.
#   1. clip/bpe_simple_vocab_16e6.txt.gz  (OpenAI CLIP tokenizer vocabulary;
#      DLED's clip/simple_tokenizer.py is byte-identical to OpenAI's and loads it)
# Uses git (github.com), because raw.githubusercontent.com may be blocked.
# The file is git-excluded locally, so the repo stays identical to the release.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BPE=clip/bpe_simple_vocab_16e6.txt.gz
BPE_SHA=924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a

if [ -f "$BPE" ] && echo "$BPE_SHA  $BPE" | sha256sum -c --status; then
  echo "$BPE: already present, checksum OK"
else
  TMP="$(mktemp -d)"
  git clone -q --depth 1 https://github.com/openai/CLIP.git "$TMP/clip"
  cp "$TMP/clip/clip/bpe_simple_vocab_16e6.txt.gz" "$BPE"
  rm -rf "$TMP"
  echo "$BPE_SHA  $BPE" | sha256sum -c
fi
grep -qx "$BPE" .git/info/exclude 2>/dev/null || echo "$BPE" >> .git/info/exclude
