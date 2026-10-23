#!/usr/bin/env bash
# Download ONE Google Drive file by id, report its size, extract it into its
# own folder, and keep the archive unless --rm is given.
#
#   repro/df40_get.sh <file_id> <dest_dir> [--rm]
#
# Example (FF++ real frames):
#   repro/df40_get.sh 1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb "$DLED_DATA/_incoming/ffpp_real"
#
# After extraction, move the method folder into $DLED_DATA/DF40/<method>/ so it
# matches the DF40 layout, then re-run the audit.
set -euo pipefail

ID="${1:?file id}"; DEST="${2:?dest dir}"; RM="${3:-}"
mkdir -p "$DEST/_archive"

echo "Free space where $DEST lives:"; df -h "$DEST" | tail -1

ARCH="$DEST/_archive/$ID"
gdown "https://drive.google.com/uc?id=$ID" -O "$ARCH" || {
    echo "Download failed. Drive often rate-limits big public files." >&2
    echo "Retry later, or download in a browser and copy it to $ARCH" >&2
    exit 1; }

SIZE=$(du -h "$ARCH" | cut -f1); echo "Downloaded $SIZE -> $ARCH"
TYPE=$(file -b "$ARCH")
echo "Type: $TYPE"
case "$TYPE" in
    Zip*)           unzip -q "$ARCH" -d "$DEST" ;;
    *gzip*)         tar -xzf "$ARCH" -C "$DEST" ;;
    *POSIX\ tar*)   tar -xf  "$ARCH" -C "$DEST" ;;
    *RAR*)          echo "RAR archive: install unrar and extract manually"; exit 1 ;;
    *)              echo "Not an archive; left as is."; exit 0 ;;
esac
echo "Extracted into $DEST:"; ls "$DEST" | head
du -sh "$DEST" --exclude=_archive

if [ "$RM" = "--rm" ]; then rm -f "$ARCH" && echo "Removed archive."; fi
