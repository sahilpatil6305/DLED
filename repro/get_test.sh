#!/usr/bin/env bash
# Download the 35 DF40 test zips NOEFS needs, one at a time:
#   rclone copy -> md5 check against Drive -> extract (skip Celeb-DF 'cdf') -> delete zip.
# Needs rclone authenticated to a Google account (README step 4); anonymous
# gdown downloads of these large public files hit Drive's "too many users" limit.
# Rerunning skips methods already marked with DF40/<method>/.done.
# Run in the background:  nohup repro/get_test.sh > ~/DLED/runs/get_test.log 2>&1 &
set -u
DLED_DATA="${DLED_DATA:-$HOME/DLED/data}"
RC="${RCLONE:-$HOME/DLED/tools/rclone} --config ${RCLONE_CONFIG:-$HOME/DLED/tools/rclone.conf}"
SRC=1980LCMAutfWvV6zvdxhoeIa67TmzKLQ_
IN="$DLED_DATA/_incoming/test"; mkdir -p "$IN"
while read -r zip name; do
  [ -z "$zip" ] && continue
  dest="$DLED_DATA/DF40/$name"
  if [ -f "$dest/.done" ]; then echo "skip $name (done)"; continue; fi
  echo "$(date +%H:%M) $name: downloading $zip"
  $RC copy gdrive: "$IN" --drive-root-folder-id $SRC --include "/$zip" -q || { echo "FAIL download $zip"; continue; }
  $RC check gdrive: "$IN" --drive-root-folder-id $SRC --include "/$zip" --one-way -q || { echo "FAIL checksum $zip"; continue; }
  top=$(zipinfo -1 "$IN/$zip" | grep -v '^__MACOSX' | head -1 | cut -d/ -f1)
  case "$top" in
    ff|cdf|fake|real|frames)
      mkdir -p "$dest"; unzip -qn "$IN/$zip" -x 'cdf/*' '__MACOSX/*' -d "$dest" ;;
    *)
      tmp="$IN/x_$name"; rm -rf "$tmp"; mkdir -p "$tmp"
      unzip -qn "$IN/$zip" -x '*/cdf/*' '__MACOSX/*' -d "$tmp"
      rm -rf "$dest"; mv "$tmp/$top" "$dest"; rm -rf "$tmp" ;;
  esac
  rm -f "$IN/$zip"; touch "$dest/.done"
  echo "$(date +%H:%M) $name: done ($(du -sh "$dest" | cut -f1)), zip top='$top', free $(df -h ~ | tail -1 | awk '{print $4}')"
done <<'LIST'
fsgan.zip fsgan
faceswap.zip faceswap
simswap.zip simswap
inswap.zip inswap
blendface.zip blendface
uniface.zip uniface
mobileswap.zip mobileswap
e4s.zip e4s
facedancer.zip facedancer
deepfacelab.zip deepfacelab
fomm.zip fomm
facevid2vid.zip facevid2vid
wav2lip.zip wav2lip
MRAA.zip MRAA
one_shot_free.zip one_shot_free
pirender.zip pirender
tpsm.zip tpsm
lia.zip lia
danet.zip danet
sadtalker.zip sadtalker
mcnet.zip mcnet
hyperreenact.zip hyperreenact
heygen.zip heygen
VQGAN.zip VQGAN
StyleGAN2.zip StyleGAN2
StyleGAN3.zip StyleGAN3
StyleGANXL.zip StyleGANXL
ddim.zip ddim
RDDM.zip rddm
DiT.zip DiT
SiT.zip SiT
MidJourney.zip MidJourney
whichfaceisreal.zip whichisreal
sd2.1.zip sd2.1
pixart.zip pixart
LIST
echo "ALL FINISHED $(date)"
