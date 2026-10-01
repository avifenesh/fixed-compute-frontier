#!/usr/bin/env bash
# Wait until every earlier launcher on the box reports "all done", then launch the 9B list on all 8 GPUs.
cd "$(dirname "$0")"
until [ "$(grep -l '^all done' /workspace/negeig/main.log /workspace/negeig/main2.log /workspace/negeig/main3.log 2>/dev/null | wc -l)" -ge 3 ]; do sleep 60; done
grep -q REPLAY "${1:?list}" && { echo "list still has the REPLAY placeholder; not launching"; exit 1; }
MODEL=/workspace/negeig/model9b exec bash box_main_m.sh "$1"
