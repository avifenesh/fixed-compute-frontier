#!/usr/bin/env bash
# Wait until the six main runs have finished (done or FAILED in main.log), then launch the next list on GPUs 0-5.
cd "$(dirname "$0")"
L=/workspace/negeig/main.log
until [ "$(grep -cE '^(done|FAILED[a-z ]*) main_(wide|ctrl)_s[012]' $L)" -ge 6 ]; do sleep 60; done
exec bash box_main.sh "${1:?list}"
