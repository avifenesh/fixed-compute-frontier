#!/usr/bin/env bash
set -euo pipefail

protected_instance_id="45944186"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  echo "usage: bash experiments/run_raw_prose_equality_plane_t10_remote.sh USER@HOST PORT NEW_INSTANCE_ID [quick|pilot]" >&2
  exit 2
}

[[ $# -ge 3 && $# -le 4 ]] || usage
remote_host="$1"
remote_port="$2"
instance_id="$3"
run_mode="${4:-pilot}"

[[ "$remote_port" =~ ^[0-9]+$ ]] || usage
[[ "$instance_id" =~ ^[0-9]+$ ]] || usage
[[ "$run_mode" == "quick" || "$run_mode" == "pilot" ]] || usage
if [[ "$instance_id" == "$protected_instance_id" ]]; then
  echo "refusing protected stopped instance $protected_instance_id" >&2
  exit 3
fi

remote_dir="/workspace/fixed-compute-frontier-t10-${instance_id}"
ssh_options=(-p "$remote_port" -o StrictHostKeyChecking=accept-new)
rsync_ssh="ssh -p $remote_port -o StrictHostKeyChecking=accept-new"

ssh "${ssh_options[@]}" "$remote_host" "mkdir -p '$remote_dir/experiments' '$remote_dir/tests' '$remote_dir/results'"

rsync -az --prune-empty-dirs -e "$rsync_ssh" \
  --include='*/' --include='*.py' --include='run_raw_prose_equality_plane_t10_gpu.sh' --exclude='*' \
  "$repo_dir/experiments/" "$remote_host:$remote_dir/experiments/"
rsync -az --prune-empty-dirs -e "$rsync_ssh" \
  --include='*/' --include='test_*.py' --exclude='*' \
  "$repo_dir/tests/" "$remote_host:$remote_dir/tests/"
rsync -az -e "$rsync_ssh" \
  "$repo_dir/results/consensus-permutation-raw-prose-stage0b.json" \
  "$repo_dir/results/raw-prose-equality-plane-t10-preregistration.md" \
  "$repo_dir/results/raw-prose-equality-plane-t10-training-pilot-protocol.md" \
  "$remote_host:$remote_dir/results/"

set +e
ssh "${ssh_options[@]}" "$remote_host" \
  "cd '$remote_dir' && bash experiments/run_raw_prose_equality_plane_t10_gpu.sh '$run_mode'"
remote_status=$?
set -e

for artifact in \
  raw-prose-equality-plane-t10-data-manifest.json \
  raw-prose-equality-plane-t10-gpu-run.log \
  raw-prose-equality-plane-t10-quick.json \
  raw-prose-equality-plane-t10-training-pilot.json
do
  rsync -az -e "$rsync_ssh" \
    "$remote_host:$remote_dir/results/$artifact" "$repo_dir/results/" 2>/dev/null || true
done

exit "$remote_status"
