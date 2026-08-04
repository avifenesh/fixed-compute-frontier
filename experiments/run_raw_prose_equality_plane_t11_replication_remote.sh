#!/usr/bin/env bash
set -euo pipefail

protected_instance_id="45944186"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
identity_file="${T11_SSH_IDENTITY:-${HOME}/.ssh/id_ed25519}"

[[ $# -eq 3 ]] || {
  echo "usage: bash experiments/run_raw_prose_equality_plane_t11_replication_remote.sh USER@HOST PORT NEW_INSTANCE_ID" >&2
  exit 2
}
remote_host="$1"
remote_port="$2"
instance_id="$3"
[[ "$remote_port" =~ ^[0-9]+$ && "$instance_id" =~ ^[0-9]+$ ]] || exit 2
[[ -f "$identity_file" ]] || { echo "missing identity $identity_file" >&2; exit 2; }
if [[ "$instance_id" == "$protected_instance_id" ]]; then
  echo "refusing protected stopped instance $protected_instance_id" >&2
  exit 3
fi

remote_dir="/workspace/fixed-compute-frontier-t10-${instance_id}"
ssh_options=(-i "$identity_file" -p "$remote_port" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)
rsync_ssh="ssh -i $identity_file -p $remote_port -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"

rsync -az --prune-empty-dirs -e "$rsync_ssh" \
  --include='*/' --include='*.py' --include='run_raw_prose_equality_plane_t11_replication_gpu.sh' --exclude='*' \
  "$repo_dir/experiments/" "$remote_host:$remote_dir/experiments/"
rsync -az --prune-empty-dirs -e "$rsync_ssh" \
  --include='*/' --include='test_*.py' --exclude='*' \
  "$repo_dir/tests/" "$remote_host:$remote_dir/tests/"
rsync -az -e "$rsync_ssh" \
  "$repo_dir/results/raw-prose-equality-plane-t11-replication-preregistration.md" \
  "$remote_host:$remote_dir/results/"

set +e
ssh "${ssh_options[@]}" "$remote_host" \
  "cd '$remote_dir' && bash experiments/run_raw_prose_equality_plane_t11_replication_gpu.sh"
remote_status=$?
set -e

for artifact in \
  raw-prose-equality-plane-t11-replication-gpu-run.log \
  raw-prose-equality-plane-t11-replication.json \
  raw-prose-equality-plane-t11-replication-seed-6703.json \
  raw-prose-equality-plane-t11-replication-seed-6997.json \
  raw-prose-equality-plane-t11-replication-seed-7307.json
do
  rsync -az -e "$rsync_ssh" \
    "$remote_host:$remote_dir/results/$artifact" "$repo_dir/results/" 2>/dev/null || true
done

exit "$remote_status"
