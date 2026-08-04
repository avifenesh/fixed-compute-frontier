#!/usr/bin/env bash
set -euo pipefail

protected_instance_id="45944186"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
identity_file="${T12_SSH_IDENTITY:-/home/avifenesh/.ssh/id_ed25519}"

[[ $# -eq 3 ]] || {
  echo "usage: bash experiments/run_hotpot_semantic_address_t12_remote.sh USER@HOST PORT NEW_INSTANCE_ID" >&2
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

rsync -az -e "$rsync_ssh" \
  "$repo_dir/experiments/hotpot_real_prose_t12_data.py" \
  "$repo_dir/experiments/hotpot_semantic_address_t12.py" \
  "$repo_dir/experiments/run_hotpot_semantic_address_t12_gpu.sh" \
  "$remote_host:$remote_dir/experiments/"
rsync -az -e "$rsync_ssh" \
  "$repo_dir/tests/test_hotpot_real_prose_t12_data.py" \
  "$repo_dir/tests/test_hotpot_semantic_address_t12.py" \
  "$remote_host:$remote_dir/tests/"
rsync -az -e "$rsync_ssh" \
  "$repo_dir/results/hotpot-real-prose-t12-data-manifest.json" \
  "$repo_dir/results/hotpot-semantic-address-t12-preregistration.md" \
  "$remote_host:$remote_dir/results/"
rsync -az -e "$rsync_ssh" \
  "$repo_dir/data/hotpot-real-prose-t12/" \
  "$remote_host:$remote_dir/data/hotpot-real-prose-t12/"

set +e
ssh "${ssh_options[@]}" "$remote_host" \
  "cd '$remote_dir' && bash experiments/run_hotpot_semantic_address_t12_gpu.sh"
remote_status=$?
set -e

for artifact in \
  hotpot-semantic-address-t12-gpu-run.log \
  hotpot-semantic-address-t12.json \
  hotpot-semantic-address-t12-shared-base.pt
do
  rsync -az -e "$rsync_ssh" \
    "$remote_host:$remote_dir/results/$artifact" "$repo_dir/results/" 2>/dev/null || true
done

exit "$remote_status"
