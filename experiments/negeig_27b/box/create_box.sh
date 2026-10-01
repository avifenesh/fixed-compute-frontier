#!/usr/bin/env bash
# create_box.sh - RIG side. Creates the two 8-GPU Nebius boxes for the Qwen3.8-27B negeig retrofit.
#
#   create_box.sh [--target b300|h200] [--count 1|2] [--names "n1 n2"] [--disk-gib N]
#                 [--preemptible|--ondemand] [--no-preflight] [--yes]
#
# Without --yes it only runs READ-ONLY checks (provider lists, subnet, image, quota records, the billing
# calculator on the exact spec) and prints the exact nebius commands and spec files it would run. Nothing is
# created. With --yes it creates the instances, waits for RUNNING and SSH, and records them in
# $STATE_DIR/instances.tsv (the only list teardown.sh and preempt_watch.sh act on).
#
# Targets (Nebius CLI, read 2026-10-01; project, subnet and profile come from common.sh's TARGETS_FILE):
#   b300 (default): uk-south1, platform gpu-b300-sxm, preset 8gpu-192vcpu-2768gb, PREEMPTIBLE (on_preemption STOP)
#   h200 (fallback): eu-north1, platform gpu-h200-sxm, preset 8gpu-128vcpu-1600gb, ON-DEMAND
#   image family ubuntu24.04-cuda13.0 (parent project-e0Xpublic-images), subnet default-subnet of the project.
#
# Preemption on Nebius is STOP: the VM is stopped, not deleted, and its managed disks stay. Bring it back with
# `nebius compute instance start --id <id>`; the public IP changes on every start. preempt_watch.sh detects it.
#
# Disk: ONE managed NETWORK_SSD boot disk per box, default 1024 GiB. Budget: 54 GB bf16 27B weights
# (Qwen/Qwen3.8-27B @ 1d4bf0f2...), up to ~270 GB for five merged eval copies, ~100 GB of GRPO LoRA
# checkpoints (about 1 GB each, every one kept), ~60 GB for the two venvs and caches, ~100 GB for data,
# rollouts and logs, plus the 40 GiB OS image. The disk is deleted with the instance (teardown.sh), so pull
# results first (teardown.sh --pull).
#
# Lane label on every instance and disk: lane=negeig-27b (CLAUDE.md: tag the pod so a sweep attributes it).
# SSH: cloud-init creates user $SSH_USER (default negeig; sudo, no password) with the rig key's public half
# (SSH_KEY, default ~/.ssh/id_ed25519). Only the public key is written into the spec.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=common.sh
. "$HERE/common.sh"

TARGET=b300; COUNT=2; NAMES=""; DISK_GIB=${DISK_GIB:-1024}; PREEMPT=auto; PREFLIGHT=1; YES=0
usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET=$2; shift 2 ;;
    --count) COUNT=$2; shift 2 ;;
    --names) NAMES=$2; shift 2 ;;
    --disk-gib) DISK_GIB=$2; shift 2 ;;
    --preemptible) PREEMPT=1; shift ;;
    --ondemand) PREEMPT=0; shift ;;
    --no-preflight) PREFLIGHT=0; shift ;;
    --yes) YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done
need nebius; need jq; need python3; need ssh-keygen
target_config "$TARGET"
[ "$PREEMPT" = auto ] && PREEMPT=$T_PREEMPTIBLE
case "$COUNT" in 1|2) ;; *) [ "${NEGEIG27_ALLOW_MORE:-0}" = 1 ] || die "--count $COUNT: this lane is two boxes; set NEGEIG27_ALLOW_MORE=1 to go beyond" ;; esac
case "$DISK_GIB" in ''|*[!0-9]*) die "--disk-gib must be a whole number of GiB" ;; esac
[ "$DISK_GIB" -ge 600 ] || die "--disk-gib $DISK_GIB is below the 600 GiB floor (weights + merged copies + checkpoints)"

if [ -z "$NAMES" ]; then
  letters=(a b c d e f g h); NAMES=""
  for ((i = 0; i < COUNT; i++)); do NAMES="$NAMES negeig27-$TARGET-${letters[$i]}"; done
fi
read -r -a NAME_ARR <<<"$NAMES"
[ "${#NAME_ARR[@]}" -eq "$COUNT" ] || die "--names gives ${#NAME_ARR[@]} names for --count $COUNT"
for n in "${NAME_ARR[@]}"; do
  [[ "$n" =~ ^negeig27-[a-z0-9][a-z0-9-]*$ ]] || die "name '$n' must match negeig27-[a-z0-9-]+ (lane prefix, so disks attribute to the lane)"
done

PUB_FILE=${SSH_PUBKEY_FILE:-$SSH_KEY.pub}
[ -r "$PUB_FILE" ] || die "no public key at $PUB_FILE (set SSH_KEY or SSH_PUBKEY_FILE)"
PUBKEY=$(head -n1 "$PUB_FILE")
[[ "$PUBKEY" =~ ^ssh-(ed25519|rsa)\  ]] || die "$PUB_FILE is not an ssh public key"
PUB_FP=$(ssh-keygen -lf "$PUB_FILE" | awk '{print $2}')

state_init

# ---------------------------------------------------------------------------------------------- spec builder
build_spec() { # name -> writes $STATE_DIR/specs/<name>.json
  NAME=$1 PROJECT=$T_PROJECT PLATFORM=$T_PLATFORM PRESET=$T_PRESET SUBNET=$T_SUBNET IMG_PARENT=$T_IMAGE_PARENT \
  IMG_FAMILY=$T_IMAGE_FAMILY DISK=$DISK_GIB PREEMPT=$PREEMPT LANE=$LANE_LABEL TARGET=$T_TARGET PUBKEY=$PUBKEY \
  SSH_USER=$SSH_USER python3 - "$STATE_DIR/specs/$1.json" <<'PY'
import json, os, sys
e = os.environ
cloud = (
    "#cloud-config\n"
    "users:\n"
    f"  - name: {e['SSH_USER']}\n"
    "    sudo: ALL=(ALL) NOPASSWD:ALL\n"
    "    shell: /bin/bash\n"
    "    ssh_authorized_keys:\n"
    f"      - {e['PUBKEY']}\n"
    "runcmd:\n"
    "  - mkdir -p /workspace\n"
    f"  - chown {e['SSH_USER']}:{e['SSH_USER']} /workspace\n"
)
labels = {"lane": e["LANE"], "box": e["NAME"], "target": e["TARGET"]}
spec = {
    "metadata": {"parent_id": e["PROJECT"], "name": e["NAME"], "labels": labels},
    "spec": {
        "hostname": e["NAME"],
        "resources": {"platform": e["PLATFORM"], "preset": e["PRESET"]},
        "boot_disk": {
            "attach_mode": "READ_WRITE",
            "managed_disk": {
                "name": e["NAME"] + "-boot",
                "spec": {
                    "size_gibibytes": e["DISK"],
                    "type": "NETWORK_SSD",
                    "source_image_family": {"image_family": e["IMG_FAMILY"], "parent_id": e["IMG_PARENT"]},
                },
            },
        },
        "network_interfaces": [
            {"subnet_id": e["SUBNET"], "name": "eth0", "ip_address": {}, "public_ip_address": {}}
        ],
        "cloud_init_user_data": cloud,
    },
}
if e["PREEMPT"] == "1":
    spec["spec"]["preemptible"] = {"on_preemption": "STOP"}
    spec["spec"]["recovery_policy"] = "FAIL"  # the provider refuses RECOVER (its default) for preemptible instances
with open(sys.argv[1], "w") as f:
    json.dump(spec, f, indent=2)
    f.write("\n")
PY
}

redacted() { # spec file -> spec JSON with the public key replaced by its fingerprint
  jq --arg fp "$PUB_FP" '.spec.cloud_init_user_data |= sub("ssh-(ed25519|rsa) [^\n]*"; "<public key " + $fp + ">")' "$1"
}

for n in "${NAME_ARR[@]}"; do build_spec "$n"; done

# ---------------------------------------------------------------------------------------------- preflight
preflight() {
  log "preflight (read-only): $T_TARGET in $T_REGION, project $T_PROJECT"
  local existing; existing=$(nb compute instance list --parent-id "$T_PROJECT" --all --format json)
  echo "  instances in the project: $(echo "$existing" | jq '[.items[]?] | length') (names: $(echo "$existing" | jq -r '[.items[]?.metadata.name] | join(",")'))"
  for n in "${NAME_ARR[@]}"; do
    echo "$existing" | jq -e --arg n "$n" '[.items[]? | select(.metadata.name == $n)] | length == 0' >/dev/null \
      || die "an instance named $n already exists in $T_PROJECT (refusing a second create)"
  done
  nb vpc subnet get --id "$T_SUBNET" --format json | jq -e '.status.state == "READY"' >/dev/null \
    || die "subnet $T_SUBNET is not READY"
  echo "  subnet $T_SUBNET READY"
  local img; img=$(nb compute image get-latest-by-family --image-family "$T_IMAGE_FAMILY" --parent-id "$T_IMAGE_PARENT" --format json) \
    || die "image family $T_IMAGE_FAMILY not found in $T_IMAGE_PARENT"
  echo "  image: $(echo "$img" | jq -r '.metadata.name + " (" + .metadata.id + "), driver " + (.metadata.labels.nvidia_gpu_drivers // "?") + ", CUDA " + (.metadata.labels.cuda_toolkit // "?")')"
  echo "$img" | jq -e --arg p "$T_PLATFORM" '(.spec.recommended_platforms // []) | index($p) != null' >/dev/null \
    && echo "  image lists $T_PLATFORM as a recommended platform" \
    || echo "  WARNING: image does not list $T_PLATFORM as recommended; accept.sh must prove the driver"
  local plat; plat=$(nb compute platform list --parent-id "$T_PROJECT" --all --format json | jq -c --arg p "$T_PLATFORM" '.items[]? | select(.metadata.name == $p)')
  [ -n "$plat" ] || die "platform $T_PLATFORM is not offered in $T_PROJECT"
  echo "$plat" | jq -e --arg r "$T_PRESET" '.spec.presets | map(.name) | index($r) != null' >/dev/null \
    || die "preset $T_PRESET is not offered on $T_PLATFORM"
  if [ "$PREEMPT" = 1 ]; then
    echo "$plat" | jq -e '.status.allowed_for_preemptibles == true' >/dev/null \
      || die "$T_PLATFORM is not allowed for preemptible instances in $T_PROJECT"
    echo "  $T_PLATFORM $T_PRESET allowed for preemptibles"
  fi
  echo "  quota records (the CLI shows usage state but no limit; sufficiency is only proven by the create):"
  nb quotas quota-allowance list --parent-id "$T_PROJECT" --all --format json \
    | jq -r --arg g "$(echo "$plat" | jq -r '.spec.gpu_count_quota_type')" \
        '.items[]? | select(.metadata.name == $g or .metadata.name == "compute.instance.preemptible.count") | "    " + .metadata.name + " " + .spec.region + " " + .status.usage_state'
  # The billing calculator parses the EXACT instance spec with the provider's own schema (field names, enum
  # values) without creating anything, so it doubles as the dry validation of the spec file.
  local est n
  for n in "${NAME_ARR[@]}"; do
    est=$(nb billing v1alpha1 calculator estimate --format json \
            "$(jq -c '{resource_spec: {compute_instance_spec: {metadata: {parent_id: .metadata.parent_id, name: .metadata.name}, spec: (.spec | del(.cloud_init_user_data))}}}' "$STATE_DIR/specs/$n.json")") \
      || die "the provider rejected the spec for $n (see the error above)"
    echo "  $n spec accepted by the provider schema; estimate $(echo "$est" | jq -r '"$" + .hourly_cost.general.total.cost_rounded + "/h"') (calculator, list price)"
  done
}
[ "$PREFLIGHT" = 1 ] && preflight

# ---------------------------------------------------------------------------------------------- the commands
echo
echo "== commands (profile $NB_PROFILE, preemptible=$PREEMPT, boot disk ${DISK_GIB} GiB NETWORK_SSD, lane=$LANE_LABEL)"
for n in "${NAME_ARR[@]}"; do
  echo "nebius --profile $NB_PROFILE compute instance create --format json --timeout 30m -f $STATE_DIR/specs/$n.json"
done
echo
for n in "${NAME_ARR[@]}"; do echo "-- $STATE_DIR/specs/$n.json (public key shown as its fingerprint)"; redacted "$STATE_DIR/specs/$n.json"; done

if [ "$YES" != 1 ]; then
  echo
  echo "DRY RUN: nothing was created. Billing starts at create (GPU hours) and the disks bill until teardown."
  echo "Re-run with --yes to create. Then: push_data.sh, bootstrap.sh and accept.sh (see README in bootstrap.sh's header)."
  exit 0
fi

# ---------------------------------------------------------------------------------------------- create
mkdir -p "$STATE_DIR/pending"
wait_running() { # id -> prints ip
  local id=$1 s i
  for i in $(seq 1 90); do
    s=$(inst_state "$id")
    case "$s" in
      RUNNING) inst_ip "$id"; return 0 ;;
      ERROR|NOT_FOUND|STOPPED) die "instance $id went to $s while starting" ;;
    esac
    sleep 10
  done
  die "instance $id not RUNNING after 15 minutes (last state $s)"
}
W_HINT=/workspace/negeig
created=()
for n in "${NAME_ARR[@]}"; do
  spec="$STATE_DIR/specs/$n.json"
  date -u +%FT%TZ >"$STATE_DIR/pending/$n"      # the create outcome can be unknown after a timeout; reconcile, never re-create blindly
  log "creating $n ..."
  # stdout only: the CLI writes progress and warnings to stderr, which is not JSON
  if out=$(nb compute instance create --format json --timeout 30m -f "$spec" 2>"$STATE_DIR/pending/$n.stderr"); then
    id=$(echo "$out" | jq -r '.metadata.id // empty' 2>/dev/null || true)
    [ -n "$id" ] || id=$(nb compute instance get-by-name --name "$n" --parent-id "$T_PROJECT" --format json 2>/dev/null | jq -r '.metadata.id // empty' || true)
    rm -f "$STATE_DIR/pending/$n.stderr"
  else
    cat "$STATE_DIR/pending/$n.stderr" >&2
    log "create returned an error for $n; reconciling from the provider list"
    id=$(nb compute instance get-by-name --name "$n" --parent-id "$T_PROJECT" --format json 2>/dev/null | jq -r '.metadata.id // empty' || true)
    [ -n "$id" ] || { echo "no instance $n exists; created so far: ${created[*]:-none}" >&2; exit 1; }
    log "adopting the instance the provider holds for $n: $id"
  fi
  [ -n "$id" ] || die "could not read the instance id for $n from the create result"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$id" "$T_PROJECT" "$T_TARGET" "$T_REGION" "$PREEMPT" "$(date -u +%FT%TZ)" >>"$TABLE"
  echo run >"$INTENT_DIR/$id"
  rm -f "$STATE_DIR/pending/$n"
  created+=("$n=$id")
  log "created $n = $id"
done

echo
log "waiting for RUNNING and SSH"
for n in "${NAME_ARR[@]}"; do
  id=$(awk -F'\t' -v n="$n" '$1==n {print $2}' "$TABLE" | tail -n1)
  ip=$(wait_running "$id")
  forget_host "$ip"
  if wait_ssh "$ip" 60; then s=ssh-ok; else s="ssh NOT ready after 10 minutes"; fi
  echo "$n  id=$id  ip=$ip  $s"
  echo "  ssh -i $SSH_KEY $SSH_USER@$ip"
done
echo
echo "Next, per box (order matters; accept before staging anything large):"
echo "  1. push_data.sh <name> --code-only   (rig -> box; the box scripts only, no data yet)"
echo "  2. ssh in and run: $W_HINT/code/experiments/negeig_27b/box/bootstrap.sh --stage base   (apt, CUDA allocation on every GPU, before any weights)"
echo "  3. push_data.sh <name>               (rig -> box; data and code, resumable)"
echo "  4. bootstrap.sh                      (venvs, model download, watchdog install)"
echo "  5. accept.sh                         (prints ACCEPT_OK or the failing step)"
echo "Confirm the boxes in the provider list any time: teardown.sh --list"
