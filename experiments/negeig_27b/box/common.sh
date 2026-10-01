#!/usr/bin/env bash
# Shared config and helpers for the RIG-side scripts of the negeig-27b box lane
# (create_box.sh, teardown.sh, preempt_watch.sh, push_data.sh). Sourced, never run.
#
# Everything here is read-only until a caller passes --yes to a script that says so.
# Provider facts below were read from the Nebius CLI on 2026-10-01:
#   uk-south1  gpu-b300-sxm  8gpu-192vcpu-2768gb  preemptible allowed
#   eu-north1  gpu-h200-sxm  8gpu-128vcpu-1600gb  preemptible allowed
# Image family ubuntu24.04-cuda13.0 (driver 580.x, CUDA toolkit 13.0, lists gpu-b300-sxm and gpu-h200-sxm
# as recommended platforms) is the same family the proven retrieval and Hebrew boxes booted from.
#
# Account facts stay out of the repo. TARGETS_FILE (default $STATE_DIR/targets.env) sets NB_PROFILE and, per region,
# NB_PROJECT_<REGION> and NB_SUBNET_<REGION> (region upper-cased, dashes as underscores, e.g. NB_PROJECT_EU_NORTH1).
# PROJECT_ID and SUBNET_ID override them for one run.

LANE_LABEL=${LANE_LABEL:-negeig-27b}
STATE_DIR=${NEGEIG27_STATE:-${XDG_STATE_HOME:-$HOME/.local/state}/negeig27}
TARGETS_FILE=${TARGETS_FILE:-$STATE_DIR/targets.env}
# shellcheck disable=SC1090
[ ! -f "$TARGETS_FILE" ] || . "$TARGETS_FILE"
NB_PROFILE=${NB_PROFILE:-}
TABLE=$STATE_DIR/instances.tsv        # name, id, project, target, region, preemptible, created_utc
INTENT_DIR=$STATE_DIR/intent          # one file per instance id: run | teardown
SSH_USER=${SSH_USER:-negeig}
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
# A lane-local known_hosts: Nebius recycles public IPs, and a stale entry in ~/.ssh/known_hosts under BatchMode is a hard
# "host key changed" failure that reads as an unreachable box. forget_host drops an IP's entry when the box (re)starts.
mkdir -p "$STATE_DIR"
SSH_OPTS=(-i "$SSH_KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
          -o "UserKnownHostsFile=$STATE_DIR/known_hosts" -o ServerAliveInterval=30 -o ConnectTimeout=10)
forget_host() { [ -n "${1:-}" ] && ssh-keygen -R "$1" -f "$STATE_DIR/known_hosts" >/dev/null 2>&1 || true; }

nb() { nebius ${NB_PROFILE:+--profile "$NB_PROFILE"} "$@"; }
log() { echo "[$(date -u +%FT%TZ)] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing tool on the rig: $1"; }

# target_config b300|h200 -> T_* variables. The H200 fallback is on-demand by default (owner's fallback spec).
target_config() {
  case "$1" in
    b300)
      T_REGION=uk-south1
      T_PROJECT=${PROJECT_ID:-${NB_PROJECT_UK_SOUTH1:-}}
      T_PLATFORM=gpu-b300-sxm
      T_PRESET=8gpu-192vcpu-2768gb
      T_SUBNET=${SUBNET_ID:-${NB_SUBNET_UK_SOUTH1:-}}
      T_IMAGE_PARENT=project-e03public-images
      T_PREEMPTIBLE=1 ;;
    b300w)  # second B300 source (uk-south1 had no capacity on 2026-10-01 01:30 IDT, preemptible or on demand)
      T_REGION=eu-west2
      T_PROJECT=${PROJECT_ID:-${NB_PROJECT_EU_WEST2:-}}
      T_PLATFORM=gpu-b300-sxm
      T_PRESET=8gpu-192vcpu-2768gb
      T_SUBNET=${SUBNET_ID:-${NB_SUBNET_EU_WEST2:-}}
      T_IMAGE_PARENT=project-e04public-images
      T_PREEMPTIBLE=0 ;;
    h200w)  # second H200 region: eu-north1 allows 3 public IPv4 addresses per tenant region (hit 2026-10-01)
      T_REGION=eu-west1
      T_PROJECT=${PROJECT_ID:-${NB_PROJECT_EU_WEST1:-}}
      T_PLATFORM=gpu-h200-sxm
      T_PRESET=8gpu-128vcpu-1600gb
      T_SUBNET=${SUBNET_ID:-${NB_SUBNET_EU_WEST1:-}}
      T_IMAGE_PARENT=project-e01public-images
      T_PREEMPTIBLE=1 ;;
    h200)
      T_REGION=eu-north1
      T_PROJECT=${PROJECT_ID:-${NB_PROJECT_EU_NORTH1:-}}
      T_PLATFORM=gpu-h200-sxm
      T_PRESET=8gpu-128vcpu-1600gb
      T_SUBNET=${SUBNET_ID:-${NB_SUBNET_EU_NORTH1:-}}
      T_IMAGE_PARENT=project-e00public-images
      T_PREEMPTIBLE=0 ;;
    *) die "unknown target '$1' (b300, b300w, h200 or h200w)" ;;
  esac
  [ -n "$T_PROJECT" ] && [ -n "$T_SUBNET" ] \
    || die "no project or subnet for $1 ($T_REGION): set them in $TARGETS_FILE or PROJECT_ID/SUBNET_ID (see the header)"
  T_TARGET=$1
  T_IMAGE_FAMILY=${IMAGE_FAMILY:-ubuntu24.04-cuda13.0}
}

state_init() { mkdir -p "$STATE_DIR" "$INTENT_DIR" "$STATE_DIR/specs"; touch "$TABLE"; }

# table_ids: every instance id in the table (the lane's own list; nothing else is ever touched).
table_ids() { [ -f "$TABLE" ] && awk -F'\t' 'NF>=2 {print $2}' "$TABLE" || true; }
table_field() { # id column-number
  awk -F'\t' -v id="$1" -v c="$2" '$2==id {print $c}' "$TABLE"
}

# Instance status through the CLI. A deleted instance reads NOT_FOUND. Everything else is the provider state
# (CREATING STARTING RUNNING STOPPING STOPPED DELETING UPDATING ERROR).
# stdout is the JSON only on success (a CLI warning on stderr must not corrupt the jq input); on failure the CLI's own
# error text goes to stdout with the non-zero status so inst_state can tell NOT_FOUND from a failed query.
inst_json() {
  local out err rc
  err=$(mktemp) || return 1
  out=$(nb compute instance get --id "$1" --format json 2>"$err"); rc=$?
  [ "$rc" -eq 0 ] || cat "$err"
  printf '%s\n' "$out"; rm -f "$err"; return "$rc"
}
inst_state() {
  local out; out=$(inst_json "$1") || { echo "$out" | grep -qi 'notfound\|not found' && { echo NOT_FOUND; return 0; }; echo "QUERY_FAILED"; return 0; }
  echo "$out" | jq -r '.status.state // "UNKNOWN"'
}
inst_ip() { local j; j=$(inst_json "$1") || return 0; echo "$j" | jq -r '(.status.network_interfaces[0].public_ip_address.address // empty) | sub("/.*$"; "")'; }
inst_labels_ok() { local j; j=$(inst_json "$1") || return 1; echo "$j" | jq -e --arg l "$LANE_LABEL" '.metadata.labels.lane == $l' >/dev/null; }

# Every instance this lane owns, read from the provider by label (not from the local table), for the
# confirm-from-the-list checks. Args: project-id. Prints "id<TAB>name<TAB>state".
provider_lane_instances() {
  nb compute instance list --parent-id "$1" --all --format json \
    | jq -r --arg l "$LANE_LABEL" '.items[]? | select(.metadata.labels.lane == $l) | [.metadata.id, .metadata.name, .status.state] | @tsv'
}
provider_lane_disks() {
  nb compute disk list --parent-id "$1" --all --format json \
    | jq -r --arg l "$LANE_LABEL" '.items[]? | select((.metadata.labels.lane == $l) or (.metadata.name | startswith("negeig27-"))) | [.metadata.id, .metadata.name] | @tsv'
}

# resolve_ip <name|id|ipv4> -> the box's CURRENT public IPv4 (it changes on every start after a preemption).
resolve_ip() {
  local a=$1 id ip
  if [[ $a =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then echo "$a"; return 0; fi
  id=$(awk -F'\t' -v a="$a" '$1==a || $2==a {print $2; exit}' "$TABLE" 2>/dev/null)
  [ -n "$id" ] || { echo "no instance '$a' in $TABLE" >&2; return 1; }
  ip=$(inst_ip "$id")
  [ -n "$ip" ] || { echo "instance $id has no public IP (state $(inst_state "$id"))" >&2; return 1; }
  echo "$ip"
}
# wait_ssh <ip> <polls>: poll until the box answers ssh (10 s apart).
wait_ssh() {
  local ip=$1 n=${2:-60} i
  for ((i = 0; i < n; i++)); do
    ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" true 2>/dev/null && return 0
    sleep 10
  done
  return 1
}
