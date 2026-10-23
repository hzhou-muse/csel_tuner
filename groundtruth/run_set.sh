#!/bin/bash
# Run the launches of a plan set within a time budget.
#
#   ./run_set.sh PLAN_DIR OUT_DIR BUDGET_SECONDS
#
# For each line of PLAN_DIR/launches.txt (id nodes ppn alg idle plan), in
# order: start mpisrv in batch mode with MPIR_CVAR_BCAST_INTRA_ALGORITHM=alg
# and the idle settings, writing OUT_DIR/<id>.jsonl. The idle field is
# MODE[:IDLE_NS[:SPINUP_US]] -> MPISRV_IDLE, MPISRV_IDLE_NS, MPISRV_SPINUP_US
# (omitted parts keep mpisrv's defaults). Each launch gets the
# remaining budget minus LAUNCH_MARGIN as its --deadline; launches that no
# longer fit (< MIN_LAUNCH seconds left) are skipped. Re-running resumes:
# finished launches are skipped and unfinished ones continue (--resume).
# OUT_DIR/launches.log gets one JSON line per launch (times, exit code).
set -u
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../launch/config.sh"
plans=$1 outdir=$2 budget=$3
mkdir -p "$outdir"
t0=$(date +%s)
log=$outdir/launches.log

{
    echo "# $(date -Is) host=$(hostname) budget=$budget"
    echo "# PBS_JOBID=${PBS_JOBID:-} nodes:"; sort -u "${PBS_NODEFILE:-/dev/null}" | sed 's/^/#   /'
    echo "# mpichversion:"; mpichversion 2>&1 | sed 's/^/#   /'
    echo "# fi_info --version:"; (fi_info --version 2>&1 || true) | sed 's/^/#   /'
    echo "# modules:"; (module list 2>&1 || true) | sed 's/^/#   /'
    echo "# env:"; env | grep -E '^(MPIR_|MPICH|FI_|PALS|MPISRV)' | sort | sed 's/^/#   /'
} >> "$outdir/environment.txt"

grep -v '^#' "$plans/launches.txt" | while read -r id nodes ppn alg idle plan; do
    [ -z "$id" ] && continue
    if [ -e "$outdir/$id.done" ]; then
        echo "skip $id (done)"; continue
    fi
    left=$((budget - ($(date +%s) - t0)))
    if [ "$left" -lt "$MIN_LAUNCH" ]; then
        echo "skip $id (only ${left}s left)"
        echo "{\"id\":\"$id\",\"skipped\":true,\"left_s\":$left}" >> "$log"
        continue
    fi
    deadline=$((left - LAUNCH_MARGIN))
    echo "$(date +%T) launch $id: nodes=$nodes ppn=$ppn alg=$alg idle=$idle deadline=${deadline}s"
    # The JSONL is appended on resume, so only lines after $before belong to this run.
    before=$(cat "$outdir/$id.jsonl" 2>/dev/null | wc -l)
    start=$(date +%s.%N)
    IFS=: read -r imode ins ispin <<< "$idle"
    env MPIR_CVAR_BCAST_INTRA_ALGORITHM=$alg MPISRV_IDLE=$imode \
        ${ins:+MPISRV_IDLE_NS=$ins} ${ispin:+MPISRV_SPINUP_US=$ispin} \
        timeout $((deadline + LAUNCH_MARGIN / 2)) \
        bash -c "$(declare -f launch); NODE_CORES='$NODE_CORES' launch $nodes $ppn $MPISRV $MAX_M \
            --plan $plan --out $outdir/$id.jsonl --deadline $deadline --resume" \
        > "$outdir/$id.stdout" 2> "$outdir/$id.stderr" < /dev/null
    rc=$?
    end=$(date +%s.%N)
    n=$(grep -c '"status":"ok"' "$outdir/$id.jsonl" 2>/dev/null); n=${n:-0}
    hit=$(tail -n +$((before + 1)) "$outdir/$id.jsonl" 2>/dev/null | grep -c '"status":"deadline"')
    echo "{\"id\":\"$id\",\"nodes\":$nodes,\"ppn\":$ppn,\"alg\":\"$alg\",\"idle\":\"$idle\",\"start\":$start,\"end\":$end,\"rc\":$rc,\"ok\":$n,\"deadline_hit\":$hit}" >> "$log"
    echo "$(date +%T)   rc=$rc ok=$n deadline_hit=$hit"
    [ "$rc" -eq 0 ] && [ "$hit" -eq 0 ] && touch "$outdir/$id.done"
done
echo "$(date +%T) set finished after $(( $(date +%s) - t0 ))s"
