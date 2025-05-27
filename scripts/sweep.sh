#!/usr/bin/env bash

{
  set -euo pipefail
  # set -x

  echo "outdir: " $outdir

  SCRIPT=$(realpath "$0")
  SCRIPT_PATH=$(dirname "$SCRIPT")
  PANTHEON_PATH=$(realpath $SCRIPT_PATH/../)
  MAHIMAHI_TRACES=/home/anupa/Projects/fairCC/experiments/mahimahi-traces/

  MM_PKT_SIZE=1504

  run_experiment() {
    echo "bw_ppms: $bw_ppms"
    echo "ow_delay_ms: $ow_delay_ms"
    echo "buf_size_bdp: $buf_size_bdp"
    echo "n_flows: $n_flows"

    bdp_pkts=$((2 * $bw_ppms * $ow_delay_ms))
    buf_size_bytes=$(($buf_size_bdp * $bdp_pkts * $MM_PKT_SIZE))
    tag=bw_ppms[${bw_ppms}]-ow_delay_ms[${ow_delay_ms}]-buf_size_bdp[${buf_size_bdp}]-n_flows[${n_flows}]
    this_outdir=$outdir/$tag

    cmd="${PANTHEON_PATH}/src/experiments/test.py local \
      ${SCHEMES} \
      --data-dir ${this_outdir} \
      --prepend-mm-cmds \"mm-delay ${ow_delay_ms}\" \
      --uplink-trace ${MAHIMAHI_TRACES}/${bw_ppms}ppms.trace \
      --downlink-trace ${MAHIMAHI_TRACES}/${bw_ppms}ppms.trace \
      --extra-mm-link-args \"--uplink-queue=droptail --uplink-queue-args=bytes=${buf_size_bytes}\" \
      --flows ${n_flows} \
      --runtime ${RUNTIME} \
      --interval ${INTERVAL}"
    echo $cmd
    eval $cmd
  }

  RUNTIME=60
  INTERVAL=0
  SCHEMES="--all"
  SCHEMES="--schemes=\"bbr vegas copa sprout ledbat pcc_experimental vivace pcc cubic indigo fillp fillp_sheep taova\""
  # SCHEMES="--schemes=\"bbr fillp fillp_sheep indigo vegas\""
  # SCHEMES="--schemes=\"bbr\""
  bw_ppms=8
  ow_delay_ms=20
  buf_size_bdp=4
  n_flows=2

  jobs=()

  # for bw_ppms in $(seq 1 8); do
  #   run_experiment &
  #   jobs+=($!)
  # done

  for bw_ppms in 2 4 8; do
    # The parallelization needs to incorporate flow count and BDP.
    for n_flows in $(seq 2 6); do
    # for n_flows in 7 8; do
      run_experiment &
      jobs+=($!)
    done
    wait "${jobs[@]}"
  done

  exit 0
}
