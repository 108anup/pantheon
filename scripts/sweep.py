#!/usr/bin/env python

import pipes
import argparse
import os
import subprocess
from multiprocessing.pool import ThreadPool


SCRIPT_PATH = os.path.abspath(os.path.realpath(__file__))
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
PANTHEON_PATH = os.path.dirname(SCRIPT_DIR)
MAHIMAHI_TRACES = '/home/anupa/Projects/fairCC/experiments/mahimahi-traces/'
MM_PKT_SIZE = 1504


RUNTIME = 60
INTERVAL = 0
SCHEMES = '--schemes=bbr vegas copa sprout ledbat pcc_experimental vivace pcc cubic indigo fillp fillp_sheep taova'


def get_cmd(bw_ppms, ow_delay_ms, buf_size_bdp, n_flows, outdir):
    print("bw_ppms: {}".format(bw_ppms))
    print("ow_delay_ms: {}".format(ow_delay_ms))
    print("buf_size_bdp: {}".format(buf_size_bdp))
    print("n_flows: {}".format(n_flows))

    bdp_pkts = 2 * bw_ppms * ow_delay_ms
    buf_size_bytes = buf_size_bdp * bdp_pkts * MM_PKT_SIZE
    tag = "bw_ppms[{}]-ow_delay_ms[{}]-buf_size_bdp[{}]-n_flows[{}]".format(
        bw_ppms, ow_delay_ms, buf_size_bdp, n_flows)
    this_outdir = os.path.join(outdir, tag)

    cmd = [
        os.path.join(PANTHEON_PATH, 'src/experiments/test.py'), 'local',
        SCHEMES,
        '--data-dir', this_outdir,
        '--prepend-mm-cmds', 'mm-delay {}'.format(ow_delay_ms),
        '--uplink-trace', os.path.join(MAHIMAHI_TRACES, '{}ppms.trace'.format(bw_ppms)),
        '--downlink-trace', os.path.join(MAHIMAHI_TRACES, '{}ppms.trace'.format(bw_ppms)),
        '--extra-mm-link-args', '--uplink-queue=droptail --uplink-queue-args=bytes={}'.format(buf_size_bytes),
        '--flows', str(n_flows),
        '--runtime', str(RUNTIME),
        '--interval', str(INTERVAL)
    ]
    return cmd


def work(cmd):
    print("Running: {}".format(" ".join([pipes.quote(s) for s in cmd])))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, help="Base output directory")
    args = parser.parse_args()

    print("outdir: {}".format(args.outdir))

    bw_ppms = 4
    ow_delay_ms = 20
    buf_size_bdp = 1
    n_flows = 2

    pool = ThreadPool(4)
    for bw_ppms in range(2, 9):
        for n_flows in range(2, 9):
            cmd = get_cmd(bw_ppms, ow_delay_ms, buf_size_bdp, n_flows, args.outdir)
            pool.apply_async(work, (cmd,))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()

