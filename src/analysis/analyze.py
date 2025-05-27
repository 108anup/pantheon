#!/usr/bin/env python

import multiprocessing
import os
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from os import path

import arg_parser
import context

from helpers.subprocess_wrappers import check_call


def get_exp_dirs(input_dir, exts=('.log')):
    exp_dirs = defaultdict(list)
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(exts):
                fpath = os.path.join(root, filename)
                exp_dirs[root].append(fpath)
    return list(exp_dirs.keys())


def work(exp_dir, args):
    analysis_dir = path.join(context.src_dir, 'analysis')
    plot = path.join(analysis_dir, 'plot.py')
    report = path.join(analysis_dir, 'report.py')

    plot_cmd = ['python', plot]
    report_cmd = ['python', report]

    for cmd in [plot_cmd, report_cmd]:
        if args.data_dir:
            cmd += ['--data-dir', exp_dir]
        if args.schemes:
            cmd += ['--schemes', args.schemes]
        if args.include_acklink:
            cmd += ['--include-acklink']

    print("Running: ", plot_cmd)
    check_call(plot_cmd)
    check_call(report_cmd)
    return None


def main():
    args = arg_parser.parse_analyze()

    assert os.path.isdir(args.data_dir)
    exp_dirs = get_exp_dirs(args.data_dir)
    print("Analyzing: ", exp_dirs)

    pool = ThreadPool(processes=multiprocessing.cpu_count()//4)
    for exp_dir in exp_dirs:
        pool.apply_async(work, (exp_dir, args))
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
