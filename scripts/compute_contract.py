import argparse
import ast
import json
import os
import subprocess
from collections import defaultdict
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

# from plot_config_light import colors, get_fig_size_paper, get_fig_size_ppt, get_style
# get_fig_size = get_fig_size_paper
# style = get_style(use_markers=False, paper=True, use_tex=True)

FILE_PATH = os.path.abspath(os.path.realpath(__file__))  # PROJECT_ROOT/scripts/script.py
SCRIPTS_ROOT = os.path.dirname(FILE_PATH)  # PROJECT_ROOT/scripts/
PANTHEON_ROOT = os.path.dirname(SCRIPTS_ROOT)  # PROJECT_ROOT/
ANALYZE_PY = os.path.join(PANTHEON_ROOT, "src", "analysis", "analyze.py")


def func_fit(x, c, d, e, f):
    return c / np.power(x - d, e) + f


def func_log(x, a, b):
    return -a * x + b


def func_exp(x, a, b):
    return np.power(np.e, b) * np.power(x, -a)


def is_loss_based(cca):
    return cca in ["reno", "cubic", "fillp", "fillp_sheep"]


style = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.preamble": "\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{usenix}",
    "text.latex.preamble": "\\usepackage[cm]{sfmath}\n\\usepackage{amsmath}\n\\usepackage[scale=0.8]{cascadia-code}",
}
@mpl.rc_context(rc=style)
def compute_and_plot_contract(df, outdir):
    cca = df["cca"].iloc[0]
    tput = df["tput"]
    delay = df["delay"]
    loss = df["loss"]

    # TODO: get rtprop in a better way
    rtprop = 6
    # delay = delay - rtprop

    loss_based = False
    short_lbl = "owd"
    if is_loss_based(cca):
        rtprop = 0
        delay = loss
        loss_based = True
        short_lbl = "loss rate"

    tput = np.array(tput)
    delay = np.array(delay)
    loss = np.array(loss)

    log_tput = np.log(tput)
    log_delay = np.log(delay)

    print("Computing contract for ", cca)
    ret = scipy.optimize.curve_fit(func_log, log_delay, log_tput)
    print(ret)
    opta, optb = ret[0]

    fit = False
    try:
        p0=[100, rtprop, 1, 0]
        ret = scipy.optimize.curve_fit(func_fit, delay, tput, p0=p0, maxfev=10000)
        print(ret)
        optc, optd, opte, optf = ret[0]
        fit = True
    except RuntimeError:
        pass

    # figsize = get_fig_size_paper(0.49, 0.49)
    # figsize = (2.54, 1.49)
    # fig, ax = plt.subplots(figsize=figsize)
    fig, ax = plt.subplots()

    ax.scatter(
        delay,
        tput,
        marker='X',
        label="Data",
        # color=colors[2]
    )

    x = np.linspace(min(delay), max(delay), 1000)

    y = func_exp(x, opta, optb)
    label = f"Fit $\\left(\\texttt{{rate}} \\propto \\frac{{1}}{{(\\texttt{{{short_lbl}}})^{{{opta:.6f}}}}}\\right)$"
    if fit:
        y = func_fit(x, optc, optd, opte, optf)
        label = f"Fit $\\left(\\texttt{{rate}} \\propto \\frac{{1}}{{(\\texttt{{{short_lbl}}} - {optd:.2f})^{{{opte:.6f}}}}}\\right)$"
    ax.plot(x, y, ls='dashed', label=label)

    ax.set_xlabel("One way delay (owd) (ms)")
    ax.set_ylabel("Rate (Mbps)")
    if loss_based:
        ax.set_xlabel("Loss rate")
    ax.legend()
    ax.grid()

    fig.tight_layout(pad=0.05)
    fig.savefig(os.path.join(outdir, cca + "-fit.pdf"))


def compute_contract_all(df, outdir):
    for cca, cca_df in df.groupby("cca"):
        # if cca == "vegas":
        #     print(cca_df)
        #     import ipdb; ipdb.set_trace()
        compute_and_plot_contract(cca_df, outdir)


def try_except(function: Callable):
    """This function is useful for debugging in python. If we call a function
    through `try_except` or decorate a function with the `try_except_wrapper`,
    then we get a python interactive debugger anytime an exception is raised in
    the function.

    Examples
    --------
    >>> def f():
    ...     raise ValueError
    >>> try_except(f)
    Traceback (most recent call last):
    ...
    ipdb>

    >>> @try_except_wrapper
    ... def f():
    ...     raise ValueError
    >>> f()
    Traceback (most recent call last):
    ...
    ipdb>
    """
    try:
        return function()
    except Exception:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)


def try_except_wrapper(function):
    """See documentation of `try_except`"""

    def func_to_return(*args, **kwargs):
        def func_to_try():
            return function(*args, **kwargs)
        return try_except(func_to_try)
    return func_to_return


def parse_literal(element: str):
    """Converts string to literal if possible, else returns the string

    Examples
    --------
    >>> parse_literal("1.0")
    1.0
    >>> parse_literal("1")
    1
    >>> type(parse_literal("1"))
    <class 'int'>
    >>> type(parse_literal("1.0"))
    <class 'float'>
    """
    if element == "":
        return element
    try:
        return ast.literal_eval(element)
    except ValueError:
        return element


def parse_exp_raw(exp_tag):
    ret = {}
    for param_tag in exp_tag.split('-'):
        param_name = param_tag.split('[')[0]
        param_val = param_tag.split('[')[1][:-1]
        ret[param_name] = parse_literal(param_val)
    return ret


def create_df(input_dir, exts=(".log")):
    exp_dirs = defaultdict(list)
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(exts):
                fpath = os.path.join(root, filename)
                exp_dirs[root].append(fpath)

    records = []
    for exp_dir in exp_dirs.keys():
        perf_json = os.path.join(exp_dir, "pantheon_perf.json")
        if not os.path.exists(perf_json):
            continue
            cmd = [ANALYZE_PY, "--data-dir", exp_dir]
            # print(cmd)
            subprocess.check_call(cmd)

        fp = open(perf_json, 'r')
        perf_dict = json.load(fp)
        fp.close()

        for cca, cca_dict in perf_dict.items():
            for run, run_dict in cca_dict.items():
                for flow, flow_dict in run_dict.items():
                    exp_params = parse_exp_raw(os.path.basename(exp_dir))
                    # if flow == "all" or exp_params["n_flows"] == 1:
                    if flow == "all":
                        continue
                    record = {
                        "cca": cca,
                        "run": run,
                        "flow": flow,
                        "exp_dir": exp_dir,
                    }
                    record.update(flow_dict)
                    record.update(exp_params)
                    records.append(record)

    df = pd.DataFrame(records)
    return df


@try_except_wrapper
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', required=True,
        type=str, action='store',
        help='path to pantheon runs')
    parser.add_argument(
        '-o', '--output', default="",
        type=str, action='store',
        help='path output figures')
    args = parser.parse_args()

    assert os.path.isdir(args.input)
    os.makedirs(args.output, exist_ok=True)
    assert os.path.isdir(args.output)

    df = create_df(args.input)
    compute_contract_all(df, args.output)


if __name__ == "__main__":
    main()
