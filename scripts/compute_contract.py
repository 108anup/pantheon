import argparse
import ast
import json
import os
import subprocess
from collections import defaultdict
from typing import Callable, NamedTuple, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

# from plot_config_light import colors, get_fig_size_paper, get_fig_size_ppt, get_style
# get_fig_size = get_fig_size_paper
# style = get_style(use_markers=False, paper=True, use_tex=True)

style = {
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "pgf.preamble": "\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{usenix}",
    "text.latex.preamble": "\\usepackage[cm]{sfmath}\n\\usepackage{amsmath}\n\\usepackage[scale=0.8]{cascadia-code}",
}


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


class Statistic(NamedTuple):
    name: str
    short_label: str
    xlabel: str


STATISTICS = {
    "loss": Statistic("loss", "loss rate", "Loss rate"),

    # "max_owd": Statistic("max_owd", "max delay", "Max delay (ms)"),
    "p50_owd": Statistic("p50_owd", "p50 delay", "p50 delay (ms)"),
    # "p90_owd": Statistic("p90_owd", "p90 delay", "p90 delay (ms)"),
    # "p95_owd": Statistic("p95_owd", "p95 delay", "p95 delay (ms)"),
    "avg_owd": Statistic("avg_owd", "avg delay", "Avg delay (ms)"),

    # "end_max_owd": Statistic("end_max_owd", "max delay", "Max delay (ms)"),
    # "end_p50_owd": Statistic("end_p50_owd", "p50 delay", "p50 delay (ms)"),
    # "end_p90_owd": Statistic("end_p90_owd", "p90 delay", "p90 delay (ms)"),
    # "end_p95_owd": Statistic("end_p95_owd", "p95 delay", "p95 delay (ms)"),
    # "end_avg_owd": Statistic("end_avg_owd", "avg delay", "Avg delay (ms)"),

}

CCA_STATISTIC = {
    "cubic": "loss",
}


def get_statistic_name(cca):
    if cca in CCA_STATISTIC:
        return CCA_STATISTIC[cca]
    return "delay"


class Contract(NamedTuple):
    label: str
    params: np.ndarray
    func: Callable
    shape_var: float
    param_var: np.ndarray


def inv_func_fit(x, c, d, e):
    return c * np.power(x - d, e)


def inv_func(x, c, d, e):
    return 96/inv_func_fit(x, c, d, e)


def contract_general(x, shape, scale, shiftx, shifty):
    return scale / np.power(x - shiftx, shape) + shifty


def contract_no_shift(x, shape, scale):
    return scale / np.power(x, shape)


def contract_no_shifty(x, shape, scale, shiftx):
    return scale / np.power(x - shiftx, shape)


def compute_contract(df: pd.DataFrame, statistic: Statistic):
    cca = df["cca"].iloc[0]
    tput = df["tput"]
    sdata = df[statistic.name]
    short_lbl = statistic.short_label
    if "owd" in statistic.name:
        sdata = sdata - df["min_owd"]

    print("Computing contract for", cca, "with statistic", statistic.name)
    tput = np.array(tput)
    sdata = np.array(sdata)

    fits = []
    for func in [contract_general, contract_no_shift, contract_no_shifty]:
        try:
            ret = scipy.optimize.curve_fit(func, sdata, tput, maxfev=10000)
            shape = ret[0][0]
            shiftx = ret[0][2] if len(ret[0]) > 2 else 0
            shape_var = ret[1][0][0]
            shape_dev = np.sqrt(shape_var)
            # label = f"Fit $\\left(\\texttt{{rate}} \\propto \\frac{{1}}{{\\texttt{{{short_lbl}}}^{{{shape:.6f} \\pm {shape_dev:.6f}}}}}\\right)$"
            label = f"Fit $\\left(\\texttt{{rate}} \\propto \\frac{{1}}{{\\texttt{{{short_lbl}}}^{{{shape:.6f}}}}}\\right)$"
            if len(ret[0]) > 2:
                label = f"Fit $\\left(\\texttt{{rate}} \\propto \\frac{{1}}{{(\\texttt{{{short_lbl}}} - {shiftx:.2f})^{{{shape:.6f}}}}}\\right)$"
                # label = f"Fit $\\left(\\texttt{{rate}} \\propto \\frac{{1}}{{(\\texttt{{{short_lbl}}} - {shiftx:.2f})^{{{shape:.6f} \\pm {shape_dev:.6f}}}}}\\right)$"
            contract = Contract(label, ret[0], func, shape_var, ret[1])
            fits.append(contract)
        except RuntimeError:
            pass
        except ValueError:
            pass

    # Pick fit with the lowest shape variance
    if len(fits) == 0:
        print("No fits found for", cca, "with statistic", statistic.name)
        return None
    fits.sort(key=lambda x: x.shape_var)
    contract = fits[0]
    return contract

    log_tput = np.log(tput)
    log_sdata = np.log(sdata)

    ret = scipy.optimize.curve_fit(func_log, log_sdata, log_tput)
    print(ret)
    opta, optb = ret[0]
    label = f"Fit $\\left(\\texttt{{rate}} \\propto \\frac{{1}}{{(\\texttt{{{short_lbl}}})^{{{opta:.6f}}}}}\\right)$"
    contract = Contract(label, ret[0], func_exp)

    try:
        p0=[100, 0, 1, 0]
        ret = scipy.optimize.curve_fit(func_fit, sdata, tput, p0=p0, maxfev=10000)
        print(ret)
        optc, optd, opte, optf = ret[0]
        label = f"Fit $\\left(\\texttt{{rate}} \\propto \\frac{{1}}{{(\\texttt{{{short_lbl}}} - {optd:.2f})^{{{opte:.6f}}}}}\\right)$"
        contract = Contract(label, ret[0], func_fit)
    except RuntimeError:
        pass

    try:
        inv_tput = 96. / tput
        ret = scipy.optimize.curve_fit(inv_func_fit, sdata, inv_tput, p0=[1, 0, 1], maxfev=10000)
        print(ret)
        optc, optd, opte = ret[0]
        label = f"Fit $\\left(\\texttt{{rate}} \\propto \\frac{{1}}{{(\\texttt{{{short_lbl}}} - {optd:.2f})^{{{opte:.2f}}}}}\\right)$"
        contract = Contract(label, ret[0], inv_func)
    except RuntimeError:
        pass

    return contract


@mpl.rc_context(rc=style)
def plot_contract(df: pd.DataFrame, outdir: str, statistic: Statistic, contract: Optional[Contract] = None):
    cca = df["cca"].iloc[0]
    tput = df["tput"]
    sdata = df[statistic.name]
    if "owd" in statistic.name:
        sdata = sdata - df["min_owd"]

    # figsize = get_fig_size_paper(0.49, 0.49)
    # figsize = (2.54, 1.49)
    # fig, ax = plt.subplots(figsize=figsize)
    fig, ax = plt.subplots()

    ax.scatter(
        sdata,
        tput,
        marker='X',
        label="Data",
        # color=colors[2]
    )

    if contract:
        x = np.linspace(min(sdata), max(sdata), 1000)
        y = contract.func(x, *contract.params)
        ax.plot(x, y, ls='dashed', label=contract.label)

    ax.set_xlabel(statistic.xlabel)
    ax.set_ylabel("Rate (Mbps)")
    ax.legend()
    ax.grid()

    fig.tight_layout(pad=0.05)
    thisdir = os.path.join(outdir, cca)
    os.makedirs(thisdir, exist_ok=True)
    fig.savefig(os.path.join(thisdir, f"{cca}-{statistic.name}-fit.pdf"))


def compute_contract_all(df, outdir):
    for cca, cca_df in df.groupby("cca"):
        # if cca == "vegas":
        #     print(cca_df)
        #     import ipdb; ipdb.set_trace()
        # compute_and_plot_contract(cca_df, outdir)
        for statistic in STATISTICS.values():
            contract = compute_contract(cca_df, statistic)
            plot_contract(cca_df, outdir, statistic, contract)


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
            # Below is done using python2 script instead
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
