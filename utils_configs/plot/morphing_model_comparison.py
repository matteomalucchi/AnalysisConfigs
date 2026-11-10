"""Compare two background-morphing ONNX models (e.g. the mean and the median
over the k-folds of the morphing weight) in plot_2bMorphedvs4b.py.

The postW categories of the coffea output contain the morphing weight of the
model used in the PocketCoffea run, and the DNN input variables of the
morphing. Here both models are evaluated again on those inputs, so the two
weights can be compared without rerunning PocketCoffea.
"""

import os

import numpy as np
from hist import Hist

from utils_configs.plot.HEPPlotter import HEPPlotter

# Suffixes added to the name of a postW category to get the same events
# reweighted with the first (average) and second (median) model.
MODEL_SUFFIXES = ("_avgModel", "_medModel")
MODEL_LABELS = {"_avgModel": "mean of weights", "_medModel": "median of weights"}
MODEL_COLORS = {"_avgModel": "red", "_medModel": "blue"}


def model_suffix(cat):
    """Suffix of a model category ('_avgModel' / '_medModel'), else ''."""
    for suffix in MODEL_SUFFIXES:
        if cat.endswith(suffix):
            return suffix
    return ""


def strip_model_suffix(cat):
    """Name of the category in the coffea file for a model category."""
    suffix = model_suffix(cat)
    return cat[: -len(suffix)] if suffix else cat


def evaluate_morphing_weight(session, input_name, output_name, input_columns):
    """Weight of one event per entry, from the ONNX model session.

    input_columns: list of 1D arrays, in the order of the model inputs.
    The k-fold models (mean or median over the folds of the weight) have one
    output, the weight.
    """
    inputs = np.stack(
        [np.asarray(col, dtype=np.float32) for col in input_columns], axis=-1
    )
    outputs = session.run(output_name, {input_name[0]: inputs})[0]
    outputs = np.asarray(outputs)
    if outputs.ndim > 1:
        if outputs.shape[1] != 1:
            raise ValueError(
                f"The morphing model must output one weight per event, got an "
                f"output of shape {outputs.shape}"
            )
        outputs = outputs[:, 0]
    return outputs.astype(np.float64)


def compare_with_stored(label, w_model, w_stored):
    """Print how well the re-evaluated weight reproduces the weight stored in
    the coffea file (the model of the PocketCoffea run has to match one)."""
    finite = np.isfinite(w_model) & np.isfinite(w_stored)
    if not finite.any():
        print(f"[{label}] no finite weights to compare with the stored ones")
        return
    diff = np.abs(w_model[finite] - w_stored[finite])
    rel = diff / np.maximum(np.abs(w_stored[finite]), 1e-12)
    print(
        f"[{label}] vs weight stored in the file: "
        f"{np.mean(rel < 1e-4) * 100:.2f}% of the events agree within 1e-4, "
        f"max relative difference {rel.max():.3g}, mean {rel.mean():.3g}"
    )


def _log_bins(values, n_bins=100):
    positive = values[values > 0]
    if positive.size == 0:
        return None
    log_min = np.floor(np.log10(positive.min()))
    log_max = np.ceil(np.log10(positive.max()))
    if log_max <= log_min:
        log_max = log_min + 1
    return np.logspace(log_min, log_max, n_bins)


def _plot(hist_dict, output, xlabel, x_log, annotation, lumitext, cmstext, legend=True, annotation_x=0.05):
    p = (
        HEPPlotter()
        .set_plot_config(lumitext=lumitext, cmstext=cmstext, figsize=[13, 13])
        .set_output(output)
        .set_labels(xlabel, "Events")
        .set_options(y_log=True, x_log=x_log, set_ylim=False, legend=legend)
        .set_data(hist_dict, plot_type="1d")
        .add_annotation(
            x=annotation_x, y=0.95, s=annotation, fontsize=20, ha="left", va="top"
        )
    )
    p.run()


def plot_weight_comparison(
    w_avg, w_med, name, outputdir, lumi, era_string, cmstext="Private"
):
    """Distribution of the weights of the two models (log-x and linear-x),
    and of the per-event ratio median/mean of the weights."""
    os.makedirs(outputdir, exist_ok=True)
    lumitext = f"{era_string}, {lumi}" + r" $fb^{-1}$, (13.6 TeV)"

    weights = {}
    for suffix, w in zip(MODEL_SUFFIXES, (w_avg, w_med)):
        w = np.asarray(w, dtype=np.float64)
        bad = ~np.isfinite(w)
        if bad.any():
            print(f"[{name}] {bad.sum()} non finite weights removed for {suffix}")
        weights[suffix] = w[~bad]

    # --- the weights of the two models
    annotation = "\n".join(
        f"{MODEL_LABELS[s]}: mean {w.mean():.2f}, std {w.std():.2f}"
        for s, w in weights.items()
    )
    for axis_scale in ("log", "linear"):
        all_w = np.concatenate(list(weights.values()))
        if axis_scale == "log":
            bins = _log_bins(all_w)
            if bins is None:
                print(f"WARNING: no positive weights for {name}, skipping log-x plot")
                continue
        else:
            if all_w.max() <= all_w.min():
                print(f"WARNING: degenerate range for {name}, skipping linear-x plot")
                continue
            bins = np.linspace(all_w.min(), all_w.max(), 100)
        hist_dict = {}
        for suffix, w in weights.items():
            h = Hist.new.Var(bins, name="Morphing weights", flow=False).Double()
            h.fill(w)
            hist_dict[MODEL_LABELS[suffix]] = {
                "data": h,
                "style": {
                    "histtype": "step",
                    "color": MODEL_COLORS[suffix],
                    "linewidth": 2,
                },
            }
        _plot(
            hist_dict,
            os.path.join(outputdir, f"weights_avg_vs_med_{name}_{axis_scale}x"),
            "Morphing weights",
            axis_scale == "log",
            annotation,
            lumitext,
            cmstext,
        )

    # --- per-event ratio median / mean
    w_a = np.asarray(w_avg, dtype=np.float64)
    w_m = np.asarray(w_med, dtype=np.float64)
    ok = np.isfinite(w_a) & np.isfinite(w_m) & (w_a > 0)
    if not ok.any():
        print(f"WARNING: no events with a positive mean weight for {name}, skipping ratio")
        return
    ratio = w_m[ok] / w_a[ok]
    annotation = (
        f"median of ratio: {np.median(ratio):.3f}\n"
        f"16%-84%: {np.percentile(ratio, 16):.3f} - {np.percentile(ratio, 84):.3f}\n"
        f"events with w_med == 0: {np.mean(ratio == 0) * 100:.2f}%"
    )
    for axis_scale in ("log", "linear"):
        if axis_scale == "log":
            bins = _log_bins(ratio)
            if bins is None:
                continue
        else:
            hi = np.percentile(ratio, 99.9)
            if hi <= 0:
                continue
            bins = np.linspace(0, hi, 100)
        h = Hist.new.Var(bins, name="w_median / w_mean", flow=False).Double()
        h.fill(ratio)
        _plot(
            {
                "median / mean of weights": {
                    "data": h,
                    "style": {"histtype": "step", "color": "black", "linewidth": 2},
                }
            },
            os.path.join(outputdir, f"weights_ratio_med_over_avg_{name}_{axis_scale}x"),
            r"$w_{median}\,/\,w_{mean}$ per event",
            axis_scale == "log",
            annotation,
            lumitext,
            cmstext,
            legend=False,
            annotation_x=0.45,
        )
