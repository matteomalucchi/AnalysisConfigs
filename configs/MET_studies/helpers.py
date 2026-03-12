import numpy as np
import itertools
import re
from coffea.util import save
from hist import Hist

def profile2d_to_graph_dict(h_profile, fixed_ax, free_ax, fixed_bin_idx, style):
        """
        Slice a 2D profile along fixed_ax at fixed_bin_idx and return a
        graph data dict (same format as responses_dict entries) along free_ax.

        Parameters
        ----------
        h_profile : Hist
            2D profile with .errors and .has_nan_mask attributes.
        fixed_ax : Axis
            The axis being fixed (sliced).
        free_ax : Axis
            The axis kept free (becomes the x-axis of the graph).
        fixed_bin_idx : int
            Bin index along fixed_ax to slice at.
        style : dict
            Plotting style to attach.

        Returns
        -------
        dict with keys "data" and "style", matching responses_dict format.
        """
        # Slice values and errors along the fixed axis
        if h_profile.axes[0].name == fixed_ax.name:
            values = h_profile.view()[fixed_bin_idx, :]
            errors = h_profile.errors[fixed_bin_idx, :]
            mask   = getattr(h_profile, "has_nan_mask",
                             np.zeros_like(values, dtype=bool))[fixed_bin_idx, :]
        else:
            values = h_profile.view()[:, fixed_bin_idx]
            errors = h_profile.errors[:, fixed_bin_idx]
            mask   = getattr(h_profile, "has_nan_mask",
                             np.zeros_like(values, dtype=bool))[:, fixed_bin_idx]

        centers    = (free_ax.edges[1:] + free_ax.edges[:-1]) / 2.0
        half_widths = (free_ax.edges[1:] - free_ax.edges[:-1]) / 2.0

        # Replace masked (empty) bins with NaN
        values = np.where(mask, np.nan, values)
        errors = np.where(mask, np.nan, errors)

        return {
            "data": {
                "x": [centers.tolist(), half_widths.tolist()],
                "y": [values.tolist(), errors.tolist()],
            },
            "style": style,
        }



def profile_to_hist2d(h_profile, x_ax, y_ax):
        """
        Convert a 2-axis profile (Double() storage + .errors) into a 2D
        Hist with Weight() storage so HEPPlotter can render it as a
        standard 2D histogram.
        The Weight() view has shape (*bins, 2): [..., 0] = value,
        [..., 1] = variance. We store the metric value in [0] and the
        squared error in [1] so that the colorbar shows the metric and
        the rendered uncertainties are consistent.
        """
        h2 = (
            Hist.new
            .Var(x_ax.edges, name=x_ax.name, label=x_ax.label, flow=False)
            .Var(y_ax.edges, name=y_ax.name, label=y_ax.label, flow=False)
            .Weight()
        )
        h2.style = h_profile.style

        values = h_profile.view()                        # shape (nx, ny)
        errors = h_profile.errors                        # shape (nx, ny)
        mask   = getattr(h_profile, "has_nan_mask",
                         np.zeros_like(values, dtype=bool))

        # Replace empty bins with 0 for the hist view (mask stored separately)
        vals_clean = np.where(mask, 0.0, values)
        errs_clean = np.where(mask, 0.0, errors)

        h2.view()["value"]    = vals_clean
        h2.view()["variance"] = errs_clean ** 2          # variance = sigma^2

        return h2



def iter_slice_combinations(hist, free_axes, also_project=True):
    """
    Yield slice/projection combinations for axes NOT in free_axes.

    Two modes are yielded for each fixed axis combination:

    1. SLICE   — all extra axes are fixed to a single bin index.
                 slice_dict contains {axis_name: bin_index}.
    2. PROJECT — one extra axis is fixed to a single bin index while all
                 other extra axes are summed over (projected out).
                 slice_dict contains {axis_name: bin_index} for the fixed
                 axis and {axis_name: slice(None)} (sum) for the others.
                 Only yielded when also_project=True and there are at least
                 2 fixed axes (otherwise it is identical to mode 1).

    Parameters
    ----------
    hist : Hist
        The histogram object.
    free_axes : list of str
        Axis names to keep free (not sliced over).
    also_project : bool, optional
        If True (default), also yield projected combinations.

    Yields
    ------
    slice_dict : dict
        Keys are axis names of the fixed axes.
        Values are either a bin index (int, fix to that bin) or
        slice(None) (sum over all bins in that axis).
    label_parts : list of str
        Human-readable strings for the fixed bins,
        e.g. ["0.0 < qt < 30.0"].
    suffix : str
        Short unique string to append to output filenames,
        e.g. "slice_qt0_nvtx2" or "proj_qt0".
    mode : str
        Either "slice" or "project", so callers can handle each differently
        (e.g. different output subdirectory).
    """
    fixed_axes = [ax for ax in hist.axes if ax.name not in free_axes]

    if not fixed_axes:
        yield {}, [], "", "slice"
        return

    # ------------------------------------------------------------------
    # Mode 1 — SLICE: fix every extra axis to one bin
    # ------------------------------------------------------------------
    ranges = [range(len(ax)) for ax in fixed_axes]
    for combo in itertools.product(*ranges):
        slice_dict   = {}
        label_parts  = []
        suffix_parts = []
        for ax, idx in zip(fixed_axes, combo):
            slice_dict[ax.name] = idx
            lo, hi = ax.edges[idx], ax.edges[idx + 1]
            label_parts.append(f"{lo:.4g} < {ax.label} < {hi:.4g}")
            suffix_parts.append(f"{ax.name}{lo:.2g}_{hi:.2g}")
        yield slice_dict, label_parts, "slice_" + "_".join(suffix_parts), "slice"

    # ------------------------------------------------------------------
    # Mode 2 — PROJECT: fix one extra axis to one bin, sum over the rest.
    # Only meaningful when there are >= 2 fixed axes.
    # ------------------------------------------------------------------
    if not also_project: # or len(fixed_axes) < 2:
        return

    if len(fixed_axes) >= 2:
        for fixed_dim, fixed_ax in enumerate(fixed_axes):
            for idx in range(len(fixed_ax)):
                slice_dict   = {}
                label_parts  = []
                suffix_parts = []

                for d, ax in enumerate(fixed_axes):
                    if d == fixed_dim:
                        # Fix this axis to a single bin
                        slice_dict[ax.name] = idx
                        lo, hi = ax.edges[idx], ax.edges[idx + 1]
                        label_parts.append(f"{lo:.4g} < {ax.label} < {hi:.4g}")
                        suffix_parts.append(f"{ax.name}{lo:.2g}_{hi:.2g}")
                    else:
                        # Sum (project) over this axis
                        slice_dict[ax.name] = slice(None)

                yield (
                    slice_dict,
                    label_parts,
                    "proj_" + "_".join(suffix_parts),
                    "project",
                )
    else:
        # project over the single fixed axis
        slice_dict   = {}
        label_parts  = []
        suffix_parts = [fixed_axes[0].name]
        slice_dict[fixed_axes[0].name] = slice(None)
        
        yield (
            slice_dict,
            label_parts,
            "proj_" + "_".join(suffix_parts),
            "project",
        )
            
def is_profile(hist):
    """Return True if hist is a pre-computed profile (has .errors attribute)."""
    return hasattr(hist, "errors")



def extract_labels(var_name, response_var_name_dict, bin_variables):
    """
    Extract y and x labels from a histogram key of the form
    'y_metric VS x0 VS x1 VS ...' (N binning dimensions).

    Parameters
    ----------
    var_name : str
        Histogram key, e.g. 'u_perp_meanVSqtVSnvtx'.

    Returns
    -------
    y_label : str
        Label for the y (response) variable.
    x_labels : list of str
        Labels for each binning dimension, in the same order as they appear
        in the key (i.e. innermost/last binning axis first.
    """
    parts = var_name.split("VS")
    y_var_name   = parts[0]
    x_var_names  = parts[1:]   # one entry per binning dimension

    y_label = (
        y_var_name
        if y_var_name not in response_var_name_dict
        else response_var_name_dict[y_var_name]
    )

    x_labels = []
    for x_var_name in x_var_names:
        matches = [
            v["label"]
            for v in bin_variables.values()
            if v["name_plot"] == x_var_name
        ]
        if matches:
            x_labels.append(matches[0])
        else:
            # Fallback: use the raw name if not found in BIN_VARIABLES
            x_labels.append(x_var_name)

    return y_label, x_labels

def save_dict_to_file(dict_to_save, output_path):
    """Save multiple dictionaries to a single coffea file.

    Parameters
    ----------
    dict_to_save : dict
        Dictionary to save.
    output_path : str
        Output file path.
    """

    save(dict_to_save, output_path)
    print(f"Saved histograms to {output_path}")


def extract_year_tag(s):
    match = re.search(r"(19|20)\d{2}(?:_[A-Za-z0-9]+)?", s)
    return match.group() if match else None


def run_plot(plotter):
    """Run a HEPPlotter instance."""
    plotter.run()


def weighted_mean(x, w):
    """
    Compute the weighted mean and its standard error.

    Parameters
    ----------
    x : array-like
        Input values.
    w : array-like
        Weights associated with the values.

    Returns
    -------
    mean_w : float
        Weighted mean of x.
    sem_w : float
        Standard error of the weighted mean, accounting for effective statistics.
    """
    mean_w = np.average(x, weights=w)
    n_eff = (np.sum(w)) ** 2 / np.sum(w**2)  # Effective number of entries
    variance_w = np.sum(w * (x - mean_w) ** 2) / np.sum(w)
    sem_w = np.sqrt(variance_w / n_eff)
    return mean_w, sem_w


def weighted_std_dev(x, w):
    """
    Compute the weighted standard deviation and its uncertainty.

    Parameters
    ----------
    x : array-like
        Input values.
    w : array-like
        Weights associated with the values.

    Returns
    -------
    std_w : float
        Weighted standard deviation of x.
    std_err_w : float
        Error estimate of the weighted standard deviation.
    """
    mean_w = np.average(x, weights=w)
    var_w = np.sum(w * (x - mean_w) ** 2) / np.sum(w)
    std_w = np.sqrt(var_w)

    n_eff = (np.sum(w) ** 2) / np.sum(w**2)
    std_err_w = std_w / np.sqrt(2 * (n_eff - 1))
    return std_w, std_err_w
