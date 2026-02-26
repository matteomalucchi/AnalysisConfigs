# HEPPlotter

`HEPPlotter` is a modular and flexible plotting helper built around `mplhep` and `matplotlib`.
It standardizes the creation of publication-quality plots for High Energy Physics (HEP) analyses.

Supports:

* ✅ 1D histograms (with ratio plots)
* ✅ 2D histograms (heatmaps)
* ✅ Graphs with error bars (e.g. efficiency or response curves)

> [!WARNING]
>
> ### 🧾 Disclaimer
>
> 📘 This documentation and example set were generated with the assistance of ChatGPT (GPT-5), so there may be errors and inconsistencies w.r.t. the actual `HEPPlotter` class  

---

## 📦 Requirements

```bash
pip install matplotlib mplhep numpy scipy hist
```

---

## 🚀 Quick Start

### 📊 Example: 1D Histogram with Ratio Plot

```python
from hep_plotter import HEPPlotter
from hist import Hist
import numpy as np

# Example: make two 1D histograms
data = Hist.new.Reg(20, 0, 100, name="pt").Weight()
mc = Hist.new.Reg(20, 0, 100, name="pt").Weight()
weights = np.random.uniform(0.5, 1.5, 1200)

data.fill(np.random.normal(50, 10, 1000))
mc.fill(np.random.normal(48, 12, 1200), weight=weights)

# Define styles
series_dict = {
    "Data": {
        "data": data,
        "style": {"color": "black", "histtype": "errorbar", "is_reference": True},
    },
    "MC": {
        "data": mc,
        "style": {"color": "red", "histtype": "step", "legend_name": "Simulation"},
    },
}

# Build and run the plot
(
    HEPPlotter("CMS")
    .set_output("plots/pt_distribution")
    .set_labels(xlabel="pT [GeV]", ylabel="Events")
    .set_data(series_dict, plot_type="1d")
    .set_options(y_log=False, legend_loc="upper right")
    .add_annotation(
        x=0.05, y=0.9, s="Example plot", fontsize=14, color="gray"
    )
    .add_line("h", y=0.5, color="gray", linestyle="--", linewidth=1)
    .show()  # show interactively (optional)
    .run()
)
```

This produces `plots/pt_distribution.png` (and also `.pdf`, `.svg` by default).

### 🟩🟥🟦 Example: 2D Histogram (HeatMap)

This example shows how to plot a **2D histogram** (e.g. recoil vs. qT, response vs. true momentum, etc.)
It supports linear or log color scales and customizable colorbars.

```python
import numpy as np
from hist import Hist
from hep_plotter import HEPPlotter

# Example: build a 2D histogram of (true_qT, response)
np.random.seed(0)
true_qT = np.random.exponential(scale=50, size=10_000)
response = np.random.normal(1.0, 0.1, size=10_000)
weights = np.random.uniform(0.5, 1.5, size=10_000)

# Create a hist.Hist object
hist2d = (
    Hist.new
    .Reg(40, 0, 200, name="true_qT", label=r"Z $q_\mathrm{T}$ [GeV]")
    .Reg(40, 0.5, 1.5, name="response", label="u$_\parallel$/q$_T$")
    .Weight()
)
hist2d.fill(true_qT, response, weight=weights)

# Define input dictionary for the HEPPlotter
series_dict = {
    "ResponseMap": {
        "data": hist2d,
        "style": {
            "cmap": "viridis",    # colormap
            "vmin": 0,            # color scale min
            "vmax": None,         # auto
        },
    },
}

(
    HEPPlotter("CMS")
    .set_output("plots/response_map")
    .set_labels(
        xlabel=r"Z $q_\mathrm{T}$ [GeV]",
        ylabel=r"$u_\parallel / q_\mathrm{T}$",
        cbar_label="Events"
    )
    .set_data(series_dict, plot_type="2d")
    .set_options(cbar_log=False, grid=False)
    .add_annotation(
        x=0.05, y=0.95,
        s="Run 3 simulation",
        fontsize=16, color="black"
    )
    .run()
)
```

📄 **Output files:**

```bash
plots/response_map.png
plots/response_map.pdf
plots/response_map.svg
```

🧠 **Notes:**

* You can switch to log color scale with `.set_options(cbar_log=True)`
* If your histogram has weights, `mplhep.hist2dplot()` automatically normalizes them unless otherwise specified.
* The `cbar_label` controls the colorbar title.

---

### 📈 Example: Graph Plotting (Efficiency / Response Curve)

This example shows how to plot **graphs with error bars**, such as efficiencies, resolutions, or scale vs. qT.
You can also plot multiple graphs on the same axes.

```python
import numpy as np
from hep_plotter import HEPPlotter

# Example: plot mean response vs. qT bin center
qT_bins = np.linspace(0, 200, 11)
qT_centers = 0.5 * (qT_bins[:-1] + qT_bins[1:])
mean_response = 1.0 + 0.05 * np.sin(qT_centers / 40)
response_unc = 0.02 * np.sqrt(qT_centers / 100 + 1)

# Another curve for comparison
mean_response_alt = 1.02 + 0.04 * np.cos(qT_centers / 50)
response_unc_alt = 0.015 * np.ones_like(qT_centers)

# Prepare data for HEPPlotter
series_dict = {
    "Data": {
        "data": {"x": (qT_centers, np.zeros_like(qT_centers)),
                 "y": (mean_response, response_unc)},
        "style": {"color": "black", "fmt": "o", "label": "Data"},
    },
    "MC": {
        "data": {"x": (qT_centers, np.zeros_like(qT_centers)),
                 "y": (mean_response_alt, response_unc_alt)},
        "style": {"color": "red", "fmt": "s", "label": "MC prediction"},
    },
}

(
    HEPPlotter("CMS")
    .set_output("plots/response_curve")
    .set_labels(
        xlabel=r"Z $q_\mathrm{T}$ [GeV]",
        ylabel=r"$\langle u_\parallel / q_\mathrm{T} \rangle$"
    )
    .set_data(series_dict, plot_type="graph")
    .set_options(
        legend_loc="lower right",
        grid=True
    )
    .add_annotation(
        x=0.05, y=0.95,
        s="Response vs. qT",
        fontsize=14,
        color="gray"
    )
    .add_line("h", y=1.0, color="gray", linestyle="--", linewidth=1)
    .run()
)
```

📄 **Output files:**

```bash
plots/response_curve.png
plots/response_curve.pdf
plots/response_curve.svg
```

🧠 **Notes:**

* Each graph entry in `series_dict` must have `"data": {"x": (values, errors), "y": (values, errors)}`.
* `"fmt"` controls the matplotlib marker format (e.g., `"o"`, `"s"`, `"^"`).
* Use `.add_line("h", y=1.0)` to mark reference values.
* Supports multiple graphs with different styles.

---

## 🧠 Design Philosophy

* Modular configuration with chainable `.set_...()` methods
* Supports ratio subplots automatically when a reference histogram is present
* Handles stacked plots, color schemes, legends, and CMS labels out-of-the-box
* Parallelization-friendly: instantiate one plot per process
* Works seamlessly with `hist` objects from `boost-histogram`

---

## ⚙️ Class Overview

### 🏗️ Constructor

```python
HEPPlotter(style="CMS", debug=False)
```

### Arguments

* `style`: mplhep style name (e.g. `"CMS"`, `"ATLAS"`, `"LHCb"`)
* `debug`: print internal steps (default `False`)

---

## 🧩 Configuration Methods

Each method returns `self`, so you can chain calls.

| Method                                                                          | Description                                                              |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `set_plot_config(figsize=None, lumitext="(13.6 TeV)", formats=None)`            | Sets figure size, CMS lumi text, and output formats (`png`, `pdf`, etc.) |
| `set_output(output_base)`                                                       | Defines output path without extension (e.g. `"plots/pt"`)                |
| `set_labels(xlabel, ylabel="Events", cbar_label="Events", ratio_label="Ratio")` | Sets axis and colorbar labels                                            |
| `set_data(series_dict, plot_type="1d")`                                         | Provides plotting data. Supports `"1d"`, `"2d"`, `"graph"`               |
| `set_extra_kwargs(**kwargs)`                                                    | Passes extra arguments directly to plotting functions                    |
| `set_options(**kwargs)`                                                         | Configures log scales, legends, grids, limits, etc. (see below)          |
| `show()`                                                                        | Displays plot interactively instead of saving it                         |
| `add_ratio_hists(ratio_hists)`                                                  | Adds precomputed ratio histograms                                        |
| `add_annotation(**kwargs)`                                                      | Adds text annotations (`ax.text`) with free positioning                  |
| `add_line(orientation="h", **kwargs)`                                           | Adds horizontal (`"h"`) or vertical (`"v"`) lines                        |
| `add_chi_square(pred_unc=False, **kwargs)`                                      | Computes and displays χ²/ndof and p-value                                |
| `run()`                                                                         | Executes the plotting                                                    |

---

## 🧾 Configurable Options (via `set_options()`)

You can pass multiple options together:

```python
.set_options(y_log=True, legend_loc="upper left", grid=False)
```

| Option               | Type  | Default  | Description                            |
| -------------------- | ----- | -------- | -------------------------------------- |
| `y_log`, `x_log`     | bool  | `False`  | Logarithmic axes                       |
| `y_log_ratio`        | bool  | `False`  | Logarithmic scale for ratio subplot    |
| `cbar_log`           | bool  | `False`  | Logarithmic colorbar for 2D histograms |
| `legend`             | bool  | `True`   | Show legend                            |
| `split_legend`       | bool  | `True`   | Two-column legend when many entries    |
| `legend_loc`         | str   | `"best"` | Legend position on main plot           |
| `legend_ratio`       | bool  | `False`  | Show legend in ratio plot              |
| `legend_ratio_loc`   | str   | `"best"` | Legend position for ratio plot         |
| `grid`               | bool  | `True`   | Show gridlines                         |
| `set_ylim`           | bool  | `True`   | Automatically scale y-limits           |
| `ylim_top_factor`    | float | `1.7`    | Scale factor for y-max                 |
| `ylim_bottom_factor` | float | `1e-2`   | Scale factor for y-min                 |
| `reference_to_den`   | bool  | `True`   | In ratio: divide by reference or not   |

---

## 🧮 Data Structure Examples

### 1D Histograms

```python
series_dict = {
    "Data": {"data": hist_data, "style": {"color": "black", "is_reference": True}},
    "MC": {"data": hist_mc, "style": {"color": "red", "histtype": "step"}},
}
```

**Optional keys in `style`:**

* `"color"` / `"facecolor"` / `"edgecolor"`
* `"histtype"`: `"step"`, `"fill"`, `"errorbar"`
* `"stack"`: `True` or `False`
* `"plot_errors"`: whether to show error bars
* `"bin_edges_plotting"`: array of custom bin edges (for visual representation only)
* `"legend_name"` / `"legend_name_ratio"`: custom labels
* `"appear_in_legend"`: include/exclude from legend

---

### 2D Histograms

```python
series_dict = {
    "response": {
        "data": hist2d,
        "style": {"cmap": "viridis"},
    }
}
```

---

### Graphs

```python
series_dict = {
    "efficiency": {
        "data": {"x": ([bins_center], [x_err]), "y": ([eff], [eff_err])},
        "style": {"color": "blue", "fmt": "o", "markersize": 6},
    }
}
```

---

## 🧾 Annotations and Lines

```python
.add_annotation(
    x=0.05,
    y=0.9,
    s=r"$\chi^2$/ndof=1.23, p=0.78$",
    fontsize=16,
    color="blue",
)
.add_line("v", x=50, color="gray", linestyle="--", linewidth=1)
```

You can add multiple annotations or lines; they are all drawn sequentially.

---

## 🧪 Chi-Square Display

To automatically compute and show χ²/ndof and p-value between reference and target histograms:

```python
.add_chi_square(
    pred_unc=True,
    x=0.05,
    y=0.95,
    fontsize=16,
    color="red",
)
```

---

## 🧰 Parallel Execution Example

The class is **multiprocessing-friendly** — simply create one instance per plot:

```python
from multiprocessing import Pool

jobs = []
for category in categories:
    plotter = (
        HEPPlotter()
        .set_output(f"plots/{category}")
        .set_labels(xlabel="qT [GeV]", ylabel="Events")
        .set_data(series_dicts[category])
    )
    jobs.append(plotter)

with Pool(8) as pool:
    pool.map(lambda p: p.run(), jobs)
```

Each process runs its own plot cleanly.

---

## 🧾 Output

By default, each plot is saved as:

```bash
{output_base}.png
{output_base}.pdf
{output_base}.svg
```

You can customize formats with:

```python
.set_plot_config(formats=["png"])
```

---

## 🧱 Internal Features

* Uses `mplhep` for styling and CMS labels
* Automatically adds “CMS Preliminary” + luminosity text
* Preserves variances when visually rebinned
* Supports stacked histograms and split legends
* Can add multiple ratio histograms or precomputed ratio curves
* Clears internal state after each plot (safe for multiprocessing)


---

## 🧑‍🔬 Tips

* To **just view** without saving, call `.show().run()`
* To **customize labels/fonts**, use standard `matplotlib.rcParams`
* Use `.set_extra_kwargs()` to forward any argument to `mplhep.histplot()` or `ax.plot()`
