import matplotlib.pyplot as plt
import matplotlib
from hist import Hist
import matplotlib.ticker as mtick
from scipy.stats.distributions import chi2

import numpy as np
import mplhep as hep
import os


class HEPPlotter:
    """
    A helper class for standardized plotting in HEP analyses using `mplhep`.

    Supports:
    - 1D histograms with optional ratio plots
    - 2D histograms
    - Graph plotting (values with error bars)

    Usage
    -----
    job = (HEPPlotter("CMS")
            .set_output("out/plot")
            .set_labels(xlabel="pT [GeV]", ylabel="Events")
            .set_data(series_dict)
            .set_options(y_log=True, ratio_label="Data/MC"))
    job.run()  # produces the plot

    Parameters
    ----------
    style : str
        The mplhep style to use (default "CMS").
    debug : bool
        If True, print debug information during plotting (default False).
    -----------

    Methods
    -------
    - set_plot_config(figsize=None, lumitext="(13.6 TeV)", cmstext="Preliminary", formats=None)
    - set_output(output_base)
    - set_labels(xlabel, ylabel="Events", cbar_label="Events", ratio_label="Ratio")
    - set_data(series_dict, plot_type="1d")
    - set_extra_kwargs(**kwargs)
    - set_options(**kwargs)
    - add_ratio_hists(ratio_hists)
    - add_annotation(**kwargs)
    - add_chi_square(**kwargs)
    - add_line(orientation="h", **kwargs)
    - run()
    """

    def __init__(self, style="CMS", debug=False):
        # core settings
        self.style = style
        self.debug = debug
        hep.style.use(style)

        # plot config
        self.figsize = None
        self.lumitext = "(13.6 TeV)"
        self.cmstext = "Preliminary"
        self.data_formats = ["png", "pdf", "svg"]

        # output
        self.output_base = None

        # show plot interactively (for debugging)
        self.show_plot = False

        # inputs
        self.series_dict = None
        self.plot_type = "1d"  # "1d", "2d", "graph"

        # labels
        self.xlabel = None
        self.ylabel = "Events"
        self.cbar_label = "Events"
        self.ratio_label = "Ratio"

        # extra kwargs for plotting functions
        self.extra_kwargs = {}

        # --- ATTRIBUTES THAT CAN BE SET WITH set_options ---
        self._configurable_options = {
            ## log scales
            "y_log": False,
            "x_log": False,
            "y_log_ratio": False,
            "cbar_log": False,
            ## legend
            "legend": True,
            "split_legend": True,
            "legend_loc": "best",
            "legend_ratio": False,
            "legend_ratio_loc": "best",
            ## y lim
            "set_ylim": True,
            "ylim_top_factor": 1.7,
            "ylim_bottom_factor": 1e-2,
            ## other
            "reference_to_den": True,
            "grid": True,
        }

        # expose as attributes too (so they're accessible normally)
        for key, val in self._configurable_options.items():
            setattr(self, key, val)

        # internal
        self._plot_chi_square = None
        self._ratio_hists = {}
        self._annotations = []
        self._lines = []

        self._change_histogram_binning = False

    # ----------------------------
    # CONFIGURATION METHODS
    # ----------------------------

    def set_plot_config(
        self, figsize=None, lumitext="(13.6 TeV)", cmstext="Preliminary", formats=None
    ):
        """Set the plotting style and related options."""
        self.figsize = figsize
        self.lumitext = lumitext
        self.cmstext = cmstext
        if formats:
            self.data_formats = formats
        return self

    def set_output(self, output_base, create_dir=False):
        """Set the base name for output files (without extension) and optionally create the output directory."""
        # remove the extension if provided
        self.output_base = os.path.splitext(output_base)[0]
        self.create_dir=create_dir
        return self

    def set_labels(
        self, xlabel, ylabel="Events", cbar_label="Events", ratio_label="Ratio"
    ):
        """Set the x and y axis labels."""
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cbar_label = cbar_label
        self.ratio_label = ratio_label
        return self

    def set_data(self, series_dict, plot_type="1d"):
        """Provide the plotting data (dict structure varies by plot_type).
        series_dict structure:
        - For 1D histograms:
            series_dict : dict
                Dictionary mapping names to {name:{"data": hist.Hist, "style": dict}}.
                To plot the ratio plot, the reference histogram must include {"is_reference": True} inside the style dict.
        - For 2D histograms:
            series_dict : dict
                {name:{"data": hist.Hist, "style": dict}}
        - For graphs:
            series_dict : dict
                Dictionary mapping names to {name:{
                                                "data": {
                                                    "x":[list, list],
                                                    "y":[list, list],
                                                },
                                                "style": dict
                                                }
                                            }.
        """
        self.series_dict = series_dict
        self.plot_type = plot_type
        return self

    def set_extra_kwargs(self, **kwargs):
        """Extra kwargs passed to the plotting functions."""
        self.extra_kwargs.update(kwargs)
        return self

    def set_options(self, **kwargs):
        """Generic options setter for configurable attributes."""
        for key, value in kwargs.items():
            if key in self._configurable_options:
                self._configurable_options[key] = value
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown option '{key}'")
        return self

    def show(self):
        """Display the plot interactively (for debugging)."""
        self.show_plot = True
        return self

    def add_ratio_hists(self, ratio_hists):
        """Add precomputed ratio histograms to be plotted on the ratio subplot.
        ratio_hists: dict of histograms with the same keys as series_dict
        """
        self._ratio_hists = ratio_hists
        return self

    def add_annotation(self, x, y, s, **kwargs):
        """
        Add a text annotation to the plot.

        Parameters
        ----------
        x, y : float
            Coordinates of the annotation. By default, interpreted as relative
            (0–1) coordinates within the axes (like ax.transAxes).
        s : str
            Text to display.
        coord_type : {"axes", "data", "figure"}, optional (in kwargs)
            Coordinate system for annotation position.
            - "axes": relative to plot area (default) -> ax.transAxes
            - "data": use data coordinates (axis values) -> ax.transData
            - "figure": relative to the whole figure -> ax.figure.transFigure
        **kwargs : dict
            Additional arguments passed directly to `ax.text()`,
            e.g. color, fontsize, ha, va, coord_type, etc.
        """
        coord_type = kwargs.pop("coord_type", "axes")  # safely extract if present

        self._annotations.append(
            {
                "x": x,
                "y": y,
                "s": s,
                "coord_type": coord_type,
                "kwargs": kwargs,
            }
        )
        return self

    def add_chi_square(self, pred_unc=False, **kwargs):
        """Add chi-square text to the plot (only for 1D with ratio).
        pred_unc: if True, include the prediction uncertainty in the chi-square calculation
        kwargs: passed directly to ax.text()
        """
        self._plot_chi_square = True
        self._chi_square_add_prediction_uncertainty = pred_unc
        self._chi_square_style = kwargs
        return self

    def add_line(self, orientation="h", **kwargs):
        """Add a horizontal or vertical line to the plot.
        orientation: 'h' for horizontal, 'v' for vertical
        kwargs: passed directly to ax.axhline or ax.axvline
        """
        self._lines.append((orientation, kwargs))
        return self

    # ----------------------------
    # INTERNAL HELPERS
    # ----------------------------

    def _close_fig(self, fig):
        """Close the figure to free memory."""
        plt.close(fig)

    def _save(self, fig):
        """Save the figure in all specified formats."""
        if self.output_base:
            for ext in self.data_formats:
                fig.savefig(f"{self.output_base}.{ext}", bbox_inches="tight", dpi=300)

    def _apply_cms_labels(self, ax):
        """Add CMS style labels to the plot."""
        hep.cms.lumitext(self.lumitext, ax=ax)
        hep.cms.text(self.cmstext, ax=ax)

    def _apply_annotations(self, ax):
        """Internal helper to draw all stored annotations on a given axis."""
        for ann in self._annotations:
            coord_type = ann.get("coord_type", "axes")
            transform = {
                "axes": ax.transAxes,
                "data": ax.transData,
                "figure": ax.figure.transFigure,
            }.get(coord_type, ax.transAxes)

            ax.text(
                ann["x"],
                ann["y"],
                ann["s"],
                transform=transform,
                **ann["kwargs"],
            )

    def _apply_lines(self, ax):
        """Add all stored horizontal/vertical lines to the axes."""
        for orient, kwargs in self._lines:
            if orient == "h":
                ax.axhline(**kwargs)
            else:
                ax.axvline(**kwargs)

    def _apply_chi_square(self, ax, hist_1d, ref_hist, index, style):
        """Compute and add chi-square text to the plot."""

        num = (hist_1d.values() - ref_hist.values()) ** 2
        den = ref_hist.variances() + (
            hist_1d.variances() if self._chi_square_add_prediction_uncertainty else 0
        )

        chi2_value = np.sum(
            np.divide(
                num,
                den,
                out=np.zeros_like(num, dtype=float),
                where=den > 0,
            )
        )

        ndof = len(hist_1d.values()) - 1

        chi2_norm = chi2_value / (ndof if ndof > 0 else 1)

        pvalue = chi2.sf(chi2_value, ndof)

        self.chi_square_text = (
            r"$\chi^2$/ndof= {:.3f},".format(chi2_norm) + f"  p-value= {pvalue:.3f}"
        )

        color_chi2 = self._chi_square_style.get(
            "color",
            style.get("color", style.get("edgecolor", style.get("facecolor"))),
        )

        # plot the chi2 text
        ax.text(
            self._chi_square_style.get("x", 0.05),
            self._chi_square_style.get("y", 0.95) - index * 0.05,
            self.chi_square_text,
            transform=ax.transAxes,
            fontsize=self._chi_square_style.get("fontsize", 20),
            color=color_chi2,
        )

    def _color_handler(self, histtype, style, kwargs, use_lists=False):
        if use_lists:
            if histtype == "fill":
                if "edgecolor" not in kwargs:
                    kwargs.update(
                        {
                            "edgecolor": [style.get("edgecolor", style.get("color"))],
                            "facecolor": [style.get("facecolor", style.get("color"))],
                            "alpha": [style.get("alpha", 0.5)],
                        }
                    )
                else:
                    kwargs["edgecolor"].append(
                        style.get("edgecolor", style.get("color"))
                    )
                    kwargs["facecolor"].append(
                        style.get("facecolor", style.get("color"))
                    )
                    kwargs["alpha"].append(style.get("alpha", 0.5))
            else:
                if "color" not in kwargs:
                    kwargs.update(
                        {
                            "color": [style.get("color")],
                        }
                    )
                else:
                    kwargs["color"].append(style.get("color"))
        else:
            if histtype == "fill":
                kwargs.update(
                    {
                        "edgecolor": style.get("edgecolor", style.get("color")),
                        "facecolor": style.get("facecolor", style.get("color")),
                        "alpha": style.get("alpha", 0.5),
                    }
                )
            else:
                kwargs.update(
                    {
                        "color": style.get("color"),
                    }
                )

    def _stack_plot_order(self, hist_1d, style):
        # reorder the histograms to be plotted in stack order (first is bottom)
        if isinstance(hist_1d, list):
            # get the sorting indexes
            idxes = sorted(
                range(len(hist_1d)),
                key=lambda i: hist_1d[i].integrate(name=hist_1d[i].axes[0].name).value,
            )
            hist_1d = [hist_1d[i] for i in idxes]
            # orer the elements in the style dict if they are lists
            for key in style:
                if isinstance(style[key], list):
                    style[key] = [style[key][i] for i in idxes]

        return hist_1d, style

    # ----------------------------
    # IMPLEMENTATIONS
    # ----------------------------

    def _plot_1d(self):
        """Plot 1D histograms with optional ratio plot."""
        ratio_plot, ref_name = self._validate_inputs(self.series_dict)
        fig, ax, ax_ratio = self._create_figure(ratio_plot)

        ref_hist = self.series_dict[ref_name]["data"] if ref_name else None

        for index, (name, props) in enumerate(self.series_dict.items()):

            hist_1d = props["data"]
            style = props.get("style", {})
            hist_1d, style = self._stack_plot_order(hist_1d.copy(), style.copy())

            is_ref = style.get("is_reference", False)

            legend_name = (
                style.get("legend_name", name)
                if style.get("appear_in_legend", True)
                else None
            )
            legend_name_ratio = (
                style.get("legend_name_ratio", name)
                if style.get("appear_in_legend_ratio", True)
                else None
            )

            kwargs = self.extra_kwargs.copy()
            histtype = style.get("histtype", "step")
            stack = style.get("stack", False)
            kwargs.update(
                {
                    "histtype": histtype,
                    "linewidth": style.get("linewidth", 2),
                    "stack": stack,
                }
            )

            self._color_handler(histtype, style, kwargs)

            # rebin for plotting if requested
            bin_edges_plotting = style.get("bin_edges_plotting", None)
            if bin_edges_plotting is not None:
                hist_1d = self._set_plotting_binning(hist_1d, bin_edges_plotting)
                # rebin also the reference histogram if not already done
                if index == 0:
                    ref_hist = self._set_plotting_binning(ref_hist, bin_edges_plotting)

            # draw histogram
            self._plot_histogram(
                ax,
                legend_name,
                hist_1d,
                style.get("plot_errors", True),
                **kwargs,
            )

            # if stack, sum the histograms
            if isinstance(hist_1d, list):
                hist_1d = sum(hist_1d)

            # if stack, keep the style of the last histogram in the list
            for key in style:
                if isinstance(style[key], list):
                    style[key] = style[key][-1]

            if self._plot_chi_square and ratio_plot and not is_ref:
                self._apply_chi_square(ax, hist_1d, ref_hist, index, style)

            # ratio
            if ratio_plot and ax_ratio is not None:
                if self.reference_to_den:
                    self._plot_ratio(
                        ax_ratio,
                        hist_1d,
                        ref_hist,
                        legend_name_ratio,
                        is_ref,
                        style,
                    )
                else:
                    self._plot_ratio(
                        ax_ratio,
                        ref_hist,
                        hist_1d,
                        legend_name_ratio,
                        is_ref,
                        style,
                    )

        # plot precomputed ratio hists
        if self._ratio_hists and ratio_plot and ax_ratio is not None:
            for name, props in self._ratio_hists.items():
                style = self._ratio_hists[name].get("style", {})
                is_ref = self._ratio_hists[name]["style"].get("is_reference", False)
                ratio_hist = props["data"]
                legend_name_ratio = (
                    style.get("legend_name_ratio", name)
                    if style.get("appear_in_legend_ratio", True)
                    else None
                )
                self._plot_ratio(
                    ax_ratio,
                    None,
                    None,
                    legend_name_ratio,
                    is_ref,
                    style,
                    ratio_hist=ratio_hist,
                )

        self._finalize(fig, ax, ax_ratio)

    def _plot_2d(self):
        """Plot 2D histograms."""
        fig, ax, _ = self._create_figure()
        for name, props in self.series_dict.items():
            if not isinstance(props["data"], Hist) or props["data"].ndim != 2:
                raise ValueError(
                    f"Expected 2D hist.Hist for {name}, got {type(props['data'])} with ndim={props['data'].ndim}"
                )
            hist2d = props["data"]
            label = props["style"].get("label", "")
            hep.hist2dplot(
                hist2d,
                ax=ax,
                label=label,
                norm=matplotlib.colors.LogNorm() if self.cbar_log else None,
                cmap=props["style"].get("cmap", "viridis"),
                **self.extra_kwargs,
            )
        self._finalize(fig, ax)

    def _plot_graph(self):
        """Plot graphs with error bars."""
        fig, ax, _ = self._create_figure()
        for name, props in self.series_dict.items():
            y_values, y_errors = (
                props["data"]["y"][0],
                props["data"]["y"][1],
            )
            x_values, x_errors = (
                props["data"]["x"][0],
                props["data"]["x"][1],
            )

            style = props.get("style", {})
            # extra_kwargs = self.extra_kwargs.copy()
            # extra_kwargs.update(
            #     {
            #         "color": style.get("color"),
            #         "markersize": style.get("markersize", 6),
            #     }
            # )
            # hep.histplot(
            #     (y_values, x_values),
            #     yerr=y_errors,
            #     xerr=x_errors,
            #     histtype="errorbar",
            #     # fmt=style.get("fmt", "o"),
            #     label=name,
            #     color=style.get("color"),
            #     markersize=style.get("markersize"),
            #     **self.extra_kwargs,
            # )
            legend_name = (
                style.get("legend_name", name)
                if style.get("appear_in_legend", True)
                else None
            )
            if np.any(np.array(x_errors) > 0) or np.any(np.array(y_errors) > 0):
                # plot with error bars
                ax.errorbar(
                    x=x_values,
                    y=y_values,
                    xerr=x_errors,
                    yerr=y_errors,
                    fmt=style.get("fmt", "o"),
                    label=legend_name,
                    color=style.get("color"),
                    markersize=style.get("markersize"),
                    **self.extra_kwargs,
                )
            else:
                # plot a curve or a graph without errors
                ax.plot(
                    x_values,
                    y_values,
                    style.get("fmt", "o"),
                    label=legend_name,
                    color=style.get("color"),
                    markersize=style.get("markersize"),
                    **self.extra_kwargs,
                )
        self._finalize(fig, ax)

    # ----------------------------
    # UTILITIES
    # ----------------------------

    def _validate_inputs(self, series_dict):
        """Validate the input series_dict for 1D plotting."""
        ratio_plot = False
        ref_name = None
        for name, props in series_dict.items():
            # check that the props has only the "data" and "style" keys
            if not all(key in ["data", "style"] for key in props.keys()):
                raise ValueError(
                    f"Invalid keys in series_dict for {name}. Expected only 'data' and 'style', got {list(props.keys())}. The provided key should probably be inside the 'style' dictionary."
                )
                
            hist_1d = props["data"]
            if not isinstance(hist_1d, Hist) and not isinstance(hist_1d[0], Hist):
                raise ValueError(f"Expected hist.Hist for {name}, got {type(hist_1d)}")
            if props.get("style", {}).get("is_reference", False):
                if ratio_plot:
                    raise ValueError("Multiple reference histograms found.")
                ratio_plot = True
                ref_name = name
            if isinstance(hist_1d[0], Hist):
                style = props.get("style", {})
                # check that the lists of histograms have the same dimension
                lenght_hists = len(hist_1d)
                for key in style:
                    if isinstance(style[key], list):
                        if len(style[key]) != lenght_hists:
                            raise ValueError(
                                f"Length mismatch in style lists for {name}: expected {lenght_hists}, got {len(style[key])} for key {key}"
                            )

        return ratio_plot, ref_name

    def _create_figure(self, ratio_plot=False):
        """Create figure and axes, with optional ratio subplot."""
        if ratio_plot:
            fig, (ax, ax_ratio) = plt.subplots(
                2,
                1,
                figsize=self.figsize,
                sharex=True,
                gridspec_kw={"height_ratios": [2.5, 1]},
                constrained_layout=True,
            )
            return fig, ax, ax_ratio
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            return fig, ax, None

    def _check_bin_consistency(self, hist_1d, bin_edges_plotting):
        """Check that the provided bin edges for plotting are consistent with the histogram."""
        if bin_edges_plotting is not None:
            if isinstance(hist_1d, list):
                for i in range(len(hist_1d)):
                    self._check_bin_consistency(hist_1d[i], bin_edges_plotting[i])

                return

            hist_bins = hist_1d.axes[0].edges
            if not np.all(np.diff(bin_edges_plotting) > 0):
                raise ValueError(
                    "Provided bin_edges_plotting must be strictly increasing"
                )
            # check that the provided bin edges have the same length
            if len(bin_edges_plotting) != len(hist_bins):
                raise ValueError(
                    f"Provided bin_edges_plotting {bin_edges_plotting} must have the same number of edges as histogram bins {hist_bins}"
                )
            self._change_histogram_binning = True

    def _set_plotting_binning(self, hist_1d, bin_edges_plotting):
        self._check_bin_consistency(hist_1d, bin_edges_plotting)
        if not self._change_histogram_binning:
            return hist_1d

        # project histogram values into new bins (for plotting only)
        if isinstance(hist_1d, list):
            histplots = []
            for i in range(len(hist_1d)):
                histplots.append(
                    self._set_plotting_binning(hist_1d[i], bin_edges_plotting[i])
                )
            return histplots

        counts = hist_1d.values()

        # replace only for plotting
        histplot = Hist.new.Var(
            bin_edges_plotting, name=hist_1d.axes[0].name, flow=False
        ).Weight()
        bin_edges_plotting_centers = (
            bin_edges_plotting[1:] + bin_edges_plotting[:-1]
        ) / 2
        histplot.fill(bin_edges_plotting_centers, weight=counts)
        # handle variances properly,
        histplot.variances()[:] = hist_1d.variances()

        return histplot

    def _plot_histogram(self, ax, name, hist_1d, plot_errors, **kwargs):
        """Plot a single 1D histogram on the given axes."""

        # The errorbars are computed as sqrt(w2) taking
        # the weights from hist.variances() without the w2 argument
        hep.histplot(
            hist_1d,
            w2method="sqrt" if plot_errors else None,
            label=name,
            yerr=False if not plot_errors else None,
            xerr=True,
            ax=ax,
            **kwargs,
        )
        ax.set_xlabel("")

    def _plot_ratio(
        self, ax_ratio, hist_num, hist_den, name, is_reference, style, ratio_hist=None
    ):
        """Plot the ratio of hist_1d to ref_hist on the ratio axes."""

        if not ratio_hist:
            bins = hist_num.axes[0].edges
            ratio, err_up, err_down = hep.get_comparison(
                hist_num,
                hist_den,
                comparison="split_ratio" if is_reference else "ratio",
                h1_w2method="sqrt",
            )
        else:
            bins = None
            ratio = ratio_hist
            err_up, err_down = None, None

        color = style.get("color", style.get("edgecolor", style.get("facecolor")))
        histtype_ratio = style.get("histtype_ratio", "errorbar")

        if is_reference:
            ax_ratio.axhline(y=1, linestyle="--", color=color, zorder=0)

            # centers = (bins[1:] + bins[:-1]) / 2
            # ax_ratio.fill_between(
            #     centers, 1 - err_up, 1 + err_up, alpha=0.2, color=color, label=name,zorder=0
            # )
            hep.histplot(
                ratio,
                bins=bins,
                yerr=err_up,
                xerr=True,
                histtype="band",
                label=name,
                ax=ax_ratio,
                facecolor=color,
                zorder=0,
                alpha=0.2,
            )
        else:
            hep.histplot(
                ratio,
                bins=bins,
                yerr=err_up if style.get("plot_errors", True) else None,
                xerr=True,
                histtype=histtype_ratio,
                label=name,
                ax=ax_ratio,
                color=color,
                edges=style.get("edges_ratio", True),
                linewidth=style.get("linewidth", 2),
            )

    def _set_legend(self, ax, pos):
        """Set the legend on the axes."""
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return

        if len(handles) > 5 and self.split_legend:
            ax.legend(loc=pos, ncol=2, fontsize="small")
        else:
            ax.legend(loc=pos)

    def _finalize(self, fig, ax, ax_ratio=None):
        """Final adjustments and saving the figure."""

        if self.y_log:
            ax.set_yscale("log")
        if self.x_log:
            ax.set_xscale("log")

        if self.grid:
            ax.grid()

        if ax_ratio:
            ax_ratio.set_xlabel(self.xlabel)
            ax_ratio.set_ylabel(self.ratio_label)
            if self.grid:
                ax_ratio.grid()
            if self.y_log_ratio:
                ax_ratio.set_yscale("log")
            if not self.y_log_ratio and self.set_ylim:
                ax_ratio.set_ylim(0.5, 1.5)
            if self.legend_ratio:
                self._set_legend(ax_ratio, self.legend_ratio_loc)
        else:
            ax.set_xlabel(self.xlabel)

        ax.set_ylabel(self.ylabel)

        # define the scalar format for y-axis
        # ax.yaxis.set_major_formatter(mtick.ScalarFormatter())
        # ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        if self.set_ylim and self.plot_type != "2d":
            ax.set_ylim(
                top=(
                    self.ylim_top_factor * ax.get_ylim()[1]
                    if not self.y_log
                    else ax.get_ylim()[1] ** (self.ylim_top_factor)
                ),
                bottom=(self.ylim_bottom_factor * ax.get_ylim()[0]),
            )

        if self.plot_type == "2d":
            # label colorbar
            cbar = ax.collections[0].colorbar
            cbar.set_label(self.cbar_label)

        self._apply_annotations(ax)
        self._apply_lines(ax)

        self._apply_cms_labels(ax)

        if self.legend:
            self._set_legend(ax, self.legend_loc)

        if self.create_dir:
            os.makedirs(os.path.dirname(self.output_base), exist_ok=True)
            
        self._save(fig)
        if self.show_plot:
            plt.show()
        else:
            self._close_fig(fig)

    # ----------------------------
    # PLOTTING DISPATCH
    # ----------------------------

    def run(self):
        """Execute the plotting based on the configured plot_type."""
        if self.debug:
            print(
                f"Running HEPPlotter with plot_type={self.plot_type}, output_base={self.output_base}"
            )
        if self.plot_type == "1d":
            self._plot_1d()
        elif self.plot_type == "2d":
            self._plot_2d()
        elif self.plot_type == "graph":
            self._plot_graph()
        else:
            raise ValueError(f"Unknown plot_type={self.plot_type}")
