"""Shared, layout-free core of the DESC Dark Energy Exploration Dashboard.

Loading (.sacc files, .txt MCMC chains, DESC Data Registry lookups), dataset
bookkeeping (labels, style slots, colors) and every figure builder live here, so
that each front end -- the multi-page app (app.py + pages/) and the older
single-module plot_many.py -- draws the same figures from the same code.

Nothing in this module builds Dash layout or registers callbacks.
"""
import argparse
import base64
import functools
import os
import re
import subprocess
import sys
import tempfile

import contourpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import dash
import sacc

### getdist_chainplot_utils imports matplotlib.pyplot at import time, so the
### backend has to be chosen before it. Dash serves callbacks on worker threads,
### and the interactive macOS backend refuses to build a figure off the main
### thread; Agg is non-interactive, which is all this app needs, since every
### chain plot is written to a PNG and embedded rather than shown in a window.
### On NERSC this went unnoticed -- with no display, matplotlib already picked Agg.
import matplotlib
matplotlib.use("Agg")

### Jessie Muir's chain-plotting utility is vendored into Muir_plottingFiles/
### (see its readme.md). Resolve it relative to this file rather than the
### working directory, so the app can be launched from anywhere. It reads
### Y6METHODSDIR at import time, so that has to be set before the import.
CHAINPLOT_UTILS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Muir_plottingFiles")
os.environ["Y6METHODSDIR"] = CHAINPLOT_UTILS_DIR
sys.path.insert(0, CHAINPLOT_UTILS_DIR)
from getdist_chainplot_utils import DEFAULT_PLABELS, get_gdchain, get_paramdict, make_chainplot

### getdist_chainplot_utils sets rc('text', usetex=True) at import time, which
### requires a real LaTeX install to render any text at all -- not available in
### this environment, so revert to matplotlib's built-in mathtext renderer
matplotlib.rc("text", usetex = False)

FONT_FAMILY = "'Source Sans 3', 'Source Sans Pro', sans-serif"

### Chain-file columns that are sampler bookkeeping rather than plottable
### parameters (see get_paramdict in getdist_chainplot_utils.py)
CHAIN_HOUSEKEEPING_COLS = {
    "weight", "post", "like", "prior",
    "old_weight", "old_post", "log_weight", "old_log_weight",
}

### One color per overplotted dataset, in fixed order -- these are the first three
### slots of a categorical palette validated for color-vision deficiency with
### *every* pair compared, not just neighbouring pairs in a legend. A fourth hue
### would break that, so instead of inventing new hues past three datasets, the
### same three repeat with a new second channel: identity is carried by the
### (color, second channel) pair, which also survives greyscale printing.
###
### The second channel is a linestyle for chain contours and for n(z) curves, and
### a marker symbol for data-vector points -- the data vector has no lines to
### style, and its filled/open distinction is already spoken for by the sign of
### the value (see make_data_vector_figure), so identity needs a third shape.
### Both are indexed by the same style slot, so a dataset that is blue and dashed
### in one view is blue and square in the other.
DATASET_COLORS     = ["#2a78d6", "#eb6834", "#1baf7a"]
DATASET_LINESTYLES = ["-", "--", ":"]
DATASET_SYMBOLS    = ["circle", "square", "diamond"]
LINESTYLE_NAMES    = {"-": "solid", "--": "dashed", ":": "dotted"}
SYMBOL_NAMES       = {"circle": "circles", "square": "squares", "diamond": "diamonds"}

### Plotly spells matplotlib's linestyles differently; DATASET_LINESTYLES is kept
### in matplotlib's form because make_chainplot passes it straight to matplotlib
PLOTLY_DASHES = {"-": "solid", "--": "dash", ":": "dot"}

### Cap on overplotted datasets of one kind: past this there are no unique
### (color, second channel) pairs left, and a triangle plot of ten posteriors --
### or a data-vector panel with ten points per bin -- is unreadable anyway
MAX_OVERLAY_CHAINS = len(DATASET_COLORS) * len(DATASET_LINESTYLES)
MAX_OVERLAY_SACCS  = len(DATASET_COLORS) * len(DATASET_SYMBOLS)
MAX_OVERLAY        = {"chain": MAX_OVERLAY_CHAINS, "sacc": MAX_OVERLAY_SACCS}

### Fractional x-offset applied per dataset when several data vectors are
### overplotted, so points measured in the same angular bin don't hide each
### other. Multiplicative because the x axes are logarithmic. The true x is
### carried in customdata and is what the hover box reports.
DATA_VECTOR_X_OFFSET = 0.02

### Parameters to select by default when comparing chains, if every loaded chain
### has them -- alphabetically-first columns (bias_lens--b1, b2) make a poor
### default for a comparison plot
DEFAULT_CHAIN_PARAMS = [
    "cosmological_parameters--omega_m",
    "cosmological_parameters--sigma_8",
]

### Layer a font-only template on top of the default "plotly" template, so every
### figure created below picks up Source Sans without needing font= on each call
pio.templates["source_sans"] = go.layout.Template(layout = go.Layout(font = dict(family = FONT_FAMILY)))
pio.templates.default = "plotly+source_sans"

### The multi-page app has a light / dark toggle, and draws every figure in the
### matching theme. Backgrounds are transparent in both, so a figure takes the
### color of the card it sits in rather than painting its own rectangle over it;
### the dark template's text and gridlines are light enough to read on Bootstrap's
### dark body color (#212529). THEME_INK is the text color of each theme.
THEMES    = ("light", "dark")
THEME_INK = {"light": "#212529", "dark": "#dee2e6"}

pio.templates["deed_light"] = go.layout.Template(
    layout = go.Layout(
        font          = dict(family = FONT_FAMILY, color = THEME_INK["light"]),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
    )
)
pio.templates["deed_dark"] = go.layout.Template(
    layout = go.Layout(
        font          = dict(family = FONT_FAMILY, color = THEME_INK["dark"]),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
    )
)
FIGURE_TEMPLATES = {"light": "plotly_white+deed_light", "dark": "plotly_dark+deed_dark"}

### matplotlib settings for a chain plot on a dark page: transparent background,
### light text, ticks and frame. make_chainplot ends in a bare plt.savefig, which
### honours rcParams, so applying these with rc_context around the call is enough
### -- Jessie Muir's file doesn't need to know about themes
CHAIN_DARK_RC = {
    "savefig.transparent": True,
    "figure.facecolor":    "none",
    "axes.facecolor":      "none",
    "text.color":          THEME_INK["dark"],
    "axes.labelcolor":     THEME_INK["dark"],
    "axes.edgecolor":      "#adb5bd",
    "xtick.color":         THEME_INK["dark"],
    "ytick.color":         THEME_INK["dark"],
    "legend.facecolor":    "none",
    "legend.edgecolor":    "#adb5bd",
}


def natural_key(name):
    """ natural_key
    
        Inputs:
        - name (str)
        
        Process:
        - if name is of the format word+integer, returns an ordered pair of the form (word, integer)
        - E.g., for "tracer1", natural_key returns ("tracer", 1) 
        
        Outputs:
        - ordered pair: (word, integer) 
    """

    match = re.search(r"(\d+)$", name)
    return (name[: match.start()], int(match.group(1))) if match else (name, -1)


def load_sacc(path):
    """ load_sacc

        Inputs:
        - path (str): path to .sacc file
        
        Outputs:
        - sacc file loaded into memory (sacc.sacc.Sacc)
    """
    
    return sacc.Sacc.load_fits(path)


def build_data_vector_df(s):
    """ build_data_vector_df
    
        Inputs:
        - s (sacc file loaded into memory, sacc.sacc.Sacc)
        
        Process:
        - For each dp (i.e., data point) in the sacc file s, extract the data points and organize their content
        into a format appropriate for a Pandas DataFrame
        
        Outputs:
        - Pandas DataFrame with the contents of the sacc file
        
    """
    rows = []
    diag_err = np.sqrt(np.diag(s.covariance.covmat)) if s.covariance is not None else None
    for i, dp in enumerate(s.data):
        
        ### Check that there are only two tracers in the DataPoint
        if len(dp.tracers) != 2:
            raise ValueError(
                f"data point {i} ({dp.data_type}) has {len(dp.tracers)} tracers "
                f"{dp.tracers!r}, expected exactly 2"
            )
        
        ### Extract tracers
        t1, t2 = dp.tracers[0], dp.tracers[-1]
        
        ### Get angular scale if the data are in Fourier space, or angular separation if the data are in real space
        x = dp.tags.get("ell", dp.tags.get("theta"))
        
        ### Organize data in rows
        rows.append(
            {
                "index":     i,
                "data_type": dp.data_type,
                "tracer1":   t1,
                "tracer2":   t2,
                "x":         x,
                "value":     dp.value,
                "err":       diag_err[i] if diag_err is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_nz_df(s):
    """build_nz_df
    
        Inputs:
        -  s (sacc file loaded into memory, sacc.sacc.Sacc)
        
        Process:
        - Checks each tracer in the sacc file.
        - For each tracer, if the tracer has a z attribute, i.e., has a redshifts array, extracts z and n(z)
        
        Outputs:
        - Pandas DataFrame with n(z) for each 
    
    """
    rows = []
    for name, t in s.tracers.items():
        
        ### Check that the tracer as a z array
        if not hasattr(t, "z"):
            continue
        
        ### Check that z and n(z) have the same lengths
        if len(t.z) != len(t.nz):
            raise ValueError(
                f"tracer {name!r} has mismatched z/nz lengths: "
                f"{len(t.z)} vs {len(t.nz)}"
            )
            
        ### Collect z and n(z)
        for z, nz in zip(t.z, t.nz):
            rows.append({"tracer": name, "z": z, "nz": nz})

    return pd.DataFrame(rows)


def data_type_label(dt):
    """ data_type_label

        A dictionary to rename datatypes to have human-readable names.

        Note: plain text with Unicode symbols (ℓ, ξ, θ, γ), not LaTeX -- Plotly's
        MathJax rendering wasn't reliably working in this environment, so figure
        titles and axis labels use this same plain-text form everywhere.
    """
    return {
        "galaxy_density_cl":        "Galaxy density (clustering) Cℓ",
        "galaxy_shearDensity_cl_e": "Galaxy-galaxy lensing Cℓ",
        "galaxy_shear_cl_ee":       "Shear-shear EE Cℓ",
        "galaxy_density_xi":        "Galaxy density (clustering) w(θ)",
        "galaxy_shearDensity_xi_t": "Galaxy-galaxy lensing γt(θ)",
        "galaxy_shear_xi_plus":     "Shear-shear ξ+(θ)",
        "galaxy_shear_xi_minus":    "Shear-shear ξ-(θ)",
    }.get(dt, dt)


def make_data_vector_figure(saccs, frames, data_type):
    """ make_data_vector_figure

        Inputs:
        - saccs (list of dict): sacc dataset entries, in load order, supplying
        the label/color/symbol each file is drawn with
        - frames (list of Pandas DataFrame): the build_data_vector_df output for
        each entry in saccs, in the same order
        - data_type (str)

        Process:
        - Draws every loaded file's data vector for this data type into one
        triangular grid of tracer1 x tracer2 panels. Color identifies the file
        and filled/open identifies the sign of the value -- the single-file
        version of this figure used color for the sign, which multi-file has to
        give up, so the marker symbol carries what's left of file identity
        (see DATASET_SYMBOLS).
        - The grid is the union of the tracer pairs across all files, so a file
        with a different binning shows up as gaps rather than being cropped or
        silently aligned to someone else's tracers.
        - Files that have no data of this type at all are skipped; the
        data-type dropdown annotates the option with who is missing from it
        (see option_label), so their absence is stated rather than inferred.

        Outputs:
        - fig (plotly go.Figure)
    """

    ### Pick out the rows corresponding to data_type, per file, and take the
    ### union of the tracers so the grid covers every file's binning
    subs = [df[df["data_type"] == data_type] for df in frames]
    t1s  = sorted({t for sub in subs for t in sub["tracer1"].unique()}, key = natural_key)
    t2s  = sorted({t for sub in subs for t in sub["tracer2"].unique()}, key = natural_key)

    ### Check if data are in real or Fourier space
    ### x_name/y_name are plain ASCII for hovertemplates; x_label/y_label are the
    ### nicer Unicode versions used for the actual axis titles
    real_space = "_xi"   in data_type
    x_name     = "theta" if real_space else "ell"
    y_name     = "xi"    if real_space else "Cl"
    x_label    = "θ"  if real_space else "ℓ"
    y_label    = "ξ"  if real_space else "Cℓ"

    ### make_subplots needs at least one row and column, so bail out before it
    ### rather than after: no loaded file carries this data type
    if not t1s or not t2s:
        return message_figure(f"No loaded file has any {data_type_label(data_type)} data points.")

    ### Generate figure
    fig = make_subplots(
        rows               = len(t1s),
        cols               = len(t2s),
        shared_xaxes       = True,
        shared_yaxes       = True,
        row_titles         = t1s,
        column_titles      = t2s,
        horizontal_spacing = 0.01,
        vertical_spacing   = 0.02,
    )

    ### Plot the tracers against each other
    ### Since each pair is stored only once (e.g. src0-src1 but not src1-src0), the grid is
    ### triangular: track the last populated row per column, and first populated column per
    ### row, so axis titles land on a panel that actually has data rather than an empty one
    bottom_row_for_col = {}
    left_col_for_row   = {}
    for i, t1 in enumerate(t1s):
        for j, t2 in enumerate(t2s):

            panels = [sub[(sub["tracer1"] == t1) & (sub["tracer2"] == t2)] for sub in subs]

            if all(panel.empty for panel in panels):
                continue

            bottom_row_for_col[j] = max(bottom_row_for_col.get(j, -1), i)
            left_col_for_row[i]   = min(left_col_for_row.get(i, len(t2s)), j)

            for k, (entry, panel) in enumerate(zip(saccs, panels)):

                if panel.empty:
                    continue

                ### Nudge this file's points off the shared bin center, so that
                ### overlaid markers at the same ell/theta stay legible. The
                ### undisplaced x rides along in customdata for the hover box.
                x = panel["x"] * (1 + DATA_VECTOR_X_OFFSET * (k - (len(saccs) - 1) / 2)) if len(saccs) > 1 else panel["x"]

                ### Pick positive and negative values
                pos = panel["value"] >= 0

                ### Plot out positive values filled, negative values open
                for sign, mask, symbol in (("+", pos, entry["symbol"]), ("-", ~pos, f"{entry['symbol']}-open")):

                    if not mask.any():
                        continue

                    fig.add_trace(
                        go.Scatter(
                            x             = x[mask],
                            y             = panel["value"][mask].abs(),
                            customdata    = panel["x"][mask],
                            error_y       = dict(type = "data", array = panel["err"][mask], visible = True),
                            mode          = "markers",
                            marker        = dict(color = entry["color"], size = 6, symbol = symbol),
                            name          = f"{entry['label']} {t1}-{t2} ({sign})",
                            legendgroup   = entry["label"],
                            showlegend    = False,
                            hovertemplate = (
                                f"{entry['label']}<br>{x_name}=%{{customdata:.1f}}<br>"
                                f"{'-' if sign == '-' else ''}{y_name}=%{{y:.3e}}<extra></extra>"
                            ),
                        ),
                        row = i + 1,
                        col = j + 1,
                    )

    ### One legend entry per file, drawn from empty traces: the real traces are
    ### split by panel and by sign, so any of them would key the legend to a
    ### tracer pair rather than to the file
    if len(saccs) > 1:
        for entry in saccs:
            fig.add_trace(
                go.Scatter(
                    x           = [None],
                    y           = [None],
                    mode        = "markers",
                    marker      = dict(color = entry["color"], size = 8, symbol = entry["symbol"]),
                    name        = entry["label"],
                    legendgroup = entry["label"],
                    showlegend  = True,
                ),
                row = 1,
                col = 1,
            )

    ### Plotting parameters
    fig.update_xaxes(type = "log")
    fig.update_yaxes(type = "log")

    ### Put the x-axis title/ticks on the bottommost populated row of each column, and the
    ### y-axis title/ticks on the leftmost populated column of each row (see note above)
    for j, i in bottom_row_for_col.items():
        fig.update_xaxes(title_text = x_label, showticklabels = True, automargin = True, row = i + 1, col = j + 1)
    for i, j in left_col_for_row.items():
        fig.update_yaxes(title_text = y_label, showticklabels = True, automargin = True, row = i + 1, col = j + 1)

    ### Say out loud that the markers are not sitting exactly on their bin
    ### centers, since the axis is the one place a reader would check
    offset_note = f", points offset by ±{100 * DATA_VECTOR_X_OFFSET:.0f}% in {x_label} per file" if len(saccs) > 1 else ""

    fig.update_layout(
        height = max(220 * len(t1s), 350),
        width  = max(220 * len(t2s), 450) + (180 if len(saccs) > 1 else 0),
        title  = f"{data_type_label(data_type)} (filled = positive, open = |negative|{offset_note})",
        margin = dict(t = 80, l = 60, r = 20, b = 40),
    )
    return fig


def make_nz_figure(saccs, nz_frames, selected_tracers, theme = "light"):
    """ make_nz_figure

        Inputs:
        - saccs (list of dict): sacc dataset entries, in load order, supplying
        the label and linestyle each file is drawn with
        - nz_frames (list of Pandas DataFrame): the build_nz_df output for each
        entry in saccs, in the same order, containing z and n(z) per tracer
        - selected_tracers (list of str), the list of tracers whose n(z)s we are plotting
        - theme (str): "light" or "dark"; on a dark page the tracer colors are
        drawn from the light end of each color scale instead of the dark end

        Process:
        - Overlays every loaded file's n(z). Color stays keyed to the tracer, so
        the same redshift bin keeps its color across files and the question this
        view answers -- did this bin move between these two files? -- reads
        directly. That leaves the linestyle to identify the file; the lens/source
        distinction it used to carry is already carried by the color scale the
        tracer is drawn from (Reds vs Blues).
        - There are only three linestyles, so past three files the file's marker
        symbol (the one it is drawn with in the data-vector view) is added to
        the line to keep every file distinguishable.
        - A tracer that only some files have is drawn for the files that have it.

        Outputs:
        - fig (plotly go.Figure)

    """

    ### Build a consistent colormap over every tracer in every file: lens tracers
    ### drawn from a reddish scale, others from a bluish scale
    all_tracers  = sorted({t for nz_df in nz_frames for t in nz_df["tracer"].unique()}, key = natural_key)
    lens_tracers = [name for name in all_tracers if "lens" in name.lower()]
    src_tracers  = [name for name in all_tracers if name not in lens_tracers]

    ### The darkest shades vanish into a dark background, so a dark page samples
    ### the scales from 0.75 down to 0.25 instead of 0.9 down to 0.3
    top, span   = (0.75, 0.5) if theme == "dark" else (0.9, 0.6)
    lens_colors = sample_colorscale("Reds",  [top - span * i / max(len(lens_tracers) - 1, 1) for i in range(len(lens_tracers))])
    src_colors  = sample_colorscale("Blues", [top - span * i / max(len(src_tracers) - 1, 1) for i in range(len(src_tracers))])
    color_map   = dict(zip(lens_tracers, lens_colors)) | dict(zip(src_tracers, src_colors))

    ### Build the figure
    crowded = len(saccs) > len(DATASET_LINESTYLES)

    fig = go.Figure()
    for entry, nz_df in zip(saccs, nz_frames):
        for name in selected_tracers:

            sub = nz_df[nz_df["tracer"] == name]

            if sub.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x                  = sub["z"],
                    y                  = sub["nz"],
                    mode               = "lines+markers" if crowded else "lines",
                    marker             = dict(symbol = entry["symbol"], size = 5, maxdisplayed = 12) if crowded else None,
                    name               = name,
                    legendgroup        = entry["label"],
                    legendgrouptitle   = dict(text = entry["label"]) if len(saccs) > 1 else None,
                    line               = dict(color = color_map[name], dash = PLOTLY_DASHES[entry["linestyle"]]),
                    hovertemplate      = f"{entry['label']}<br>{name}<br>z=%{{x:.3f}}<br>n(z)=%{{y:.3f}}<extra></extra>",
                )
            )

    ### Update plotting parameters
    fig.update_layout(
        title       = "Tracer redshift distributions n(z)",
        xaxis_title = "z",
        yaxis_title = "n(z)",
        height      = 500,
    )
    return fig


def message_figure(text, height = 700, width = 750):
    """ message_figure

        Inputs:
        - text (str): the message to show in place of a figure
        - height, width (int): figure size, so the message occupies the space
        the figure would have

        Process:
        - Builds an empty figure carrying a centered message. Used wherever a
        view can't be drawn for a reason the reader should see rather than
        being left with a blank panel.

        Outputs:
        - fig (plotly go.Figure)
    """
    fig = go.Figure()
    fig.add_annotation(
        text      = text,
        showarrow = False,
        xref      = "paper",
        yref      = "paper",
        x         = 0.5,
        y         = 0.5,
        font      = dict(size = 14),
    )
    fig.update_layout(height = height, width = width)
    return fig


def apply_theme(fig, theme):
    """ apply_theme

        Inputs:
        - fig (plotly go.Figure)
        - theme (str): "light" or "dark"

        Process:
        - Sets the figure's template to the one for this theme (see
        FIGURE_TEMPLATES). Explicit layout settings on the figure still win, so
        this only changes what the figure left to its template: backgrounds,
        text and gridline colors.

        Outputs:
        - fig (plotly go.Figure), the same object, for chaining
    """
    fig.update_layout(template = FIGURE_TEMPLATES.get(theme, FIGURE_TEMPLATES["light"]))
    return fig


def make_covariance_figure(s, df, mode, label = None):
    """ make_covariance_figure

        Inputs:
        - s (sacc file loaded into memory, sacc.sacc.Sacc)
        - df (Pandas DataFrame), output of build_data_vector_df, used for axis labels and block boundaries
        - mode (str), one of "correlation", "covariance", "precision"
        - label (str or None): the dataset label, named in the title when
        several sacc files are loaded and this is one of them

        Process:
        - Builds the requested matrix representation of the covariance. One file
        at a time: heatmaps can't be overlaid the way the data vector and n(z)
        views overlay their curves, so the Covariance tab picks a file instead.

        Outputs:
        - fig (plotly go.Figure)

    """

    ### Get covariance matrix
    cov = s.covariance.covmat

    ### Compute correlation matrix, if chosen with `mode`
    if mode == "correlation":
        d          = np.sqrt(np.diag(cov))
        matrix     = cov / np.outer(d, d)
        zmin, zmax = -1, 1
        colorscale = "RdBu"
        title      = "Correlation matrix"

    ### Compute precision matrix, if chosen with `mode`
    elif mode == "precision":

        ### Guard: refuse to invert a covariance that is too ill-conditioned to trust
        cond = np.linalg.cond(cov)
        if cond > 1.0 / np.finfo(cov.dtype).eps:
            return message_figure(
                f"Precision matrix not shown: covariance is too ill-conditioned to invert "
                f"reliably (condition number = {cond:.2e})."
            )

        precision  = s.covariance.inverse
        matrix     = np.sign(precision) * np.log10(np.abs(precision) + 1e-30)
        zmin, zmax = -np.nanmax(np.abs(matrix)), np.nanmax(np.abs(matrix))
        colorscale = "RdBu"
        title      = f"Precision matrix (sign * log10|inverse cov|), condition number = {cond:.2e}"

    ### Keep covariance matrix, in log10, if chosen with `mode`
    else:
        matrix     = np.sign(cov) * np.log10(np.abs(cov) + 1e-30)
        zmin, zmax = -np.nanmax(np.abs(matrix)), np.nanmax(np.abs(matrix))
        colorscale = "RdBu"
        title      = "Covariance matrix (sign * log10|cov|)"

    ### Build figure
    labels = [f"{r.data_type}:{r.tracer1}-{r.tracer2}:{i}" for i, r in df.iterrows()]

    fig = go.Figure(
        data = go.Heatmap(
            z             = matrix,
            x             = labels,
            y             = labels,
            colorscale    = colorscale,
            zmin          = zmin,
            zmax          = zmax,
            hovertemplate = "%{y}<br>%{x}<br>value=%{z:.3f}<extra></extra>",
        )
    )
    
    ### Update plotting parameters
    fig.update_layout(
        title  = f"{title} -- {label}" if label else title,
        height = 700,
        width  = 750,
        xaxis  = dict(showticklabels = False),
        yaxis  = dict(showticklabels = False, autorange = "reversed"),
    )


    boundaries = df["data_type"].ne(df["data_type"].shift()).to_numpy().nonzero()[0]
    for b in boundaries[1:]:
        ### Drawn on the heatmap, which is mostly white in either theme, so the
        ### boundaries stay dark even on a dark page
        fig.add_vline(x = b - 0.5, line = dict(color = THEME_INK["light"], width = 1))
        fig.add_hline(y = b - 0.5, line = dict(color = THEME_INK["light"], width = 1))

    return fig

### Dataset bookkeeping
###
### The dashboard holds a list of datasets rather than a single path. Each entry
### is a plain dict (it lives in a dcc.Store, so it has to be JSON-serializable):
###
###   {"path": str, "origin": str, "kind": "chain"|"sacc", "label": str,
###    "slot": int, "color": str, "linestyle": str, "symbol": str|None}
###
### "symbol" is None for chains, which have no markers to style.
###
### Only "path" and "origin" are supplied when a dataset is added; the rest are
### filled in by normalize_datasets.
def dataset_kind(path):
    """ dataset_kind

        Inputs:
        - path (str): path to a dataset file

        Outputs:
        - "chain" for a .txt MCMC chain, "sacc" otherwise
    """
    return "chain" if path.endswith(".txt") else "sacc"


def dataset_labels(paths):
    """ dataset_labels

        Inputs:
        - paths (list of str): dataset paths

        Process:
        - Labels each dataset by its file name without the extension, for the
        getdist legend and the loaded-datasets panel. Comparison runs often
        pull the same file name out of different directories, so any name
        shared by two datasets falls back to parent directory + full file name
        -- which also separates a chain from a sacc file of the same stem.

        Outputs:
        - labels (list of str), one per path, in the same order
    """
    names  = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    shared = {name for name in names if names.count(name) > 1}
    labels = [
        os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p)) if name in shared else name
        for p, name in zip(paths, names)
    ]

    ### Two chains can still land on the same label (same file name under
    ### same-named parent directories), and a legend with two identical entries
    ### is worse than an ugly one, so number any that are still ambiguous
    for label in {label for label in labels if labels.count(label) > 1}:
        for n, i in enumerate([i for i, l in enumerate(labels) if l == label], start = 1):
            labels[i] = f"{label} ({n})"

    return labels


def assign_style_slots(entries):
    """ assign_style_slots

        Inputs:
        - entries (list of dict): dataset entries, some already carrying a "slot"

        Process:
        - Gives every entry a style slot, which indexes into DATASET_COLORS and
        DATASET_LINESTYLES/DATASET_SYMBOLS. Entries that already have one keep
        it, and new entries take the lowest free slot -- so removing one dataset
        never repaints the others, and a dataset's color means the same thing
        for as long as it stays loaded.
        - Chains and sacc files draw from separate slot pools: no figure mixes
        the two kinds, so the first chain and the first sacc file can both be
        blue without ever being confusable.

        Outputs:
        - None (entries are updated in place)
    """
    for kind, cap in MAX_OVERLAY.items():

        group = [e for e in entries if e["kind"] == kind]
        taken = {e["slot"] for e in group if e.get("slot") is not None}

        for entry in group:
            if entry.get("slot") is None:
                ### The caller caps how many datasets of a kind can be loaded
                ### (add_dataset, and main() for the command line), so a free
                ### slot always exists by the time this runs
                entry["slot"] = next(slot for slot in range(cap) if slot not in taken)
                taken.add(entry["slot"])


def normalize_datasets(items):
    """ normalize_datasets

        Inputs:
        - items (None, or list of str, or list of dict): datasets to load, as
        bare paths or as partial entries carrying at least a "path"

        Process:
        - Fills in every derived field of a dataset entry: absolute path, kind,
        style slot, color, linestyle, label. Called both when the app starts
        and on every add/remove, so the store always holds complete entries.

        Outputs:
        - entries (list of dict)
    """
    entries = []
    for item in items or []:
        item = {"path": item} if isinstance(item, str) else dict(item)
        path = os.path.abspath(os.path.expanduser(item["path"]))
        entries.append(
            {
                "path":   path,
                "origin": item.get("origin", "file"),
                "kind":   dataset_kind(path),
                "slot":   item.get("slot"),
            }
        )

    assign_style_slots(entries)

    for entry, label in zip(entries, dataset_labels([e["path"] for e in entries])):
        entry["label"] = label
        entry["color"] = DATASET_COLORS[entry["slot"] % len(DATASET_COLORS)]

        ### The two kinds index the linestyle differently, because the two views
        ### spend their color differently. A chain overlay colors by chain, so
        ### the linestyle only has to break ties between chains that share a hue
        ### (slot // 3). The n(z) overlay colors by *tracer*, leaving nothing but
        ### the linestyle to identify the file -- so there it has to vary first
        ### (slot % 3), in step with the color the same file carries in the
        ### data-vector view.
        if entry["kind"] == "sacc":
            entry["linestyle"] = DATASET_LINESTYLES[entry["slot"] %  len(DATASET_LINESTYLES)]
            entry["symbol"]    = DATASET_SYMBOLS[entry["slot"]    // len(DATASET_COLORS)]
        else:
            entry["linestyle"] = DATASET_LINESTYLES[entry["slot"] // len(DATASET_COLORS)]
            entry["symbol"]    = None

    return entries


def datasets_of_kind(datasets, kind):
    """ datasets_of_kind

        Inputs:
        - datasets (list of dict): dataset entries
        - kind (str): "chain" or "sacc"

        Outputs:
        - the subset of entries of that kind, in load order
    """
    return [d for d in (datasets or []) if d["kind"] == kind]


@functools.lru_cache(maxsize = 32)
def chain_params(chain_path):
    """ chain_params

        Inputs:
        - chain_path (str): path to a .txt MCMC chain file

        Process:
        - Reads the chain's column header and keeps the plottable parameters.
        get_paramdict returns derived columns twice, in their original upper
        case and lower-cased (e.g. COSMOLOGICAL_PARAMETERS--SIGMA_8 and
        cosmological_parameters--sigma_8), so keeping only lower-case keys
        deduplicates them; sampler bookkeeping columns are dropped outright.
        Cached because the parameter checklist is rebuilt on every add/remove.

        Outputs:
        - params (frozenset of str)
    """
    pdict = get_paramdict(chain_path)
    return frozenset(k for k in pdict if k.islower() and k not in CHAIN_HOUSEKEEPING_COLS)


@functools.lru_cache(maxsize = 4)
def load_chain_samples(chain_path):
    """ load_chain_samples

        Inputs:
        - chain_path (str): path to a .txt MCMC chain file

        Process:
        - Reads the chain's samples (whitespace-separated, '#' header lines) and
        its per-sample weights: the "weight" column if there is one, else
        exp(log_weight) rescaled to a maximum of 1, else uniform. No burn-in is
        removed -- matching what make_chainplot does for nested-sampler chains
        such as these PolyChord ones. Cached because the summary table is
        rebuilt on every checklist change.

        Outputs:
        - samples (np.ndarray), one row per sample
        - weights (np.ndarray)
        - pdict (dict): column name -> index, from get_paramdict
    """
    pdict   = get_paramdict(chain_path)
    samples = np.loadtxt(chain_path, comments = "#", ndmin = 2)
    if "weight" in pdict:
        weights = samples[:, pdict["weight"]]
    elif "log_weight" in pdict:
        log_w   = samples[:, pdict["log_weight"]]
        weights = np.exp(log_w - log_w.max())
    else:
        weights = np.ones(len(samples))
    return samples, weights, pdict


### --------------------------------------------------- interactive chain plots
###
### getdist computes, Plotly draws (deed_notes/interactive_chains_design.md):
### each chain is read with Muir's get_gdchain -- the same smoothing and prior
### ranges make_chainplot uses -- and its densities, contour levels and
### marginal statistics are pulled out of getdist as arrays.

### Probability contained by the inner and outer 2D contours, and by the
### intervals quoted for 1D marginals -- getdist's own defaults
CHAIN_CONTOURS = (0.68, 0.95)

### Figure of merit for a parameter pair: 1 / area of the ellipse containing
### FOM_CONTOUR of a 2D Gaussian with the pair's covariance C, whose area is
### pi * (-2 ln(1 - FOM_CONTOUR)) * sqrt(det C).
### PROVISIONAL (Leo, 2026-10-28): to be re-checked. Alternatives are the DETF
### 1 / sqrt(det C) (same ratios between chains) and 1 / the area inside
### getdist's smoothed 68% contour. Change pair_figure_of_merit, nothing else.
FOM_CONTOUR = 0.68

### Past this many parameters the Plotly triangle (n(n+1)/2 panels) gets slow to
### draw and hard to read; the view warns rather than refusing
CHAIN_PARAMS_WARN = 8

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "eta": "η", "theta": "θ",
    "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ",
    "Theta": "Θ", "Lambda": "Λ", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}


def latex_to_plotly(label):
    r""" latex_to_plotly

        Inputs:
        - label (str): a getdist / Muir LaTeX label, e.g. "$\Omega_{\rm m}$"

        Process:
        - Turns the simple LaTeX used for parameter labels into the HTML subset
        Plotly renders without MathJax (which isn't reliable in this
        environment): Greek letters to Unicode, _{...} / ^{...} to <sub> / <sup>,
        \rm / \mathrm dropped, braces and $ removed.

        Outputs:
        - label (str) for Plotly axis titles and hover text
    """
    text = label.strip("$ ")
    text = re.sub(r"\\(?:rm|mathrm|text|textrm)\s*", "", text)
    text = re.sub(r"\\([A-Za-z]+)", lambda m: GREEK.get(m.group(1), m.group(1)), text)
    for marker, tag in (("_", "sub"), ("^", "sup")):
        text = re.sub(re.escape(marker) + r"\{([^{}]*)\}", rf"<{tag}>\1</{tag}>", text)
        text = re.sub(re.escape(marker) + r"(\w)", rf"<{tag}>\1</{tag}>", text)
    return text.replace("{", "").replace("}", "").replace("\\", "")


def param_label(param):
    """ param_label

        Inputs:
        - param (str): a chain column name, e.g. "cosmological_parameters--omega_m"

        Outputs:
        - display label (str): Muir's label for it if she has one, converted for
        Plotly; else the part after "--"
    """
    latex = DEFAULT_PLABELS.get(param)
    return latex_to_plotly(latex) if latex else param.split("--")[-1]


@functools.lru_cache(maxsize = 2 * MAX_OVERLAY_CHAINS)
def load_gd_samples(chain_path):
    """ load_gd_samples

        Inputs:
        - chain_path (str): path to a .txt MCMC chain file

        Process:
        - Reads every column of the chain into a getdist MCSamples with Muir's
        get_gdchain, called as make_chainplot calls it (default kdesmooth, prior
        ranges from the chain header). Cached by path: about 0.2 s per chain.

        Outputs:
        - getdist.MCSamples
    """
    return get_gdchain(chain_path, flabel = os.path.basename(chain_path))


@functools.lru_cache(maxsize = 256)
def chain_1d_density(chain_path, param):
    """ chain_1d_density

        Inputs:
        - chain_path (str), param (str)

        Outputs:
        - x, P (np.ndarray): getdist's smoothed 1D marginal, peak-normalized to 1
    """
    density = load_gd_samples(chain_path).get1DDensity(param)
    return np.asarray(density.x), np.asarray(density.P) / np.max(density.P)


@functools.lru_cache(maxsize = 512)
def chain_2d_contours(chain_path, param_x, param_y):
    """ chain_2d_contours

        Inputs:
        - chain_path (str), param_x (str), param_y (str)

        Process:
        - getdist's smoothed 2D density on a grid, the levels enclosing
        CHAIN_CONTOURS of the probability, and the outline(s) at each level, from
        contourpy (already a matplotlib dependency). A level can have several
        disjoint outlines (a multimodal posterior); open outlines that run into a
        prior edge are closed along it by the fill.

        Outputs:
        - dict: contour fraction -> list of (x, y) arrays, one per outline
    """
    density = load_gd_samples(chain_path).get2DDensityGridData(param_x, param_y, get_density = True)
    levels  = density.getContourLevels(CHAIN_CONTOURS)
    gen     = contourpy.contour_generator(density.x, density.y, density.P)
    return {
        fraction: [(line[:, 0], line[:, 1]) for line in gen.lines(level) if len(line) > 2]
        for fraction, level in zip(CHAIN_CONTOURS, levels)
    }


@functools.lru_cache(maxsize = 512)
def chain_param_stats(chain_path, param):
    """ chain_param_stats

        Inputs:
        - chain_path (str), param (str)

        Outputs:
        - dict: "mean", "std" and, per fraction in CHAIN_CONTOURS, the
        two-tail interval ("lo68", "hi68", "lo95", "hi95") -- getdist's
        marginal statistics, so they match the plotted curves
    """
    marge = load_gd_samples(chain_path).getMargeStats().parWithName(param)
    stats = {"mean": float(marge.mean), "std": float(marge.err)}
    for fraction, limit in zip(CHAIN_CONTOURS, marge.limits):
        pct = round(100 * fraction)
        stats[f"lo{pct}"], stats[f"hi{pct}"] = float(limit.lower), float(limit.upper)
    return stats


def pair_figure_of_merit(cov):
    """ pair_figure_of_merit

        Inputs:
        - cov (2x2 np.ndarray): covariance of a parameter pair

        Outputs:
        - the pair's figure of merit (float): 1 / area of its FOM_CONTOUR ellipse.
        PROVISIONAL -- see FOM_CONTOUR
    """
    area = np.pi * -2 * np.log(1 - FOM_CONTOUR) * np.sqrt(np.linalg.det(cov))
    return 1 / area


@functools.lru_cache(maxsize = 512)
def chain_pair_stats(chain_path, param_x, param_y):
    """ chain_pair_stats

        Inputs:
        - chain_path (str), param_x (str), param_y (str)

        Outputs:
        - dict: "rho" (correlation coefficient) and "fom" (pair_figure_of_merit)
        from the weighted-sample covariance
    """
    cov = np.asarray(load_gd_samples(chain_path).cov([param_x, param_y]))
    return {"rho": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])), "fom": float(pair_figure_of_merit(cov))}


def param_pairs(params):
    """ param_pairs

        Inputs:
        - params (list of str), in display order

        Outputs:
        - list of (param_x, param_y): every pair, in triangle order (x before y)
    """
    return [(params[j], params[i]) for i in range(len(params)) for j in range(i)]


def fom_ratio_text(fom, ref_fom):
    """Ratio of an FoM to the reference chain's, as hover / table text."""
    return f"×{fom / ref_fom:.2f} vs reference" if ref_fom else ""


def hex_to_rgba(color, alpha):
    """'#rrggbb' -> 'rgba(r,g,b,alpha)', for translucent contour fills."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def make_interactive_chain_figure(chains, params, reference = None):
    """ make_interactive_chain_figure

        Inputs:
        - chains (list of dict): chain dataset entries, in load order (label,
        color, linestyle as everywhere else)
        - params (list of str): parameters to plot; must be in every chain (see
        shared_chain_params)
        - reference (str or None): path of the chain FoM ratios are quoted
        against; None for the first chain

        Process:
        - A Plotly triangle plot of getdist's smoothed marginals: 1D curves on the
        diagonal, filled 68% / 95% regions below it, each chain in its own color
        and linestyle. Every curve and region carries hover text with its
        statistics (mean ± std and intervals; for a pair also the correlation and
        the figure of merit, with the ratio to the reference chain).
        - Under each panel sits an invisible, clickable heatmap, so a click
        anywhere in a panel -- not only on a curve -- reports which panel it was.
        Click events on filled regions alone aren't reliable.
        - Column x axes are shared, as are the y axes of each row's 2D panels, so
        zooming one panel zooms its row and column.

        Outputs:
        - fig (plotly go.Figure)
        - trace_panels (list): per trace, the panel it belongs to -- [param] on
        the diagonal, [param_x, param_y] below it -- so a clickData's
        curveNumber can be mapped back to a panel
    """
    n          = len(params)
    ref_path   = reference if any(d["path"] == reference for d in chains) else chains[0]["path"]
    labels     = {p: param_label(p) for p in params}
    trace_tags = []

    fig = make_subplots(rows = n, cols = n, shared_xaxes = True, horizontal_spacing = 0.02, vertical_spacing = 0.02)

    ### Axis ranges: the union of every chain's 1D support, where the smoothed
    ### marginal is above 0.1% of its peak
    ranges = {}
    for p in params:
        lows, highs = [], []
        for d in chains:
            x, P = chain_1d_density(d["path"], p)
            keep = x[P > 1e-3]
            lows.append(keep.min()); highs.append(keep.max())
        ranges[p] = [min(lows), max(highs)]

    for i, p_y in enumerate(params):
        for j, p_x in enumerate(params[:i + 1]):
            row, col = i + 1, j + 1
            on_diagonal = i == j

            ### Clickable backdrop: transparent, no hover label of its own
            x0, x1 = ranges[p_x]
            y0, y1 = (0, 1.08) if on_diagonal else ranges[p_y]
            fig.add_trace(
                go.Heatmap(
                    x = np.linspace(x0, x1, 12), y = np.linspace(y0, y1, 12), z = np.zeros((12, 12)),
                    opacity = 0, showscale = False, hoverinfo = "none",
                ),
                row = row, col = col,
            )
            trace_tags.append([p_x] if on_diagonal else [p_x, p_y])

            for slot, d in enumerate(chains):
                if on_diagonal:
                    x, P  = chain_1d_density(d["path"], p_x)
                    stats = chain_param_stats(d["path"], p_x)
                    text  = (
                        f"<b>{d['label']}</b><br>{labels[p_x]} = {stats['mean']:.4g} ± {stats['std']:.3g}<br>"
                        f"68%: [{stats['lo68']:.4g}, {stats['hi68']:.4g}]<br>95%: [{stats['lo95']:.4g}, {stats['hi95']:.4g}]"
                    )
                    fig.add_trace(
                        go.Scatter(
                            x = x, y = P, mode = "lines", name = d["label"], legendgroup = d["label"],
                            showlegend = i == 0 and len(chains) > 1,
                            line = dict(color = d["color"], dash = PLOTLY_DASHES[d["linestyle"]], width = 2),
                            text = text, hoverinfo = "text", hoveron = "points",
                        ),
                        row = row, col = col,
                    )
                    trace_tags.append([p_x])
                    continue

                pair    = chain_pair_stats(d["path"], p_x, p_y)
                ref_fom = chain_pair_stats(ref_path, p_x, p_y)["fom"] if d["path"] != ref_path else None
                sx, sy  = chain_param_stats(d["path"], p_x), chain_param_stats(d["path"], p_y)
                outlines = chain_2d_contours(d["path"], p_x, p_y)

                ### Outer region first, so the inner one is drawn (and hovered) on top
                for fraction, alpha in ((CHAIN_CONTOURS[1], 0.25), (CHAIN_CONTOURS[0], 0.55)):
                    text = (
                        f"<b>{d['label']}</b> -- {round(100 * fraction)}% region<br>"
                        f"{labels[p_x]} = {sx['mean']:.4g} ± {sx['std']:.3g}<br>"
                        f"{labels[p_y]} = {sy['mean']:.4g} ± {sy['std']:.3g}<br>"
                        f"ρ = {pair['rho']:+.3f}<br>FoM = {pair['fom']:.4g} {fom_ratio_text(pair['fom'], ref_fom)}"
                    )
                    for k, (cx, cy) in enumerate(outlines[fraction]):
                        fig.add_trace(
                            go.Scatter(
                                x = cx, y = cy, mode = "lines", fill = "toself",
                                fillcolor = hex_to_rgba(d["color"], alpha),
                                line = dict(color = d["color"], dash = PLOTLY_DASHES[d["linestyle"]], width = 1.2),
                                name = d["label"], legendgroup = d["label"], showlegend = False,
                                text = text, hoverinfo = "text", hoveron = "fills+points",
                            ),
                            row = row, col = col,
                        )
                        trace_tags.append([p_x, p_y])

            ### Axes: ranges, titles on the outer edge only, diagonal y hidden
            fig.update_xaxes(range = ranges[p_x], row = row, col = col,
                             title_text = labels[p_x] if i == n - 1 else None, showticklabels = i == n - 1)
            if on_diagonal:
                fig.update_yaxes(range = [0, 1.08], showticklabels = False, showgrid = False, row = row, col = col)
            else:
                fig.update_yaxes(range = ranges[p_y], row = row, col = col,
                                 title_text = labels[p_y] if j == 0 else None, showticklabels = j == 0)
                ### Every 2D panel in a row shows the same y parameter: link them
                if j > 0:
                    first = fig.get_subplot(row, 1).yaxis.plotly_name.replace("axis", "")
                    fig.update_yaxes(matches = first, row = row, col = col)

    size = max(220 * n, 480) + 90
    fig.update_layout(
        width = size + (200 if len(chains) > 1 else 0), height = size,
        margin = dict(t = 30, l = 70, r = 20, b = 60),
        hovermode = "closest", dragmode = "zoom",
        legend = dict(x = 1.02, y = 1, xanchor = "left"),
    )
    return fig, trace_tags


def chain_summary(chain_path, params):
    """ chain_summary

        Inputs:
        - chain_path (str): path to a .txt MCMC chain file
        - params (list of str): parameter column names

        Process:
        - getdist's marginal statistics for each parameter (chain_param_stats),
        so the table agrees with the interactive plot's hover text. Parameters
        the chain doesn't have are skipped.

        Outputs:
        - list of dict: {"param", "mean", "std", "lo68", "hi68", "lo95", "hi95"}
        - n_eff (float): Kish effective sample size, (sum w)^2 / sum w^2
    """
    _, weights, pdict = load_chain_samples(chain_path)
    rows  = [{"param": p, **chain_param_stats(chain_path, p)} for p in params if p in pdict]
    n_eff = weights.sum() ** 2 / (weights ** 2).sum()
    return rows, n_eff


def shared_chain_params(chains):
    """ shared_chain_params

        Inputs:
        - chains (list of dict): chain dataset entries

        Process:
        - Splits the chains' parameters into those present in every chain and
        those that are not. Only shared parameters can be overplotted: asked
        for a column a chain doesn't have, get_gdchain prints a note and
        silently drops it for that chain (see the `except` around the pdict
        lookup in getdist_chainplot_utils.get_gdchain), which would quietly
        compare different parameter sets. E.g. chain_3x2pt_fixednu_lcdm.txt
        holds the neutrino mass fixed and so has no
        cosmological_parameters--omnuh2 column at all.

        Outputs:
        - shared (list of str), sorted
        - missing (dict): parameter -> list of labels of the chains lacking it,
        for the parameters that are not shared
    """
    param_sets = [chain_params(d["path"]) for d in chains]

    shared  = sorted(frozenset.intersection(*param_sets)) if param_sets else []
    missing = {}
    for param in sorted(frozenset().union(*param_sets) - set(shared)) if param_sets else []:
        missing[param] = [d["label"] for d, params in zip(chains, param_sets) if param not in params]

    return shared, missing


@functools.lru_cache(maxsize = 8)
def make_chain_figure(paths, params, labels, colors, linestyles, theme = "light"):
    """ make_chain_figure

        Inputs (all tuples, so this is hashable and can be cached -- a triangle
        plot of several chains takes seconds, and toggling a parameter off and
        back on should not pay for it twice):
        - paths (tuple of str): paths to .txt MCMC chain files, in load order
        - params (tuple of str): chain column names to plot; must be present in
        every chain (see shared_chain_params)
        - labels (tuple of str): legend label per chain
        - colors (tuple of str): contour color per chain
        - linestyles (tuple of str): matplotlib linestyle per chain
        - theme (str): "light" for the usual white-background PNG, "dark" for a
        transparent one with light text (see CHAIN_DARK_RC); part of the cache key

        Process:
        - Overplots the chains with make_chainplot, which takes a list of chain
        files and keeps labels/colors/linestyles aligned with them (it drops any
        chain it fails to read). Renders into a temp PNG -- make_chainplot draws
        with matplotlib/getdist and saves to disk rather than returning a Plotly
        figure -- then reads it back as a base64 data URI, since Dash can't point
        an <img> at an arbitrary filesystem path. clearfig=True keeps
        make_chainplot from leaking a matplotlib figure into memory on every
        checklist change.
        - indatdir="" because paths are already absolute. rangedict is left at
        its default, so getdist reads each chain's prior ranges from its own
        header and corrects that chain's posterior accordingly.

        Outputs:
        - data URI (str) suitable for an html.Img's src
    """
    ### With one parameter selected, make_chainplot draws a 1D posterior -- and
    ### that branch indexes param_limits by parameter name (line 2038) before the
    ### None check further down, so the default None raises TypeError. Passing an
    ### explicit "no limits for this parameter" entry avoids patching her file.
    ### The triangle branch (2+ parameters) handles None itself, so leave it.
    param_limits = {params[0]: None} if len(params) == 1 else None

    with tempfile.TemporaryDirectory() as tmpdir, matplotlib.rc_context(CHAIN_DARK_RC if theme == "dark" else {}):
        make_chainplot(
            inputflist   = list(paths),
            indatdir     = "",
            params       = list(params),
            outname      = "chain_plot",
            inlabels     = list(labels),
            colors       = list(colors),
            linestyles   = list(linestyles),
            param_limits = param_limits,
            outdir       = tmpdir,
            ftype        = "png",
            clearfig     = True,
        )
        with open(os.path.join(tmpdir, "chain_plot.png"), "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"



### Sized so that every sacc file the app will let you load at once stays
### resident: each figure callback now touches all of them, and reparsing a
### 5 MB FITS file per overlaid dataset per interaction would be felt
@functools.lru_cache(maxsize = MAX_OVERLAY_SACCS)
def load_dataset_bundle(sacc_path):
    """ load_dataset_bundle

        Inputs:
        - sacc_path (str): path to a .sacc file

        Process:
        - Loads the sacc file and derives everything the dashboard needs from it.
        Cached by path, so switching back to an already-seen dataset (including the
        one the app started with) is instant, and the figure callbacks below don't
        reparse the FITS file on every interaction with an unrelated control.

        Outputs:
        - s (sacc.sacc.Sacc)
        - df (Pandas DataFrame), output of build_data_vector_df
        - nz_df (Pandas DataFrame), output of build_nz_df
        - data_types (list of str)
        - tracers (list of str)
        - has_cov (bool)
    """
    s          = load_sacc(sacc_path)
    df         = build_data_vector_df(s)
    nz_df      = build_nz_df(s)
    data_types = list(df["data_type"].unique())
    tracers    = sorted(nz_df["tracer"].unique(), key = natural_key)
    has_cov    = s.covariance is not None
    return s, df, nz_df, data_types, tracers, has_cov



def shared_sacc_contents(saccs):
    """ shared_sacc_contents

        Inputs:
        - saccs (list of dict): sacc dataset entries, in load order

        Process:
        - Takes the union of the data types and of the tracers across the loaded
        files, and records which files are missing each one. Unlike a chain
        parameter -- which getdist would silently drop from one chain, quietly
        comparing different parameter sets -- a data type or tracer that only
        some files have is harmless to offer: the overlay simply draws fewer
        datasets. So these stay selectable, and the option is annotated with
        who is missing instead of being disabled.

        Outputs:
        - data_types (list of str), in first-seen order
        - tracers (list of str), natural-sorted
        - missing (dict): data type or tracer -> labels of the files lacking it
    """
    bundles = [load_dataset_bundle(d["path"]) for d in saccs]

    data_types = list(dict.fromkeys(dt for bundle in bundles for dt in bundle[3]))
    tracers    = sorted({t for bundle in bundles for t in bundle[4]}, key = natural_key)

    missing = {}
    for name, index in [(dt, 3) for dt in data_types] + [(t, 4) for t in tracers]:
        absent = [d["label"] for d, bundle in zip(saccs, bundles) if name not in bundle[index]]
        if absent:
            missing[name] = absent

    return data_types, tracers, missing


def option_label(name, missing, pretty = None):
    """ option_label

        Inputs:
        - name (str): the data type or tracer the option selects
        - missing (dict): output of shared_sacc_contents
        - pretty (str or None): display name, if it differs from `name`

        Outputs:
        - the option's label, annotated with which files lack it
    """
    text = pretty or name
    return f"{text} (not in {', '.join(missing[name])})" if name in missing else text



def add_dataset(datasets, path, origin):
    """ add_dataset

        Inputs:
        - datasets (list of dict): the currently-loaded entries
        - path (str): absolute path of the dataset to add
        - origin (str): "file", or "registry:<dataset_id>"

        Process:
        - Appends the dataset, refusing duplicates and refusing to exceed
        MAX_OVERLAY[kind] datasets of that kind (past that there are no unique
        (color, linestyle/symbol) pairs left to tell them apart with)

        Outputs:
        - datasets (list of dict) or dash.no_update
        - status message (str)
    """
    if any(d["path"] == path for d in datasets):
        return dash.no_update, f"{os.path.basename(path)} is already loaded."

    kind = dataset_kind(path)
    if len(datasets_of_kind(datasets, kind)) >= MAX_OVERLAY[kind]:
        return dash.no_update, (
            f"Already overplotting {MAX_OVERLAY[kind]} {kind} datasets, which is as many as can be "
            f"told apart -- remove one first."
        )

    datasets = normalize_datasets(datasets + [{"path": path, "origin": origin}])
    return datasets, f"Added {os.path.basename(path)} ({kind}, from {origin})."


def check_dataset_extension(path):
    """ check_dataset_extension

        Inputs:
        - path (str): candidate path to a dataset file

        Process:
        - Checks that path has the .sacc or .txt (MCMC chain) extension

        Outputs:
        - path (str), unchanged, if valid
    """
    if not (path.endswith(".sacc") or path.endswith(".txt")):
        raise ValueError(f"expected a .sacc or .txt file, got: {path!r}")
    return path


def dataset_file_path(path):
    """ dataset_file_path

        argparse `type=` wrapper around check_dataset_extension, for --file validation
    """
    try:
        return check_dataset_extension(path)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def resolve_registry_path(dataset_id, config_file=None):
    """ resolve_registry_path

        Inputs:
        - dataset_id (int): the dataset_id to look up in the DESC Data Registry's
        working schema -- this is the table's primary key, so it's unique by
        construction (unlike name or relative_path, no disambiguation is needed)
        - config_file (str or None): path to a dataregistry config file;
        None uses the library's default config resolution (NERSC by default)

        Process:
        - Resolves the dataset's absolute path on disk directly by dataset_id,
        restricted to the working schema (as opposed to production)
        - Checks that the resolved path is a .sacc or .txt (MCMC chain) file

        Outputs:
        - path (str): absolute path of the registered dataset
    """
    from dataregistry import DataRegistry

    registry = DataRegistry(config_file = config_file)
    path     = registry.get_dataset_absolute_path(dataset_id, schema = "working", silent = False)
    return check_dataset_extension(path)


### Columns shown by the Registry page, in display order
REGISTRY_COLUMNS = [
    "dataset.dataset_id", "dataset.name", "dataset.version_string", "dataset.owner",
    "dataset.owner_type", "dataset.relative_path", "dataset.creation_date", "dataset.description",
]
REGISTRY_MAX_ROWS = 200

_REGISTRY = None


def get_registry(config_file = None):
    """ get_registry

        Inputs:
        - config_file (str or None): dataregistry config file, None for the
        library's default resolution (see deed_notes/open_questions.md: on NERSC
        that currently means the shared writer config)

        Process:
        - Opens one DataRegistry connection and reuses it, so every search on the
        Registry page doesn't pay for a new database connection

        Outputs:
        - dataregistry.DataRegistry
    """
    global _REGISTRY
    if _REGISTRY is None:
        from dataregistry import DataRegistry
        _REGISTRY = DataRegistry(config_file = config_file)
    return _REGISTRY


def search_registry(name = None, owner = None, extension = None):
    """ search_registry

        Inputs:
        - name, owner (str or None): case-insensitive substrings to match
        - extension (str or None): ".sacc" or ".txt" to keep only files DEED can
        load, matched against the dataset's relative path

        Process:
        - Queries the working schema's dataset table. The registry's "~=" filter
        is a case-insensitive wildcard match ('*' is the wildcard) and is only
        allowed on name / owner / relative_path, which is why those are the
        search fields. Filters are ANDed. Newest dataset_id first, capped at
        REGISTRY_MAX_ROWS.

        Outputs:
        - Pandas DataFrame with the REGISTRY_COLUMNS (table prefix stripped)
    """
    try:
        return _search_registry(get_registry(), name, owner, extension)
    except Exception:
        ### A long-running server's connection can be dropped by the database
        ### (idle timeout, restart); reconnect once before giving up
        global _REGISTRY
        _REGISTRY = None
        return _search_registry(get_registry(), name, owner, extension)


def _search_registry(registry, name, owner, extension):
    """ _search_registry

        Inputs:
        - registry (dataregistry.DataRegistry)
        - name, owner, extension: as for search_registry

        Outputs:
        - Pandas DataFrame, as for search_registry
    """
    query   = registry.Query
    filters = []
    if name:
        filters.append(query.gen_filter("dataset.name", "~=", f"*{name.strip()}*"))
    if owner:
        filters.append(query.gen_filter("dataset.owner", "~=", f"*{owner.strip()}*"))
    if extension:
        filters.append(query.gen_filter("dataset.relative_path", "~=", f"*{extension}"))

    df = query.find_datasets(REGISTRY_COLUMNS, filters, return_format = "dataframe", strip_table_names = True)

    ### No match comes back as a DataFrame without any columns at all
    if df.empty:
        return pd.DataFrame(columns = [c.split(".", 1)[1] for c in REGISTRY_COLUMNS])
    return df.sort_values("dataset_id", ascending = False).head(REGISTRY_MAX_ROWS).reset_index(drop = True)


def kill_port(port):
    """ kill_port

        Inputs:
        - port (int): local port to free

        Process:
        - Finds any process(es) currently listening on this port (via `lsof`) and
        kills them. In debug mode, Dash's Werkzeug reloader runs as a separate
        child process, so quitting the main process alone can leave the port bound.
        Does nothing where `lsof` isn't installed (e.g. Windows, minimal Linux).

        Outputs:
        - None
    """
    try:
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output = True, text = True)
    except FileNotFoundError:
        return
    pids   = [pid for pid in result.stdout.split() if pid]
    for pid in pids:
        subprocess.run(["kill", "-9", pid])
