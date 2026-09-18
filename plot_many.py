"""Interactive dashboard for comparing several MCMC chains or several .sacc
files.

Multi-dataset sibling of plot.py. Several .txt MCMC chains can be loaded at once
and are overplotted in a single getdist triangle/1D plot, one color (and
linestyle) per chain; several .sacc files can be loaded at once and are
overplotted in the data-vector and n(z) views, one color (and marker symbol) per
file. Datasets are added and removed at runtime -- by local path, or by
dataset_id in the DESC Data Registry -- without restarting.

Covariance matrices are not overplotted -- heatmaps don't overlay -- so that tab
shows one loaded file at a time, chosen from a dropdown.

Usage:
    python plot_many.py --file path/to/chain_a.txt --file path/to/chain_b.txt [--port 8050]
    python plot_many.py --file path/to/a.sacc --file path/to/b.sacc [--port 8050]
    python plot_many.py --registry-id <dataset_id in the DESC Data Registry's working schema> [--port 8050]

--file and --registry-id are both repeatable, and can be mixed.
"""
import argparse
import base64
import functools
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State
import sacc

### getdist_chainplot_utils imports matplotlib.pyplot at import time, so the
### backend has to be chosen before it. Dash serves callbacks on worker threads,
### and the interactive macOS backend refuses to build a figure off the main
### thread; Agg is non-interactive, which is all this app needs, since every
### chain plot is written to a PNG and embedded rather than shown in a window.
### On NERSC this went unnoticed -- with no display, matplotlib already picked Agg.
import matplotlib
matplotlib.use("Agg")

### Jessie Muir's chain-plotting utility lives outside this repo, in her own
### NERSC directory -- not yet vendored in or made configurable, so this will
### need revisiting before this app can run anywhere other than NERSC (e.g. Spin)
CHAINPLOT_UTILS_DIR = "./Muir_plottingFiles"
os.environ["Y6METHODSDIR"] = CHAINPLOT_UTILS_DIR
sys.path.insert(0, CHAINPLOT_UTILS_DIR)
from getdist_chainplot_utils import get_paramdict, make_chainplot

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


def make_nz_figure(saccs, nz_frames, selected_tracers):
    """ make_nz_figure

        Inputs:
        - saccs (list of dict): sacc dataset entries, in load order, supplying
        the label and linestyle each file is drawn with
        - nz_frames (list of Pandas DataFrame): the build_nz_df output for each
        entry in saccs, in the same order, containing z and n(z) per tracer
        - selected_tracers (list of str), the list of tracers whose n(z)s we are plotting

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

    lens_colors = sample_colorscale("Reds",  [0.9 - 0.6 * i / max(len(lens_tracers) - 1, 1) for i in range(len(lens_tracers))])
    src_colors  = sample_colorscale("Blues", [0.9 - 0.6 * i / max(len(src_tracers) - 1, 1) for i in range(len(src_tracers))])
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
        fig.add_vline(x = b - 0.5, line = dict(color = "black", width = 1))
        fig.add_hline(y = b - 0.5, line = dict(color = "black", width = 1))

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
def make_chain_figure(paths, params, labels, colors, linestyles):
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

    with tempfile.TemporaryDirectory() as tmpdir:
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


def make_chain_main_content(chains):
    """ make_chain_main_content

        Inputs:
        - chains (list of dict): chain dataset entries, in load order

        Process:
        - Builds the main-content children for the loaded chains: a checklist of
        the parameters they have in common, and an (initially empty) image that
        the update_chain_plot callback fills in via make_chain_figure.
        - Parameters that only some chains have are still listed, but disabled
        and annotated with which chains lack them, so a column that can't be
        compared is visibly absent rather than silently missing.

        Outputs:
        - children (list of Dash components)
    """
    shared, missing = shared_chain_params(chains)

    if not shared:
        return [html.P("The loaded chains have no parameters in common, so there is nothing to overplot.")]

    ### Default to the headline cosmology parameters when every chain has them
    default = [p for p in DEFAULT_CHAIN_PARAMS if p in shared] or shared[:2]

    options = [{"label": p, "value": p} for p in shared]
    options += [
        {"label": f"{p} (not in {', '.join(missing[p])})", "value": p, "disabled": True}
        for p in missing
    ]

    header = [html.H4(f"Overplotting {len(chains)} chain{'s' if len(chains) != 1 else ''}: {', '.join(d['label'] for d in chains)}")]
    if missing:
        header.append(
            html.P(
                f"{len(missing)} parameter(s) are not in every chain and can't be compared -- "
                f"they are greyed out below.",
                style = {"color": "#52514e"},
            )
        )

    return [
        html.Div(header),
        dcc.Checklist(
            id      = "chain-param-checklist",
            options = options,
            value   = default,
            inline  = True,
            style   = {"marginTop": "10px"},
        ),
        html.Img(id = "chain-graph", style = {"marginTop": "10px"}),
    ]


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


def make_cov_tab_content(saccs):
    """ make_cov_tab_content

        Inputs:
        - saccs (list of dict): sacc dataset entries, in load order

        Process:
        - Builds the Covariance tab's children: a file selector, the mode
        selector and the graph -- or an explanatory message if not one of the
        loaded files has a covariance matrix.
        - Two heatmaps can't be overlaid the way two sets of points or two
        curves can, so this is the one sacc view that shows a single file. The
        selector is rendered even for a single loaded file (hidden), so the
        callback below always has its Input present in the layout.

        Outputs:
        - children (list of Dash components)
    """
    with_cov = [d for d in saccs if load_dataset_bundle(d["path"])[5]]

    if not with_cov:
        return [html.P("No covariance matrix present in the loaded sacc file(s).")]

    return [
        html.Div(
            [
                html.Span("Covariance from: ", style = {"marginRight": "6px"}),
                dcc.Dropdown(
                    id        = "cov-dataset",
                    options   = [{"label": d["label"], "value": d["path"]} for d in with_cov],
                    value     = with_cov[0]["path"],
                    clearable = False,
                    style     = {"width": "500px", "display": "inline-block", "verticalAlign": "middle"},
                ),
            ],
            style = {"marginTop": "10px"} if len(saccs) > 1 else {"display": "none"},
        ),
        dcc.RadioItems(
            id      = "cov-mode",
            options = [
                {"label": "Correlation", "value": "correlation"},
                {"label": "Covariance (signed log10)", "value": "covariance"},
                {"label": "Precision (inverse covariance)", "value": "precision"},
            ],
            value  = "correlation",
            inline = True,
            style  = {"marginTop": "10px"},
        ),
        dcc.Graph(id = "cov-graph"),
    ]


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


def make_sacc_main_content(saccs):
    """ make_sacc_main_content

        Inputs:
        - saccs (list of dict): sacc dataset entries, in load order

        Process:
        - Builds the main-content children for the loaded sacc datasets: a
        summary line per file, and the Data vector / n(z) / Covariance tabs.
        - The data-type dropdown and the tracer checklist offer the union over
        the loaded files, so a file with an extra data type or an extra
        redshift bin doesn't vanish from the controls.

        Outputs:
        - children (list of Dash components)
    """
    data_types, tracers, missing = shared_sacc_contents(saccs)

    summary = []
    for entry in saccs:
        _, df, _, _, file_tracers, has_cov = load_dataset_bundle(entry["path"])
        summary.append(
            html.P(
                [
                    html.Span(entry["label"], style = {"fontWeight": "600"}),
                    html.Span(
                        f" -- {len(df)} data points, {len(file_tracers)} tracers, "
                        f"covariance: {has_cov} -- {entry['path']}",
                    ),
                ],
                style = {"margin": "2px 0"},
            )
        )

    return [
        html.Div(
            [
                html.H4(f"{len(saccs)} sacc file{'s' if len(saccs) != 1 else ''} loaded"),
                *summary,
            ],
        ),
        dcc.Tabs(
            [
                dcc.Tab(
                    label    = "Data vector",
                    children = [
                        dcc.Dropdown(
                            id        = "data-type-dropdown",
                            options   = [
                                {"label": option_label(dt, missing, data_type_label(dt)), "value": dt}
                                for dt in data_types
                            ],
                            value     = data_types[0] if data_types else None,
                            clearable = False,
                            style     = {"width": "500px", "marginTop": "10px"},
                        ),
                        dcc.Graph(id = "data-vector-graph"),
                    ],
                ),
                dcc.Tab(
                    label    = "n(z)",
                    children = [
                        dcc.Checklist(
                            id      = "tracer-checklist",
                            options = [{"label": option_label(t, missing), "value": t} for t in tracers],
                            value   = tracers,
                            inline  = True,
                            style   = {"marginTop": "10px"},
                        ),
                        dcc.Graph(id = "nz-graph"),
                    ],
                ),
                dcc.Tab(
                    label    = "Covariance",
                    children = html.Div(
                        id       = "cov-tab-content",
                        children = make_cov_tab_content(saccs),
                    ),
                ),
            ]
        ),
    ]


def make_dataset_panel(datasets):
    """ make_dataset_panel

        Inputs:
        - datasets (list of dict): dataset entries

        Process:
        - Lists the loaded datasets with a Remove button each, showing the color
        and second style channel each is drawn with -- linestyle for a chain,
        marker symbol for a sacc file -- so the panel doubles as a legend for
        both overlays. Since it names every dataset in text, identity never
        rests on color alone.

        Outputs:
        - children (list of Dash components)
    """
    if not datasets:
        return [html.P("No datasets loaded.")]

    rows = []
    for i, d in enumerate(datasets):

        ### A sacc file carries both channels -- the symbol in the data-vector
        ### view, the linestyle in n(z) -- so the panel names both. (The
        ### covariance view is the one place a sacc file isn't drawn in its own
        ### color; it names the file in its title instead.)
        channel = (
            f"{LINESTYLE_NAMES[d['linestyle']]} {SYMBOL_NAMES[d['symbol']]}"
            if d["kind"] == "sacc"
            else LINESTYLE_NAMES[d["linestyle"]]
        )

        swatch = [
            html.Span(
                style = {
                    "display":         "inline-block",
                    "width":           "14px",
                    "height":          "14px",
                    "backgroundColor": d["color"],
                    "border":          "1px solid #c3c2b7",
                    "marginRight":     "8px",
                    "verticalAlign":   "middle",
                },
            ),
            html.Span(
                f"{channel}, ",
                style = {"color": "#898781"},
            ),
        ]

        rows.append(
            html.Div(
                [
                    *swatch,
                    html.Span(d["label"], style = {"fontWeight": "600"}),
                    html.Span(f" ({d['kind']}, from {d['origin']})", style = {"color": "#52514e"}),
                    html.Button(
                        "Remove",
                        id       = {"type": "dataset-remove", "index": i},
                        n_clicks = 0,
                        style    = {"marginLeft": "10px"},
                    ),
                ],
                style = {"marginTop": "4px"},
            )
        )

    return [html.H4("Loaded datasets"), *rows]


def make_main_content(datasets):
    """ make_main_content

        Inputs:
        - datasets (list of dict): dataset entries, or empty if nothing has
          been loaded yet

        Process:
        - Builds the main-content children for whatever is loaded: one section
        overplotting every loaded chain, and one overplotting every loaded sacc
        file. Both can be present at once, so a mixed set of datasets degrades
        to two stacked sections rather than an error.

        Outputs:
        - children (list of Dash components)
    """
    chains = datasets_of_kind(datasets, "chain")
    saccs  = datasets_of_kind(datasets, "sacc")

    if not chains and not saccs:
        return [html.P("Add a dataset above -- a local .sacc/.txt file, or a dataset_id from the DESC Data Registry.")]

    children = []

    if chains:
        children += make_chain_main_content(chains)

    if saccs:
        if chains:
            children.append(html.Hr())
        children += make_sacc_main_content(saccs)

    return children


### Build the interactive app
def build_app(initial_datasets=None):
    """ build_app

        Inputs:
        - initial_datasets (list of str or dict, or None): datasets to load at
        startup. If empty, the app boots with nothing loaded -- just the two
        "add a dataset" boxes.

        Process:
        - Builds the Dash app layout and callbacks. The loaded datasets live in
        a dcc.Store, so they can change at runtime without restarting the app:
        one callback owns every write to that store (add by path, add by
        registry id, remove), and callbacks keyed off it rebuild the
        loaded-datasets panel and the whole main-content area. The figure
        callbacks read the datasets they need straight out of the store.

        Outputs:
        - app (dash.Dash)
    """
    datasets = normalize_datasets(initial_datasets)
    app = dash.Dash(
        __name__,
        external_stylesheets = ["https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap"],
    )
    app.title = "DESC Dark Energy Exploration Dashboard"

    ### main-content's children depend on which dataset kind is loaded, and can
    ### be swapped by the update_main_content callback below, so Dash's startup
    ### validation (which can't see ids that come and go dynamically) is disabled
    app.config.suppress_callback_exceptions = True

    app.layout = html.Div(
        [
            html.H2("DESC Dark Energy Exploration Dashboard"),

            ### Datasets are added at runtime, either by local path or by
            ### dataset_id in the DESC Data Registry (https://desc-dataregistry.lbl.gov)
            ### working schema -- the primary key, so it's unique by construction.
            ### Chains accumulate, and every loaded chain is overplotted.
            html.Div(
                [
                    dcc.Input(
                        id          = "file-path-input",
                        type        = "text",
                        placeholder = "path to a local .sacc file or .txt MCMC chain",
                        style       = {"width": "420px"},
                    ),
                    html.Button("Add file", id = "file-add-button", n_clicks = 0),
                ],
                style = {"marginTop": "10px"},
            ),
            html.Div(
                [
                    dcc.Input(
                        id          = "registry-id-input",
                        type        = "number",
                        step        = 1,
                        placeholder = "dataset_id (working schema) in DESC Data Registry",
                        style       = {"width": "420px"},
                    ),
                    html.Button("Add from Registry", id = "registry-load-button", n_clicks = 0),
                ],
                style = {"marginTop": "6px"},
            ),
            html.Div(id = "registry-status", style = {"color": "crimson"}),

            dcc.Store(id = "dataset-store", data = datasets),

            html.Div(id = "dataset-panel",  children = make_dataset_panel(datasets), style = {"marginTop": "10px"}),
            html.Div(id = "main-content",   children = make_main_content(datasets)),
        ],
        style = {"fontFamily": FONT_FAMILY},
    )

    @app.callback(
        Output("dataset-panel", "children"),
        Input("dataset-store", "data"),
        prevent_initial_call = True,
    )
    def update_dataset_panel(datasets):
        return make_dataset_panel(datasets or [])

    @app.callback(
        Output("main-content", "children"),
        Input("dataset-store", "data"),
        prevent_initial_call = True,
    )
    def update_main_content(datasets):
        return make_main_content(datasets or [])

    @app.callback(
        Output("data-vector-graph", "figure"),
        Input("data-type-dropdown", "value"),
        Input("dataset-store", "data"),
    )
    def update_data_vector(data_type, datasets):
        saccs = datasets_of_kind(datasets, "sacc")
        if not saccs or not data_type:
            return go.Figure()
        frames = [load_dataset_bundle(d["path"])[1] for d in saccs]
        return make_data_vector_figure(saccs, frames, data_type)

    @app.callback(
        Output("nz-graph", "figure"),
        Input("tracer-checklist", "value"),
        Input("dataset-store", "data"),
    )
    def update_nz(selected_tracers, datasets):
        saccs = datasets_of_kind(datasets, "sacc")
        if not saccs:
            return go.Figure()
        nz_frames = [load_dataset_bundle(d["path"])[2] for d in saccs]
        return make_nz_figure(saccs, nz_frames, selected_tracers or [])

    @app.callback(
        Output("cov-graph", "figure"),
        Input("cov-mode", "value"),
        Input("cov-dataset", "value"),
        Input("dataset-store", "data"),
    )
    def update_cov(mode, cov_path, datasets):
        saccs = datasets_of_kind(datasets, "sacc")
        if not saccs:
            return go.Figure()

        ### The selector is rebuilt whenever the store changes, so a removal can
        ### leave this callback holding a path that is no longer loaded -- fall
        ### back to the first loaded file rather than reading a stale one
        entry = next((d for d in saccs if d["path"] == cov_path), saccs[0])

        s, df, _, _, _, has_cov = load_dataset_bundle(entry["path"])
        if not has_cov:
            return message_figure(f"No covariance matrix present in {entry['label']}.")
        return make_covariance_figure(s, df, mode, label = entry["label"] if len(saccs) > 1 else None)

    @app.callback(
        Output("chain-graph", "src"),
        Input("chain-param-checklist", "value"),
        Input("dataset-store", "data"),
    )
    def update_chain_plot(selected_params, datasets):
        chains = datasets_of_kind(datasets, "chain")
        if not chains or not selected_params:
            return ""

        ### Tuples, because make_chain_figure is cached
        return make_chain_figure(
            paths      = tuple(d["path"]      for d in chains),
            params     = tuple(selected_params),
            labels     = tuple(d["label"]     for d in chains),
            colors     = tuple(d["color"]     for d in chains),
            linestyles = tuple(d["linestyle"] for d in chains),
        )

    ### One callback owns every write to dataset-store: two callbacks writing the
    ### same Output isn't allowed, and add/remove have to see the same list
    @app.callback(
        Output("dataset-store", "data"),
        Output("registry-status", "children"),
        Input("file-add-button", "n_clicks"),
        Input("registry-load-button", "n_clicks"),
        Input({"type": "dataset-remove", "index": ALL}, "n_clicks"),
        State("file-path-input", "value"),
        State("registry-id-input", "value"),
        State("dataset-store", "data"),
        prevent_initial_call = True,
    )
    def update_datasets(file_clicks, registry_clicks, remove_clicks, file_path, dataset_id, datasets):
        """ update_datasets

            Adds a dataset by local path or by DESC Data Registry dataset_id, or
            removes one, depending on which control fired. Returns the new store
            contents and a status message.
        """
        datasets  = list(datasets or [])
        triggered = dash.callback_context.triggered_id

        ### Removal -- triggered_id is the dict id of the button that was clicked.
        ### Rendering the panel creates these buttons, which fires this callback
        ### with every n_clicks still 0, so check the click count before acting.
        if isinstance(triggered, dict) and triggered.get("type") == "dataset-remove":
            index = triggered["index"]
            if not (0 <= index < len(datasets)) or not (remove_clicks and remove_clicks[index]):
                return dash.no_update, dash.no_update
            dropped = datasets.pop(index)
            return normalize_datasets(datasets), f"Removed {dropped['label']}."

        if triggered == "file-add-button":
            if not file_path:
                return dash.no_update, "Enter a file path to add."

            path = os.path.abspath(os.path.expanduser(file_path.strip()))
            try:
                check_dataset_extension(path)
            except ValueError as e:
                return dash.no_update, f"Error: {e}"
            if not os.path.isfile(path):
                return dash.no_update, f"Error: no such file: {path}"

            return add_dataset(datasets, path, origin = "file")

        if triggered == "registry-load-button":
            if dataset_id is None:
                return dash.no_update, "Enter a dataset_id to look up."

            try:
                dataset_id = int(dataset_id)
            except (TypeError, ValueError):
                return dash.no_update, f"dataset_id must be an integer, got: {dataset_id!r}"

            try:
                path = resolve_registry_path(dataset_id)
            except Exception as e:
                return dash.no_update, f"Error: {e}"

            return add_dataset(datasets, path, origin = f"registry:{dataset_id}")

        return dash.no_update, dash.no_update

    return app


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


def kill_port(port):
    """ kill_port

        Inputs:
        - port (int): local port to free

        Process:
        - Finds any process(es) currently listening on this port (via `lsof`) and
        kills them. In debug mode, Dash's Werkzeug reloader runs as a separate
        child process, so quitting the main process alone can leave the port bound.

        Outputs:
        - None
    """
    result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output = True, text = True)
    pids   = [pid for pid in result.stdout.split() if pid]
    for pid in pids:
        subprocess.run(["kill", "-9", pid])


def main():
    parser = argparse.ArgumentParser()

    ### Both are repeatable and can be mixed, so several chains can be
    ### overplotted from the command line: --file a.txt --file b.txt
    parser.add_argument("--file", action = "append", default = [], type = dataset_file_path,
                        help = "path to a local .sacc file or a .txt MCMC chain file (repeatable)")
    parser.add_argument("--registry-id", action = "append", default = [], type = int,
                        help = "dataset_id (in the working schema) to look up in the DESC Data Registry (repeatable)")
    parser.add_argument("--registry-config", default = None, help = "optional dataregistry config file (defaults to NERSC config resolution)")
    parser.add_argument("--port", type = int, default = 8050)
    args = parser.parse_args()

    initial_datasets = [{"path": path, "origin": "file"} for path in args.file]

    for dataset_id in args.registry_id:
        try:
            path = resolve_registry_path(dataset_id, config_file = args.registry_config)
        except Exception as e:
            parser.error(f"--registry-id {dataset_id} lookup failed: {e}")
        initial_datasets.append({"path": path, "origin": f"registry:{dataset_id}"})

    for kind, cap in MAX_OVERLAY.items():
        given = [d for d in initial_datasets if dataset_kind(d["path"]) == kind]
        if len(given) > cap:
            parser.error(
                f"{len(given)} {kind} datasets given, but at most {cap} can be overplotted "
                f"distinguishably (one per color/linestyle pair)"
            )

    app = build_app(initial_datasets)
    try:
        ### use_reloader=False: Werkzeug's reloader runs the server in a second,
        ### child process that (per testing) ignores a solitary SIGINT sent to just
        ### the parent -- it relies on the terminal delivering Ctrl+C to the whole
        ### process group. Disabling it keeps this to one process, so kill_port
        ### below reliably runs on any exit path (Ctrl+C, error, or normal return).
        app.run(debug=True, port=args.port, use_reloader=False)
    finally:
        kill_port(args.port)


if __name__ == "__main__":
    main()
