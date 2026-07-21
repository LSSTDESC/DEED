"""Interactive dashboard for exploring the contents of a .sacc file.

Usage:
    python plot.py --file path/to/data.sacc [--port 8050]
    python plot.py --registry-id <dataset_id in the DESC Data Registry's working schema> [--port 8050]
"""
import argparse
import functools
import re
import subprocess

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import sacc

FONT_FAMILY = "'Source Sans 3', 'Source Sans Pro', sans-serif"

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


def make_data_vector_figure(df, data_type):
    """ make_data_vector_figure
    
        Inputs:
        - df (Pandas DataFrame)
        - data_type (str)
        
        Process:
        - 
        
        Outputs:
        - 
    """
    
    ### Pick out the dataframe corresponding to data_type, and pick out the tracers
    sub = df[df["data_type"] == data_type]
    t1s = sorted(sub["tracer1"].unique(), key = natural_key)
    t2s = sorted(sub["tracer2"].unique(), key = natural_key)

    ### Check if data are in real or Fourier space
    ### x_name/y_name are plain ASCII for hovertemplates; x_label/y_label are the
    ### nicer Unicode versions used for the actual axis titles
    real_space = "_xi"   in data_type
    x_name     = "theta" if real_space else "ell"
    y_name     = "xi"    if real_space else "Cl"
    x_label    = "θ"  if real_space else "ℓ"
    y_label    = "ξ"  if real_space else "Cℓ"

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

            panel = sub[(sub["tracer1"] == t1) & (sub["tracer2"] == t2)]

            if panel.empty:
                continue

            bottom_row_for_col[j] = max(bottom_row_for_col.get(j, -1), i)
            left_col_for_row[i]   = min(left_col_for_row.get(i, len(t2s)), j)

            ### Pick positive and negative values
            pos = panel[panel["value"] >= 0]
            neg = panel[panel["value"] < 0]
            
            ### Plot out positive values
            if not pos.empty:
                fig.add_trace(
                    go.Scatter(
                        x             = pos["x"],
                        y             = pos["value"],
                        error_y       = dict(type = "data", array = pos["err"], visible = True),
                        mode          = "markers",
                        marker        = dict(color = "royalblue", size = 6, symbol = "circle"),
                        name          = f"{t1}-{t2} (+)",
                        showlegend    = False,
                        hovertemplate = f"{x_name}=%{{x:.1f}}<br>{y_name}=%{{y:.3e}}<extra></extra>",
                    ),
                    row = i + 1,
                    col = j + 1,
                )
                
            ### Plot out negative values
            if not neg.empty:
                fig.add_trace(
                    go.Scatter(
                        x             = neg["x"],
                        y             = -neg["value"],
                        error_y       = dict(type = "data", array = neg["err"], visible = True),
                        mode          = "markers",
                        marker        = dict(color = "crimson", size = 6, symbol = "circle-open"),
                        name          = f"{t1}-{t2} (-)",
                        showlegend    = False,
                        hovertemplate = f"{x_name}=%{{x:.1f}}<br>-{y_name}=%{{y:.3e}}<extra></extra>",
                    ),
                    row = i + 1,
                    col = j + 1,
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
    fig.update_layout(
        height = max(220 * len(t1s), 350),
        width  = max(220 * len(t2s), 450),
        title  = f"{data_type_label(data_type)} (filled = positive, open = |negative|)",
        margin = dict(t = 80, l = 60, r = 20, b = 40),
    )
    return fig


def make_nz_figure(nz_df, selected_tracers):
    """ make_nz_figure
    
        Inputs:
        - nz_df (Pandas DataFrame), containing the z and n(z) for each tracer
        - selected_tracers (list of str), the list of tracers whose n(z)s we are plotting
        
        Outputs:
        - fig (plotly go.Figure)
    
    """

    ### Build a consistent colormap: lens tracers drawn from a reddish scale, others from a bluish scale
    all_tracers  = sorted(nz_df["tracer"].unique(), key = natural_key)
    lens_tracers = [name for name in all_tracers if "lens" in name.lower()]
    src_tracers  = [name for name in all_tracers if name not in lens_tracers]

    lens_colors = sample_colorscale("Reds",  [0.9 - 0.6 * i / max(len(lens_tracers) - 1, 1) for i in range(len(lens_tracers))])
    src_colors  = sample_colorscale("Blues", [0.9 - 0.6 * i / max(len(src_tracers) - 1, 1) for i in range(len(src_tracers))])
    color_map   = dict(zip(lens_tracers, lens_colors)) | dict(zip(src_tracers, src_colors))

    ### Build the figure
    fig = go.Figure()
    for name in selected_tracers:

        sub = nz_df[nz_df["tracer"] == name]
        fig.add_trace(
            go.Scatter(
                x    = sub["z"],
                y    = sub["nz"],
                mode = "lines",
                name = name,
                line = dict(color = color_map[name], dash = "solid" if name in lens_tracers else "dash"),
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


def make_covariance_figure(s, df, mode):
    """ make_covariance_figure

        Inputs:
        - s (sacc file loaded into memory, sacc.sacc.Sacc)
        - df (Pandas DataFrame), output of build_data_vector_df, used for axis labels and block boundaries
        - mode (str), one of "correlation", "covariance", "precision"

        Process:
        - Builds the requested matrix representation of the covariance

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
            fig = go.Figure()
            fig.add_annotation(
                text      = (
                    f"Precision matrix not shown: covariance is too ill-conditioned to invert "
                    f"reliably (condition number = {cond:.2e})."
                ),
                showarrow = False,
                xref      = "paper",
                yref      = "paper",
                x         = 0.5,
                y         = 0.5,
                font      = dict(size = 14),
            )
            fig.update_layout(height = 700, width = 750)
            return fig

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
        title  = title,
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

@functools.lru_cache(maxsize = 4)
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


def make_cov_tab_content(has_cov):
    """ make_cov_tab_content

        Inputs:
        - has_cov (bool)

        Process:
        - Builds the Covariance tab's children: the mode selector + graph if the
        dataset has a covariance matrix, or an explanatory message if it doesn't

        Outputs:
        - children (list of Dash components)
    """
    return (
        [
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
        if has_cov
        else [html.P("No covariance matrix present in this sacc file.")]
    )


### Build the interactive app
def build_app(sacc_path=None):
    """ build_app

        Inputs:
        - sacc_path (str or None): path to the .sacc file to load at startup.
        If None, the app boots with no dataset loaded -- just the "Load from
        Registry" box -- until one is loaded from the DESC Data Registry.

        Process:
        - Builds the Dash app layout and callbacks. The currently-loaded dataset's
        path lives in a dcc.Store, so it can change at runtime (via the "Load from
        Registry" box) without restarting the app: a callback keyed off the store
        rebuilds the dropdown/checklist/covariance-tab options, and the three figure
        callbacks reload their data from the store's current path.

        Outputs:
        - app (dash.Dash)
    """
    if sacc_path:
        _, df, _, data_types, tracers, has_cov = load_dataset_bundle(sacc_path)
        dataset_info_children = [
            html.H4(f"From file: {sacc_path}"),
            html.P(f"Found {len(df)} data points, {len(tracers)} tracers, covariance: {has_cov}"),
        ]
    else:
        data_types, tracers, has_cov = [], [], False
        dataset_info_children = [
            html.H4("No dataset loaded"),
            html.P("Use the box above to load a dataset from the DESC Data Registry."),
        ]

    app = dash.Dash(
        __name__,
        external_stylesheets = ["https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap"],
    )
    app.title = "sacc dashboard"

    ### The Covariance tab's children get swapped in by a callback after the initial
    ### layout (has_cov can change between datasets), which Dash's startup validation
    ### would otherwise reject since it can't see those ids in the initial layout
    app.config.suppress_callback_exceptions = True

    app.layout = html.Div(
        [
            html.H2("DESC Dark Energy Exploration Dashboard"),

            ### Query the DESC Data Registry (https://desc-dataregistry.lbl.gov) for a
            ### different dataset, by dataset_id in the working schema (the primary
            ### key, so it's unique by construction), without restarting the app
            html.Div(
                [
                    dcc.Input(
                        id          = "registry-id-input",
                        type        = "number",
                        step        = 1,
                        placeholder = "dataset_id (working schema) in DESC Data Registry",
                        style       = {"width": "350px"},
                    ),
                    html.Button("Load from Registry", id = "registry-load-button", n_clicks = 0),
                ],
                style = {"marginTop": "10px"},
            ),
            html.Div(id = "registry-status", style = {"color": "crimson"}),

            html.Div(
                id       = "dataset-info",
                children = dataset_info_children,
            ),

            dcc.Store(id = "sacc-path-store", data = sacc_path),

            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Data vector",
                        children=[
                            dcc.Dropdown(
                                id="data-type-dropdown",
                                options=[{"label": data_type_label(dt), "value": dt} for dt in data_types],
                                value=data_types[0] if data_types else None,
                                clearable=False,
                                style={"width": "500px", "marginTop": "10px"},
                            ),
                            dcc.Graph(id="data-vector-graph"),
                        ],
                    ),
                    dcc.Tab(
                        label="n(z)",
                        children=[
                            dcc.Checklist(
                                id="tracer-checklist",
                                options=[{"label": t, "value": t} for t in tracers],
                                value=tracers,
                                inline=True,
                                style={"marginTop": "10px"},
                            ),
                            dcc.Graph(id="nz-graph"),
                        ],
                    ),
                    dcc.Tab(
                        label    = "Covariance",
                        children = html.Div(
                            id       = "cov-tab-content",
                            children = make_cov_tab_content(has_cov) if sacc_path else [html.P("No dataset loaded.")],
                        ),
                    ),
                ]
            ),
        ],
        style = {"fontFamily": FONT_FAMILY},
    )

    @app.callback(
        Output("data-vector-graph", "figure"),
        Input("data-type-dropdown", "value"),
        Input("sacc-path-store", "data"),
    )
    def update_data_vector(data_type, sacc_path):
        if not sacc_path or not data_type:
            return go.Figure()
        _, df, _, _, _, _ = load_dataset_bundle(sacc_path)
        return make_data_vector_figure(df, data_type)

    @app.callback(
        Output("nz-graph", "figure"),
        Input("tracer-checklist", "value"),
        Input("sacc-path-store", "data"),
    )
    def update_nz(selected_tracers, sacc_path):
        if not sacc_path:
            return go.Figure()
        _, _, nz_df, _, _, _ = load_dataset_bundle(sacc_path)
        return make_nz_figure(nz_df, selected_tracers or [])

    @app.callback(
        Output("cov-graph", "figure"),
        Input("cov-mode", "value"),
        Input("sacc-path-store", "data"),
    )
    def update_cov(mode, sacc_path):
        if not sacc_path:
            return go.Figure()
        s, df, _, _, _, has_cov = load_dataset_bundle(sacc_path)
        if not has_cov:
            return dash.no_update
        return make_covariance_figure(s, df, mode)

    @app.callback(
        Output("dataset-info", "children"),
        Output("data-type-dropdown", "options"),
        Output("data-type-dropdown", "value"),
        Output("tracer-checklist", "options"),
        Output("tracer-checklist", "value"),
        Output("cov-tab-content", "children"),
        Input("sacc-path-store", "data"),
        prevent_initial_call = True,
    )
    def update_dataset_controls(sacc_path):
        _, df, _, data_types, tracers, has_cov = load_dataset_bundle(sacc_path)

        info = [
            html.H4(f"From file: {sacc_path}"),
            html.P(f"Found {len(df)} data points, {len(tracers)} tracers, covariance: {has_cov}"),
        ]
        dropdown_options  = [{"label": data_type_label(dt), "value": dt} for dt in data_types]
        checklist_options = [{"label": t, "value": t} for t in tracers]

        return info, dropdown_options, data_types[0], checklist_options, tracers, make_cov_tab_content(has_cov)

    @app.callback(
        Output("sacc-path-store", "data"),
        Output("registry-status", "children"),
        Input("registry-load-button", "n_clicks"),
        State("registry-id-input", "value"),
        prevent_initial_call = True,
    )
    def load_from_registry(n_clicks, dataset_id):
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

        return path, f"Loaded dataset_id={dataset_id} from the DESC Data Registry (working schema)."

    return app


def check_sacc_extension(path):
    """ check_sacc_extension

        Inputs:
        - path (str): candidate path to a .sacc file

        Process:
        - Checks that path has the .sacc extension

        Outputs:
        - path (str), unchanged, if valid
    """
    if not path.endswith(".sacc"):
        raise ValueError(f"expected a .sacc file, got: {path!r}")
    return path


def sacc_file_path(path):
    """ sacc_file_path

        argparse `type=` wrapper around check_sacc_extension, for --file validation
    """
    try:
        return check_sacc_extension(path)
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
        - Checks that the resolved path is a .sacc file

        Outputs:
        - path (str): absolute path of the registered dataset
    """
    from dataregistry import DataRegistry

    registry = DataRegistry(config_file = config_file)
    path     = registry.get_dataset_absolute_path(dataset_id, schema = "working", silent = False)
    return check_sacc_extension(path)


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
    group  = parser.add_mutually_exclusive_group(required = False)
    group.add_argument("--file", type = sacc_file_path, help = "path to a local .sacc file")
    group.add_argument("--registry-id", type = int, help = "dataset_id (in the working schema) to look up in the DESC Data Registry")
    parser.add_argument("--registry-config", default = None, help = "optional dataregistry config file (defaults to NERSC config resolution)")
    parser.add_argument("--port", type = int, default = 8050)
    args = parser.parse_args()

    if args.registry_id is not None:
        try:
            sacc_path = resolve_registry_path(args.registry_id, config_file = args.registry_config)
        except Exception as e:
            parser.error(f"--registry-id lookup failed: {e}")
    else:
        sacc_path = args.file

    app = build_app(sacc_path)
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
