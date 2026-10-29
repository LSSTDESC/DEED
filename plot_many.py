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
import os

import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State

### Loading, dataset bookkeeping and the figure builders are shared with the
### multi-page app (app.py), so they live in deed_core; this module keeps only
### its own single-page layout and callbacks
from deed_core import (
    DEFAULT_CHAIN_PARAMS,
    FONT_FAMILY,
    LINESTYLE_NAMES,
    MAX_OVERLAY,
    SYMBOL_NAMES,
    add_dataset,
    check_dataset_extension,
    data_type_label,
    dataset_file_path,
    dataset_kind,
    datasets_of_kind,
    kill_port,
    load_dataset_bundle,
    make_chain_figure,
    make_covariance_figure,
    make_data_vector_figure,
    make_nz_figure,
    message_figure,
    normalize_datasets,
    option_label,
    resolve_registry_path,
    shared_chain_params,
    shared_sacc_contents,
)


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
    parser.add_argument("--host", default = "127.0.0.1",
                        help = "interface to listen on (default 127.0.0.1, this machine only; pass it explicitly "
                               "so Dash doesn't fall back on $HOST, which NERSC sets to the node's hostname)")
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
        app.run(debug=True, host=args.host, port=args.port, use_reloader=False)
    finally:
        kill_port(args.port)


if __name__ == "__main__":
    main()
