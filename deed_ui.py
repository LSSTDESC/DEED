"""Layout pieces shared by the pages of the multi-page DEED app (app.py, pages/).

The figures themselves come from deed_core; this module wraps them in
dash-bootstrap-components layout (cards, tabs, form controls) and registers the
callbacks that redraw them. Two pages show the same kinds of view -- /compare
overlays every loaded dataset, /explore shows one -- so the view builders and
their callbacks are parametrized by an id prefix ("cmp", "exp") and by which
loaded datasets the view draws.
"""
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

from deed_core import (
    DEFAULT_CHAIN_PARAMS,
    LINESTYLE_NAMES,
    SYMBOL_NAMES,
    apply_theme,
    data_type_label,
    datasets_of_kind,
    load_dataset_bundle,
    make_chain_figure,
    make_covariance_figure,
    make_data_vector_figure,
    make_nz_figure,
    message_figure,
    option_label,
    shared_chain_params,
    shared_sacc_contents,
)

### Plotly's modebar without the logo, and figures that resize with their card
GRAPH_CONFIG = {"displaylogo": False, "responsive": True}


def theme_of(dark):
    """ theme_of

        Inputs:
        - dark (bool or None): the value of the navbar's theme-switch

        Outputs:
        - "dark" or "light"
    """
    return "dark" if dark else "light"


def card(title, children, subtitle = None, className = ""):
    """ card

        Inputs:
        - title (str or component): card header text
        - children (list of components): card body
        - subtitle (str or None): muted line under the title
        - className (str): extra classes for the card

        Outputs:
        - dbc.Card
    """
    header = [html.Div(title, className = "fw-semibold")]
    if subtitle:
        header.append(html.Div(subtitle, className = "small text-body-secondary"))
    return dbc.Card(
        [dbc.CardHeader(header), dbc.CardBody(children)],
        className = f"deed-card mb-4 {className}".strip(),
    )


def empty_state(text):
    """ empty_state

        Inputs:
        - text (str): what to tell the user

        Outputs:
        - a muted, centered placeholder (html.Div)
    """
    return html.Div(text, className = "deed-empty text-body-secondary")


def status_message(text):
    """ status_message

        Inputs:
        - text (str or None): status line from adding / removing a dataset

        Process:
        - Messages starting with "Error" are shown in the danger color, anything
        else muted, so a failed add is noticeable without a modal

        Outputs:
        - html.Div, or None for no message
    """
    if not text:
        return None
    is_error = text.startswith("Error") or "must be" in text
    return html.Div(text, className = "text-danger" if is_error else "text-body-secondary")


def swatch(entry):
    """ swatch

        Inputs:
        - entry (dict): a dataset entry

        Outputs:
        - a small square in the dataset's color (html.Span)
    """
    return html.Span(className = "deed-swatch", style = {"backgroundColor": entry["color"]})


def origin_text(origin):
    """ origin_text

        Inputs:
        - origin (str): "file" or "registry:<dataset_id>"

        Outputs:
        - how the sidebar describes where a dataset came from (str)
    """
    if origin.startswith("registry:"):
        return f"Data Registry, dataset_id {origin.split(':', 1)[1]}"
    return "local file"


def make_dataset_list(datasets):
    """ make_dataset_list

        Inputs:
        - datasets (list of dict): dataset entries

        Process:
        - The sidebar's list of loaded datasets: color swatch, label, kind, the
        second style channel each is drawn with (linestyle for a chain; marker
        symbol and n(z) linestyle for a sacc file), where it came from, and a
        remove button. It doubles as the legend for every overlay, and names each
        dataset in text so identity never rests on color alone.

        Outputs:
        - children (list of components)
    """
    if not datasets:
        return [empty_state("Nothing loaded yet.")]

    items = []
    for i, d in enumerate(datasets):
        channel = (
            f"{SYMBOL_NAMES[d['symbol']]}, {LINESTYLE_NAMES[d['linestyle']]} n(z)"
            if d["kind"] == "sacc"
            else f"{LINESTYLE_NAMES[d['linestyle']]} contours"
        )
        items.append(
            dbc.ListGroupItem(
                html.Div(
                    [
                        swatch(d),
                        html.Div(
                            [
                                html.Div(d["label"], className = "fw-semibold text-break"),
                                html.Div(
                                    [dbc.Badge(d["kind"], color = "secondary", className = "me-1"), channel],
                                    className = "small text-body-secondary",
                                ),
                                html.Div(origin_text(d["origin"]), className = "small text-body-secondary text-break"),
                            ],
                            className = "flex-grow-1 min-w-0",
                        ),
                        dbc.Button(
                            "×",
                            id        = {"type": "dataset-remove", "index": i},
                            n_clicks  = 0,
                            color     = "link",
                            size      = "sm",
                            className = "deed-remove text-body-secondary",
                            title     = f"Remove {d['label']}",
                        ),
                    ],
                    className = "d-flex align-items-start gap-2",
                ),
                className = "px-2 py-2",
            )
        )
    return [dbc.ListGroup(items, flush = True)]


### ------------------------------------------------------------ sacc views

def make_cov_controls(saccs, prefix):
    """ make_cov_controls

        Inputs:
        - saccs (list of dict): sacc dataset entries
        - prefix (str): id prefix of the page

        Process:
        - The Covariance tab: a file selector (hidden when only one file is
        loaded, but always present so the callback's Input exists), the matrix
        mode, and the graph. Heatmaps don't overlay, so this view shows one file.

        Outputs:
        - children (list of components)
    """
    with_cov = [d for d in saccs if load_dataset_bundle(d["path"])[5]]
    if not with_cov:
        return [empty_state("No covariance matrix in the loaded sacc file(s).")]

    return [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("File", html_for = f"{prefix}-cov-dataset", className = "small mb-1"),
                        dbc.Select(
                            id      = f"{prefix}-cov-dataset",
                            options = [{"label": d["label"], "value": d["path"]} for d in with_cov],
                            value   = with_cov[0]["path"],
                            size    = "sm",
                        ),
                    ],
                    md        = 5,
                    className = "" if len(saccs) > 1 else "d-none",
                ),
                dbc.Col(
                    [
                        dbc.Label("Show", className = "small mb-1"),
                        dbc.RadioItems(
                            id      = f"{prefix}-cov-mode",
                            options = [
                                {"label": "Correlation", "value": "correlation"},
                                {"label": "Covariance (signed log10)", "value": "covariance"},
                                {"label": "Precision", "value": "precision"},
                            ],
                            value   = "correlation",
                            inline  = True,
                        ),
                    ],
                ),
            ],
            className = "deed-controls g-3",
        ),
        dcc.Loading(dcc.Graph(id = f"{prefix}-cov-graph", config = GRAPH_CONFIG), type = "dot"),
    ]


def make_sacc_tabs(saccs, prefix):
    """ make_sacc_tabs

        Inputs:
        - saccs (list of dict): the sacc dataset entries this view draws
        - prefix (str): id prefix of the page

        Process:
        - Data vector / n(z) / Covariance tabs. The data-type selector and the
        tracer checklist offer the union over the files, annotated with which
        files lack an option (see shared_sacc_contents), so nothing vanishes from
        the controls because one file doesn't have it.

        Outputs:
        - dbc.Tabs
    """
    data_types, tracers, missing = shared_sacc_contents(saccs)

    data_vector_tab = [
        dbc.Row(
            dbc.Col(
                [
                    dbc.Label("Data type", html_for = f"{prefix}-data-type", className = "small mb-1"),
                    dbc.Select(
                        id      = f"{prefix}-data-type",
                        options = [{"label": option_label(dt, missing, data_type_label(dt)), "value": dt} for dt in data_types],
                        value   = data_types[0] if data_types else None,
                        size    = "sm",
                    ),
                ],
                md = 6,
            ),
            className = "deed-controls",
        ),
        ### The triangular grid has a fixed size per panel, so it scrolls
        ### sideways in a narrow window rather than squashing its panels
        dcc.Loading(
            html.Div(dcc.Graph(id = f"{prefix}-dv-graph", config = GRAPH_CONFIG), className = "deed-figure-scroll"),
            type = "dot",
        ),
    ]

    nz_tab = [
        html.Div(
            [
                dbc.Label("Tracers", className = "small mb-1"),
                dbc.Checklist(
                    id      = f"{prefix}-tracers",
                    options = [{"label": option_label(t, missing), "value": t} for t in tracers],
                    value   = tracers,
                    inline  = True,
                ),
            ],
            className = "deed-controls",
        ),
        dcc.Graph(id = f"{prefix}-nz-graph", config = GRAPH_CONFIG),
    ]

    return dbc.Tabs(
        [
            dbc.Tab(data_vector_tab, label = "Data vector", tab_id = "dv"),
            dbc.Tab(nz_tab, label = "n(z)", tab_id = "nz"),
            dbc.Tab(make_cov_controls(saccs, prefix), label = "Covariance", tab_id = "cov"),
        ],
        active_tab = "dv",
        className  = "mb-3",
    )


### ----------------------------------------------------------- chain views

def make_chain_view(chains, prefix, with_summary = False):
    """ make_chain_view

        Inputs:
        - chains (list of dict): the chain dataset entries this view draws
        - prefix (str): id prefix of the page
        - with_summary (bool): add a numeric summary table under the plot (the
        /explore page, which shows one chain)

        Process:
        - A checklist of the parameters every chain has (parameters only some
        chains have are listed but disabled, naming who lacks them -- getdist
        would otherwise silently drop them from one chain) and the getdist
        image, filled by the callback.

        Outputs:
        - children (list of components)
    """
    shared, missing = shared_chain_params(chains)
    if not shared:
        return [empty_state("The loaded chains have no parameters in common, so there is nothing to overplot.")]

    default = [p for p in DEFAULT_CHAIN_PARAMS if p in shared] or shared[:2]
    options = [{"label": p, "value": p} for p in shared]
    options += [{"label": f"{p} (not in {', '.join(missing[p])})", "value": p, "disabled": True} for p in missing]

    children = []
    if missing:
        children.append(
            dbc.Alert(
                f"{len(missing)} parameter(s) aren't in every chain and can't be compared; they're greyed out.",
                color     = "secondary",
                className = "py-2 small",
            )
        )
    children += [
        html.Div(
            [
                dbc.Label("Parameters", className = "small mb-1"),
                dbc.Checklist(id = f"{prefix}-chain-params", options = options, value = default, inline = True),
            ],
            className = "deed-controls deed-param-list",
        ),
        dcc.Loading(
            html.Div(html.Img(id = f"{prefix}-chain-img", className = "deed-chain-img"), className = "text-center"),
            type = "dot",
        ),
    ]
    if with_summary:
        children.append(html.Div(id = f"{prefix}-chain-summary", className = "mt-3"))
    return children


### ------------------------------------------------------------- callbacks

def register_view_callbacks(prefix, select, extra_inputs = ()):
    """ register_view_callbacks

        Inputs:
        - prefix (str): id prefix of the page's views
        - select (callable): select(datasets, *extra) -> the dataset entries the
        page's views draw (all of them on /compare, the chosen one on /explore)
        - extra_inputs (tuple of Input): page-specific inputs passed to select

        Process:
        - Registers the figure callbacks for one page's sacc and chain views. Each
        also listens to the navbar's theme-switch, so toggling dark mode redraws
        the figures in the matching template.

        Outputs:
        - None
    """
    extra_inputs = list(extra_inputs)
    theme_input  = Input("theme-switch", "value")
    store_input  = Input("dataset-store", "data")

    @dash.callback(
        Output(f"{prefix}-dv-graph", "figure"),
        Input(f"{prefix}-data-type", "value"),
        theme_input, store_input, *extra_inputs,
    )
    def update_data_vector(data_type, dark, datasets, *extra):
        theme = theme_of(dark)
        saccs = datasets_of_kind(select(datasets, *extra), "sacc")
        if not saccs or not data_type:
            return apply_theme(go.Figure(), theme)
        frames = [load_dataset_bundle(d["path"])[1] for d in saccs]
        return apply_theme(make_data_vector_figure(saccs, frames, data_type), theme)

    @dash.callback(
        Output(f"{prefix}-nz-graph", "figure"),
        Input(f"{prefix}-tracers", "value"),
        theme_input, store_input, *extra_inputs,
    )
    def update_nz(selected_tracers, dark, datasets, *extra):
        theme = theme_of(dark)
        saccs = datasets_of_kind(select(datasets, *extra), "sacc")
        if not saccs:
            return apply_theme(go.Figure(), theme)
        nz_frames = [load_dataset_bundle(d["path"])[2] for d in saccs]
        return apply_theme(make_nz_figure(saccs, nz_frames, selected_tracers or [], theme = theme), theme)

    @dash.callback(
        Output(f"{prefix}-cov-graph", "figure"),
        Input(f"{prefix}-cov-mode", "value"),
        Input(f"{prefix}-cov-dataset", "value"),
        theme_input, store_input, *extra_inputs,
    )
    def update_cov(mode, cov_path, dark, datasets, *extra):
        theme = theme_of(dark)
        saccs = datasets_of_kind(select(datasets, *extra), "sacc")
        if not saccs:
            return apply_theme(go.Figure(), theme)

        ### A removal can leave the selector holding a path that is no longer
        ### loaded; fall back to the first file rather than read a stale one
        entry = next((d for d in saccs if d["path"] == cov_path), saccs[0])
        s, df, _, _, _, has_cov = load_dataset_bundle(entry["path"])
        if not has_cov:
            return apply_theme(message_figure(f"No covariance matrix in {entry['label']}."), theme)
        label = entry["label"] if len(saccs) > 1 else None
        return apply_theme(make_covariance_figure(s, df, mode, label = label), theme)

    @dash.callback(
        Output(f"{prefix}-chain-img", "src"),
        Input(f"{prefix}-chain-params", "value"),
        theme_input, store_input, *extra_inputs,
    )
    def update_chain(selected_params, dark, datasets, *extra):
        chains = datasets_of_kind(select(datasets, *extra), "chain")
        if not chains or not selected_params:
            return ""
        ### Tuples, because make_chain_figure is cached
        return make_chain_figure(
            paths      = tuple(d["path"] for d in chains),
            params     = tuple(selected_params),
            labels     = tuple(d["label"] for d in chains),
            colors     = tuple(d["color"] for d in chains),
            linestyles = tuple(d["linestyle"] for d in chains),
            theme      = theme_of(dark),
        )
