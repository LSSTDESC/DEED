"""/explore -- one loaded dataset in depth.

Pick a dataset from the ones in the sidebar. Every kind gets a description of
what the file is (from describe_file.py); a sacc file then gets a table of its
contents and the Data vector / n(z) / Covariance tabs for that file alone; a
chain gets the getdist plot of any of its parameters and a numeric summary of
the selected ones.
"""
import os

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import html
from dash.dependencies import Input, Output, State

from deed_core import chain_summary, data_type_label, dataset_kind, load_dataset_bundle, natural_key
from deed_ui import card, empty_state, make_chain_view, make_sacc_tabs, origin_text, register_view_callbacks, swatch
from describe_file import describe_file

dash.register_page(__name__, path = "/explore", name = "Explore", order = 0, title = "DEED -- Explore")

layout = html.Div(
    [
        html.Div(
            [
                html.H4("Explore", className = "mb-1"),
                html.P("One dataset in depth.", className = "text-body-secondary mb-2"),
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.Label("Dataset", html_for = "exp-dataset", className = "small mb-1"),
                            dbc.Select(id = "exp-dataset", size = "sm"),
                        ],
                        md = 6,
                    ),
                ),
            ],
            className = "deed-page-header",
        ),
        html.Div(id = "exp-content"),
    ]
)


def selected_entry(datasets, path):
    """ selected_entry

        Inputs:
        - datasets (list of dict): loaded dataset entries
        - path (str or None): the value of the exp-dataset selector

        Outputs:
        - list holding the chosen entry, or an empty list
    """
    return [d for d in (datasets or []) if d["path"] == path]


@dash.callback(
    Output("exp-dataset", "options"),
    Output("exp-dataset", "value"),
    Input("dataset-store", "data"),
    State("exp-dataset", "value"),
)
def update_dataset_selector(datasets, current):
    datasets = datasets or []
    options  = [{"label": f"{d['label']} ({d['kind']})", "value": d["path"]} for d in datasets]
    paths    = [d["path"] for d in datasets]
    ### Keep the current choice while it stays loaded; otherwise take the first
    return options, current if current in paths else (paths[0] if paths else None)


def make_description_card(entry):
    """ make_description_card

        Inputs:
        - entry (dict): the dataset entry

        Outputs:
        - card with what describe_file makes of the file, and where it came from
    """
    desc = describe_file(entry["path"])
    return card(
        html.Div([swatch(entry), html.Span(entry["label"])], className = "d-flex align-items-center gap-2"),
        [
            html.P([dbc.Badge(desc.file_type, color = "primary", className = "me-2"), desc.represents], className = "mb-2"),
            html.Div(
                [html.Span("Path ", className = "text-body-secondary"), html.Code(entry["path"], className = "text-break")],
                className = "small",
            ),
            html.Div(
                [html.Span("Source ", className = "text-body-secondary"), origin_text(entry["origin"])],
                className = "small",
            ),
        ],
    )


def make_sacc_contents_table(entry):
    """ make_sacc_contents_table

        Inputs:
        - entry (dict): a sacc dataset entry

        Process:
        - One row per data type: number of data points, number of tracer pairs,
        and the range of ell or theta it covers.

        Outputs:
        - dbc.Table
    """
    _, df, nz_df, _, tracers, has_cov = load_dataset_bundle(entry["path"])
    rows = []
    for data_type, sub in df.groupby("data_type", sort = False):
        pairs = sub[["tracer1", "tracer2"]].drop_duplicates()
        x     = sub["x"].astype(float)
        scale = "θ" if "_xi" in data_type else "ℓ"
        rows.append(
            html.Tr(
                [
                    html.Td(data_type_label(data_type)),
                    html.Td(html.Code(data_type), className = "small"),
                    html.Td(len(sub), className = "text-end"),
                    html.Td(len(pairs), className = "text-end"),
                    html.Td(f"{scale} {np.nanmin(x):.4g} – {np.nanmax(x):.4g}" if x.notna().any() else "--"),
                ]
            )
        )
    table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Data type"), html.Th("sacc name"), html.Th("Points", className = "text-end"),
                                html.Th("Tracer pairs", className = "text-end"), html.Th("Range")])),
            html.Tbody(rows),
        ],
        size      = "sm",
        hover     = True,
        className = "mb-2",
    )
    footer = html.Div(
        f"{len(df)} data points in total · tracers with n(z): {', '.join(sorted(tracers, key = natural_key)) or 'none'} · "
        f"covariance {'present' if has_cov else 'absent'}",
        className = "small text-body-secondary",
    )
    return [table, footer]


@dash.callback(
    Output("exp-content", "children"),
    Input("exp-dataset", "value"),
    Input("dataset-store", "data"),
)
def update_explore_content(path, datasets):
    chosen = selected_entry(datasets, path)
    if not chosen:
        return empty_state("Add a dataset in the sidebar, then pick it above.")
    entry = chosen[0]

    children = [make_description_card(entry)]
    if entry["kind"] == "sacc":
        children += [
            card("Contents", make_sacc_contents_table(entry)),
            card("Views", make_sacc_tabs(chosen, "exp")),
        ]
    else:
        children.append(
            card(
                "Posterior",
                make_chain_view(chosen, "exp", with_summary = True),
                subtitle = f"getdist marginals of {os.path.basename(entry['path'])}: hover for statistics, click a panel to pin them.",
            )
        )
    return children


@dash.callback(
    Output("exp-chain-summary", "children"),
    Input("exp-chain-params", "value"),
    Input("exp-dataset", "value"),
)
def update_chain_summary(params, path):
    ### Switching from a chain to a sacc file fires this once more, with the
    ### old checklist still on the page and the new (sacc) path -- which would
    ### be read as chain text. Only a chain has a summary.
    if not path or not params or dataset_kind(path) != "chain":
        return None
    rows, n_eff = chain_summary(path, params)
    body = [
        html.Tr(
            [
                html.Td(html.Code(r["param"]), className = "small"),
                html.Td(f"{r['mean']:.4g}", className = "text-end"),
                html.Td(f"{r['std']:.3g}", className = "text-end"),
                html.Td(f"[{r['lo68']:.4g}, {r['hi68']:.4g}]", className = "text-end"),
                html.Td(f"[{r['lo95']:.4g}, {r['hi95']:.4g}]", className = "text-end"),
            ]
        )
        for r in rows
    ]
    return [
        dbc.Table(
            [
                html.Thead(html.Tr([html.Th("Parameter"), html.Th("Mean", className = "text-end"), html.Th("Std", className = "text-end"),
                                    html.Th("68% interval", className = "text-end"), html.Th("95% interval", className = "text-end")])),
                html.Tbody(body),
            ],
            size  = "sm",
            hover = True,
        ),
        html.Div(f"getdist marginal statistics (as in the plot); effective sample size ≈ {n_eff:,.0f}.", className = "small text-body-secondary"),
    ]


register_view_callbacks("exp", select = selected_entry, extra_inputs = [Input("exp-dataset", "value")])
