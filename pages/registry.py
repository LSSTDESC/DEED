"""/registry -- search the DESC Data Registry and add datasets from it.

Searches the working schema (the same one --registry-id and the sidebar's
dataset_id box resolve against) by name, owner and file type. Each result that
DEED can load gets an "Add" button, which goes through the app's single
dataset-store callback (registry-add pattern ids, see app.py).
"""
import os

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, html
from dash.dependencies import Input, Output, State

from deed_core import REGISTRY_MAX_ROWS, search_registry
from deed_ui import card, empty_state

dash.register_page(__name__, path = "/registry", name = "Registry", order = 2, title = "DEED -- Registry")

LOADABLE_EXTENSIONS = {".sacc": "sacc", ".txt": "chain"}

layout = html.Div(
    [
        html.Div(
            [
                html.H4("Data Registry", className = "mb-1"),
                html.P(
                    "Search the DESC Data Registry's working schema and add datasets to the sidebar.",
                    className = "text-body-secondary mb-0",
                ),
            ],
            className = "deed-page-header",
        ),
        card(
            "Search",
            [
                dbc.Row(
                    [
                        dbc.Col([dbc.Label("Name contains", className = "small mb-1"),
                                 dbc.Input(id = "reg-name", type = "text", size = "sm", placeholder = "e.g. chain_3x2pt")], md = 4),
                        dbc.Col([dbc.Label("Owner contains", className = "small mb-1"),
                                 dbc.Input(id = "reg-owner", type = "text", size = "sm", placeholder = "e.g. sajkov")], md = 3),
                        dbc.Col([dbc.Label("File type", className = "small mb-1"),
                                 dbc.Select(id = "reg-type", size = "sm", value = "loadable",
                                            options = [{"label": "Files DEED can load", "value": "loadable"},
                                                       {"label": ".sacc only", "value": ".sacc"},
                                                       {"label": ".txt (chains) only", "value": ".txt"},
                                                       {"label": "Any type", "value": "any"}])], md = 3),
                        dbc.Col(dbc.Button("Search", id = "reg-search", n_clicks = 0, color = "primary", size = "sm", className = "w-100"),
                                md = 2, className = "d-flex align-items-end"),
                    ],
                    className = "g-2",
                ),
            ],
        ),
        dcc.Store(id = "reg-results-store"),
        dcc.Loading(html.Div(id = "reg-results"), type = "dot"),
    ]
)


@dash.callback(
    Output("reg-results-store", "data"),
    Input("reg-search", "n_clicks"),
    Input("reg-name", "n_submit"),
    Input("reg-owner", "n_submit"),
    State("reg-name", "value"),
    State("reg-owner", "value"),
    State("reg-type", "value"),
)
def run_search(n_clicks, name_submit, owner_submit, name, owner, extension):
    """ run_search

        Runs on page load (no filters: the newest datasets) and on every search.
        The results go to a store rather than straight into the table, so adding
        a dataset can re-render the table's buttons without re-querying.
    """
    ### The registry ANDs its filters, so "either loadable extension" is one
    ### query per extension, merged here
    extensions = list(LOADABLE_EXTENSIONS) if extension == "loadable" else [None if extension in (None, "", "any") else extension]
    try:
        frames = [search_registry(name = name, owner = owner, extension = ext) for ext in extensions]
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    df = pd.concat(frames).sort_values("dataset_id", ascending = False).head(REGISTRY_MAX_ROWS)
    df["creation_date"] = df["creation_date"].astype(str).str[:10]
    return {"rows": df.to_dict("records")}


def loadable_kind(relative_path):
    """ loadable_kind

        Inputs:
        - relative_path (str or None): the dataset's path inside the registry

        Outputs:
        - "sacc" / "chain" if DEED can load it, else None
    """
    return LOADABLE_EXTENSIONS.get(os.path.splitext(relative_path or "")[1])


@dash.callback(
    Output("reg-results", "children"),
    Input("reg-results-store", "data"),
    Input("dataset-store", "data"),
)
def render_results(results, datasets):
    if not results:
        return None
    if "error" in results:
        return dbc.Alert(["Registry search failed: ", html.Code(results["error"])], color = "danger")

    rows = results["rows"]
    if not rows:
        return empty_state("No datasets match.")

    loaded_ids = {d["origin"].split(":", 1)[1] for d in (datasets or []) if d["origin"].startswith("registry:")}
    body = []
    for r in rows:
        kind   = loadable_kind(r["relative_path"])
        loaded = str(r["dataset_id"]) in loaded_ids
        if loaded:
            action = dbc.Badge("loaded", color = "success")
        elif kind:
            action = dbc.Button("Add", id = {"type": "registry-add", "index": int(r["dataset_id"])},
                                n_clicks = 0, size = "sm", color = "primary", outline = True)
        else:
            action = html.Span("not loadable", className = "small text-body-secondary text-nowrap")
        body.append(
            html.Tr(
                [
                    html.Td(r["dataset_id"], className = "text-end"),
                    html.Td([html.Div(r["name"], className = "fw-semibold text-break"),
                             html.Div(r["description"] or "", className = "small text-body-secondary deed-clamp")]),
                    html.Td(r["version_string"]),
                    html.Td(r["owner"]),
                    html.Td(dbc.Badge(kind, color = "secondary") if kind else os.path.splitext(r["relative_path"] or "")[1] or "dir"),
                    html.Td(r["creation_date"]),
                    html.Td(action, className = "text-end"),
                ]
            )
        )
    capped = f" (newest {REGISTRY_MAX_ROWS} shown)" if len(rows) >= REGISTRY_MAX_ROWS else ""
    return card(
        f"{len(rows)} dataset{'s' if len(rows) != 1 else ''}{capped}",
        dbc.Table(
            [
                html.Thead(html.Tr([html.Th("id", className = "text-end"), html.Th("Name"), html.Th("Version"), html.Th("Owner"),
                                    html.Th("Type"), html.Th("Created"), html.Th("")])),
                html.Tbody(body),
            ],
            size       = "sm",
            hover      = True,
            responsive = True,
            className  = "mb-0 align-middle",
        ),
    )
