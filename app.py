"""DESC Dark Energy Exploration Dashboard -- multi-page app.

A sidebar holds the loaded datasets (added by local path or by DESC Data
Registry dataset_id, removed with x); the pages work on them:

    /explore    one dataset in depth
    /           every loaded dataset overplotted (the plot_many.py views; /compare redirects here)
    /registry   search the Data Registry's working schema and add from it

A switch in the navbar toggles light / dark mode for the page and every figure.

Usage:
    python app.py [--file path/to/a.sacc] [--file path/to/chain.txt] [--registry-id N] [--host 127.0.0.1] [--port 8050] [--debug]

--file and --registry-id are repeatable and can be mixed. For a production
server (e.g. NERSC Spin), gunicorn serves the Flask object: gunicorn app:server
"""
import argparse
import os

import dash
import dash_bootstrap_components as dbc
from dash import clientside_callback, dcc, html
from dash.dependencies import ALL, Input, Output, State

from deed_core import (
    MAX_OVERLAY,
    add_dataset,
    check_dataset_extension,
    dataset_file_path,
    dataset_kind,
    kill_port,
    normalize_datasets,
    resolve_registry_path,
)
from deed_ui import make_dataset_list, status_message

FONT_STYLESHEET = "https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap"

### Datasets given on the command line. The layout is served by a function (see
### serve_layout), so main() can fill this in after the app object exists --
### which is what lets gunicorn import a module-level `server`
INITIAL_DATASETS = []

app = dash.Dash(
    __name__,
    use_pages                     = True,
    external_stylesheets          = [dbc.themes.BOOTSTRAP, FONT_STYLESHEET],
    title                         = "DEED -- DESC Dark Energy Exploration Dashboard",
    ### Page content, and the views inside it, come and go with navigation and
    ### with what's loaded, so Dash can't validate every callback id at startup
    suppress_callback_exceptions  = True,
)
server = app.server


def make_navbar():
    """ make_navbar

        Process:
        - Brand, one link per registered page (in page order), and the light /
        dark switch. The switch persists in the browser's local storage, so the
        choice survives a reload.

        Outputs:
        - dbc.Navbar
    """
    links = [
        dbc.NavItem(dbc.NavLink(page["name"], href = page["relative_path"], active = "exact"))
        for page in dash.page_registry.values()
    ]
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    [
                        html.Span("DEED", className = "deed-brand"),
                        html.Span("Dark Energy Exploration Dashboard", className = "deed-tagline d-none d-lg-inline"),
                    ],
                    href      = "/",
                    className = "navbar-brand d-flex align-items-baseline gap-2",
                ),
                dbc.Nav(links, navbar = True, className = "me-auto"),
                dbc.Switch(
                    id               = "theme-switch",
                    label            = "Dark mode",
                    value            = False,
                    persistence      = True,
                    persistence_type = "local",
                    className        = "mb-0 small",
                ),
            ],
            fluid = True,
        ),
        id        = "navbar",
        expand    = True,
        color     = "body-tertiary",
        ### dbc's Navbar defaults to dark = True (light text), which vanishes
        ### on the light theme; the theme callback below keeps it in step
        dark      = False,
        className = "deed-navbar border-bottom",
    )


def make_sidebar():
    """ make_sidebar

        Process:
        - "Add data" (local path or registry dataset_id) and the loaded-dataset
        list. Present on every page, so what's loaded follows you between pages.

        Outputs:
        - html.Aside
    """
    return html.Aside(
        [
            html.Div("Add data", className = "deed-sidebar-heading"),
            dbc.InputGroup(
                [
                    dbc.Input(id = "file-path-input", type = "text", placeholder = "path to .sacc or chain .txt"),
                    dbc.Button("Add", id = "file-add-button", n_clicks = 0, color = "primary"),
                ],
                size      = "sm",
                className = "mb-2",
            ),
            dbc.InputGroup(
                [
                    dbc.Input(id = "registry-id-input", type = "number", step = 1, placeholder = "Data Registry dataset_id"),
                    dbc.Button("Add", id = "registry-load-button", n_clicks = 0, color = "primary"),
                ],
                size      = "sm",
            ),
            html.Div(["or search the ", dcc.Link("Registry", href = "/registry"), "."], className = "small text-body-secondary mt-1"),
            html.Div(id = "registry-status", className = "small mt-2"),
            html.Hr(),
            html.Div(
                ["Loaded datasets ", dbc.Badge(id = "dataset-count", color = "secondary", pill = True)],
                className = "deed-sidebar-heading",
            ),
            html.Div(id = "dataset-panel"),
        ],
        className = "deed-sidebar",
    )


def serve_layout():
    """ serve_layout

        Process:
        - Called by Dash on every page load. The dataset store starts from the
        command-line datasets and is held in browser memory, so it follows the
        user between pages (navigation doesn't reload) but a reload starts over.

        Outputs:
        - the app layout (html.Div)
    """
    return html.Div(
        [
            dcc.Store(id = "dataset-store", data = normalize_datasets(INITIAL_DATASETS)),
            dcc.Store(id = "theme-applied"),
            make_navbar(),
            html.Div(
                [make_sidebar(), html.Main(dash.page_container, className = "deed-main")],
                className = "deed-shell",
            ),
        ],
        className = "deed-app",
    )


app.layout = serve_layout

### Bootstrap 5.3 switches every component's colors on the data-bs-theme
### attribute of <html>; setting it client-side avoids a server round trip.
### The navbar's text color is a prop of its own (navbar-dark), so it flips here too
clientside_callback(
    """
    function(dark) {
        document.documentElement.setAttribute("data-bs-theme", dark ? "dark" : "light");
        return [dark ? "dark" : "light", Boolean(dark)];
    }
    """,
    Output("theme-applied", "data"),
    Output("navbar", "dark"),
    Input("theme-switch", "value"),
)


@dash.callback(
    Output("dataset-panel", "children"),
    Output("dataset-count", "children"),
    Input("dataset-store", "data"),
)
def update_dataset_panel(datasets):
    return make_dataset_list(datasets or []), str(len(datasets or []))


### One callback owns every write to dataset-store: two callbacks writing the
### same Output isn't allowed, and add / remove have to see the same list. The
### Registry page's "Add" buttons feed in here too, through the registry-add
### pattern -- absent on other pages, where it's simply an empty list.
@dash.callback(
    Output("dataset-store", "data"),
    Output("registry-status", "children"),
    Input("file-add-button", "n_clicks"),
    Input("registry-load-button", "n_clicks"),
    Input("file-path-input", "n_submit"),
    Input("registry-id-input", "n_submit"),
    Input({"type": "dataset-remove", "index": ALL}, "n_clicks"),
    Input({"type": "registry-add", "index": ALL}, "n_clicks"),
    State("file-path-input", "value"),
    State("registry-id-input", "value"),
    State("dataset-store", "data"),
    prevent_initial_call = True,
)
def update_datasets(file_clicks, registry_clicks, file_submit, registry_submit,
                    remove_clicks, registry_add_clicks, file_path, dataset_id, datasets):
    """ update_datasets

        Adds a dataset by local path, by registry dataset_id (typed in the
        sidebar or clicked on the Registry page), or removes one, depending on
        which control fired. Returns the new store contents and a status line.
    """
    datasets  = list(datasets or [])
    triggered = dash.callback_context.triggered_id

    ### Rendering buttons fires this callback with their n_clicks still 0 or
    ### None, so a pattern-matched click only counts if its value is truthy
    clicked = bool(dash.callback_context.triggered and dash.callback_context.triggered[0]["value"])

    if isinstance(triggered, dict) and triggered.get("type") == "dataset-remove":
        index = triggered["index"]
        if not clicked or not (0 <= index < len(datasets)):
            return dash.no_update, dash.no_update
        dropped = datasets.pop(index)
        return normalize_datasets(datasets), status_message(f"Removed {dropped['label']}.")

    if isinstance(triggered, dict) and triggered.get("type") == "registry-add":
        if not clicked:
            return dash.no_update, dash.no_update
        return add_from_registry(datasets, triggered["index"])

    if triggered in ("file-add-button", "file-path-input"):
        if not file_path:
            return dash.no_update, status_message("Enter a file path to add.")
        path = os.path.abspath(os.path.expanduser(file_path.strip()))
        try:
            check_dataset_extension(path)
        except ValueError as e:
            return dash.no_update, status_message(f"Error: {e}")
        if not os.path.isfile(path):
            return dash.no_update, status_message(f"Error: no such file: {path}")
        new, message = add_dataset(datasets, path, origin = "file")
        return new, status_message(message)

    if triggered in ("registry-load-button", "registry-id-input"):
        if dataset_id is None:
            return dash.no_update, status_message("Enter a dataset_id to look up.")
        try:
            dataset_id = int(dataset_id)
        except (TypeError, ValueError):
            return dash.no_update, status_message(f"Error: dataset_id must be an integer, got {dataset_id!r}")
        return add_from_registry(datasets, dataset_id)

    return dash.no_update, dash.no_update


def add_from_registry(datasets, dataset_id):
    """ add_from_registry

        Inputs:
        - datasets (list of dict): the currently-loaded entries
        - dataset_id (int): dataset_id in the registry's working schema

        Outputs:
        - (new store contents or dash.no_update, status component)
    """
    try:
        path = resolve_registry_path(dataset_id)
    except Exception as e:
        return dash.no_update, status_message(f"Error: {e}")
    new, message = add_dataset(datasets, path, origin = f"registry:{dataset_id}")
    return new, status_message(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", action = "append", default = [], type = dataset_file_path,
                        help = "path to a local .sacc file or a .txt MCMC chain file (repeatable)")
    parser.add_argument("--registry-id", action = "append", default = [], type = int,
                        help = "dataset_id (in the working schema) to look up in the DESC Data Registry (repeatable)")
    parser.add_argument("--registry-config", default = None, help = "optional dataregistry config file (defaults to NERSC config resolution)")
    parser.add_argument("--port", type = int, default = 8050)
    parser.add_argument("--host", default = "127.0.0.1",
                        help = "interface to listen on (default 127.0.0.1, this machine only; pass it explicitly "
                               "so Dash doesn't fall back on $HOST, which NERSC sets to the node's hostname)")
    parser.add_argument("--debug", action = "store_true",
                        help = "Dash debug mode: hot error reporting and the dev-tools toolbar (which covers part of the page)")
    args = parser.parse_args()

    INITIAL_DATASETS.extend({"path": path, "origin": "file"} for path in args.file)
    for dataset_id in args.registry_id:
        try:
            path = resolve_registry_path(dataset_id, config_file = args.registry_config)
        except Exception as e:
            parser.error(f"--registry-id {dataset_id} lookup failed: {e}")
        INITIAL_DATASETS.append({"path": path, "origin": f"registry:{dataset_id}"})

    for kind, cap in MAX_OVERLAY.items():
        given = [d for d in INITIAL_DATASETS if dataset_kind(d["path"]) == kind]
        if len(given) > cap:
            parser.error(f"{len(given)} {kind} datasets given, but at most {cap} can be overplotted distinguishably")

    try:
        ### use_reloader=False keeps this to one process, so kill_port below
        ### reliably runs on any exit path (see plot_many.py for the history)
        app.run(debug = args.debug, host = args.host, port = args.port, use_reloader = False)
    finally:
        kill_port(args.port)


if __name__ == "__main__":
    main()
