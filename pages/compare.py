"""/compare -- every loaded dataset overplotted.

The plot_many.py views in the new layout: all loaded chains in one getdist
figure, all loaded sacc files in shared data-vector and n(z) views, and the
covariance of one chosen file. The deferred comparison modes (ratio / pull to a
reference, covariance difference) belong on this page too.
"""
import dash
from dash import html
from dash.dependencies import Input, Output

from deed_core import datasets_of_kind, load_dataset_bundle
from deed_ui import card, empty_state, make_chain_view, make_sacc_tabs, register_view_callbacks, swatch

### The home page: comparing what is loaded is what DEED has always opened on.
### /compare redirects here, so the obvious URL works too
dash.register_page(__name__, path = "/", name = "Compare", order = 1, redirect_from = ["/compare"], title = "DEED -- Compare")

layout = html.Div(
    [
        html.Div(
            [
                html.H4("Compare", className = "mb-1"),
                html.P(
                    "Every loaded dataset overplotted: chains in one triangle plot, sacc files in shared "
                    "data-vector and n(z) views. Colors and symbols match the sidebar.",
                    className = "text-body-secondary mb-0",
                ),
            ],
            className = "deed-page-header",
        ),
        html.Div(id = "cmp-content"),
    ]
)


def make_sacc_summary(saccs):
    """ make_sacc_summary

        Inputs:
        - saccs (list of dict): sacc dataset entries

        Outputs:
        - one line per file: swatch, label, data points, tracers, covariance (html.Div)
    """
    rows = []
    for entry in saccs:
        _, df, _, _, tracers, has_cov = load_dataset_bundle(entry["path"])
        rows.append(
            html.Div(
                [
                    swatch(entry),
                    html.Span(entry["label"], className = "fw-semibold me-2"),
                    html.Span(
                        f"{len(df)} data points · {len(tracers)} tracers · covariance {'yes' if has_cov else 'no'}",
                        className = "small text-body-secondary",
                    ),
                ],
                className = "d-flex align-items-center gap-2 mb-1",
            )
        )
    return html.Div(rows, className = "mb-3")


@dash.callback(Output("cmp-content", "children"), Input("dataset-store", "data"))
def update_compare_content(datasets):
    chains = datasets_of_kind(datasets, "chain")
    saccs  = datasets_of_kind(datasets, "sacc")

    if not chains and not saccs:
        return empty_state("Add a dataset in the sidebar -- a local .sacc / chain .txt file, or a Data Registry dataset_id.")

    children = []
    if chains:
        children.append(
            card(
                f"MCMC chains ({len(chains)})",
                make_chain_view(chains, "cmp"),
                subtitle = "Only parameters every chain has can be overplotted.",
            )
        )
    if saccs:
        children.append(
            card(
                f"sacc files ({len(saccs)})",
                [make_sacc_summary(saccs), make_sacc_tabs(saccs, "cmp")],
                subtitle = "Filled markers are positive values, open markers |negative|.",
            )
        )
    return children


register_view_callbacks("cmp", select = lambda datasets: datasets or [])
