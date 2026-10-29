# DESC Dark Energy Exploration Dashboard (DEED)

This is the visualization component of the DESC Dark Energy Exploration Dashboard (DEED). In its current state, `plot.py` takes a `.sacc` file that stores the results of a 3x2pt analysis. This is a first draft whose features will keep evolving with time.

See `requirements.txt` for the required packages.

## Running it

The dashboard is `app.py`, a multi-page Dash app. A sidebar holds the loaded
datasets (`.sacc` files and `.txt` MCMC chains, added by local path or by DESC
Data Registry `dataset_id`), and three pages work on them:

| Page | What it does |
| --- | --- |
| **Compare** (`/`) | Overplots every loaded dataset: chains in one getdist triangle plot, sacc files in shared data-vector and n(z) views, and the covariance of a chosen file |
| **Explore** (`/explore`) | One dataset in depth: what the file is, a contents table or parameter summary, and its plots |
| **Registry** (`/registry`) | Searches the Data Registry's working schema and adds datasets from it |

A switch in the navbar toggles light and dark mode.

```bash
python app.py --file path/to/a.sacc --file path/to/chain.txt [--registry-id N] [--port 8050] [--debug]
gunicorn app:server    # production server -- planned for Spin; gunicorn isn't in requirements.txt yet
```

`plot.py` (one dataset) and `plot_many.py` (overlays, single page) are the
earlier front ends and still run. `app.py` and `plot_many.py` share
`deed_core.py` for loading and figures; `plot.py` is self-contained. `tests/smoke_test.py` exercises the callbacks without a browser,
and `tests/screenshots.py` captures every page with a headless browser.

Claude built this code with human input.
