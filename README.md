# DESC Dark Energy Exploration Dashboard (DEED)

DEED is a browser dashboard for exploring DESC 3x2pt analysis products:
`.sacc` files (data vector, n(z), covariance) and `.txt` MCMC chains (CosmoSIS
format, plotted with getdist). Files are loaded from a local path or, at NERSC,
by `dataset_id` from the DESC Data Registry. It is a first draft whose features
will keep evolving.

## Installation

Python 3.12 or newer (tested on 3.13).

```bash
git clone https://github.com/LSSTDESC/DEED.git
cd DEED
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -r requirements-dataregistry.txt   # optional: Data Registry support
```

The second install is only needed for Data Registry lookups. It must use
`--no-deps`; `requirements-dataregistry.txt` explains why.

**On Perlmutter**, run `module load python` before creating the venv, since the
system `python3` is too old. Each later session then needs only
`source .venv/bin/activate`.

## Running the app

```bash
python app.py
```

Then open <http://localhost:8050>. Datasets can be added from the sidebar once
the app is running, or given at startup:

```bash
python app.py --file path/to/a.sacc --file path/to/chain.txt --registry-id 46
```

| Option | Meaning |
| --- | --- |
| `--file PATH` | A local `.sacc` or `.txt` chain file (repeatable) |
| `--registry-id N` | A dataset in the Data Registry's working schema (repeatable) |
| `--registry-config PATH` | A dataregistry config file, if not the default (see below) |
| `--port N` | Port to serve on (default 8050) |
| `--host ADDR` | Interface to listen on (default `127.0.0.1`, this machine only) |
| `--debug` | Dash debug mode: error pop-ups and the dev-tools toolbar |

Stop the app with Ctrl+C. Relative `--file` paths are resolved against the
directory you run from; the app itself can be run from any directory.

The repo includes one example file to try:
`paul_auto_forecast_fid_3x2pt_linear_sys_limber_20_log_bins.sacc`.

### On Perlmutter

The app serves on the node it runs on, so forward the port to your own machine
with SSH. Open the tunnel when you log in, then start the app in that session:

```bash
ssh -L 8050:localhost:8050 <username>@perlmutter.nersc.gov
cd DEED && source .venv/bin/activate
python app.py --port 8050
```

Then open <http://localhost:8050> on your own machine. Login nodes are shared:
if port 8050 is taken, pick another and use it in both the `ssh -L` and
`--port`. VS Code Remote-SSH forwards the port for you.

For heavy chains, run the app on a compute node instead (`salloc`). There,
start it with `--host $(hostname)` and tunnel to that node's name, e.g.
`ssh -L 8050:nid001234:8050 <username>@perlmutter.nersc.gov`. Note that a
non-default `--host` makes the app reachable from inside NERSC by anyone who
knows the address, and "Add file" can open any file your account can read.

## Using the dashboard

The sidebar lists the loaded datasets and adds more, either by path or by Data
Registry `dataset_id`; × removes one. Three pages work on the loaded datasets:

| Page | What it does |
| --- | --- |
| **Compare** (`/`) | Overplots every loaded dataset. Sacc files share data-vector and n(z) views and show a chosen file's covariance (or correlation, or precision). Chains share one triangle plot. |
| **Explore** (`/explore`) | One dataset in depth: what the file is, a table of its contents or a parameter summary, and its plots. |
| **Registry** (`/registry`) | Searches the Data Registry's working schema by name and owner and adds results to the sidebar. |

The chain triangle plot has two modes:

- **Interactive** (Plotly). Hover a curve or contour for its statistics: mean
  ± std and intervals, and for a parameter pair the correlation and figure of
  merit (FoM), with the ratio to a chosen reference chain. Click a panel to pin
  its statistics below the plot. A table lists every selected pair's FoM per
  chain. More than 8 parameters gets slow and shows a warning.
- **Static**: getdist's own PNG.

The FoM is currently defined as 1 / area of the pair's 68% ellipse.

A switch in the navbar toggles light and dark mode.

## Data Registry access

Registry lookups (`--registry-id`, the sidebar's ID box, the Registry page)
need a DESC account at NERSC. dataregistry looks for its config in this order:

1. the `DATAREG_CONFIG` environment variable;
2. NERSC's shared config under `/global/common/software/lsst/dbaccess/dataregistry/`,
   readable by the `lsst` group;
3. `~/.config_reg_access`.

`--registry-config` overrides all three. Off NERSC, the registry features
report an error, but everything else works with local files.

## Other front ends and tests

- `plot.py` (one dataset) and `plot_many.py` (overlays, single page) are the
  earlier front ends and still run, with the same `--file`, `--port` and
  `--host` options. `app.py` and `plot_many.py` share `deed_core.py`;
  `plot.py` is self-contained.
- `describe_file.py` classifies ceci / TXPipe / RAIL output files; the Explore
  page uses it.
- `Muir_plottingFiles/` is Jessie Muir's getdist plotting utility, included
  as received.
- `tests/smoke_test.py` exercises every page's callbacks without a browser:

  ```bash
  python tests/smoke_test.py
  DEED_TEST_CHAIN_DIR=/path/to/chains python tests/smoke_test.py
  ```

  The chain checks need `chain_2x2pt_lcdm_SR_maglim.txt` and
  `chain_3x2pt_lcdm_SR_maglim.txt` (DES Y3) in `$DEED_TEST_CHAIN_DIR`. The
  registry checks need Data Registry access. Checks that can't run are reported
  as skipped, not failed.
- `tests/screenshots.py` captures every page with a headless browser. It needs
  Playwright, set up as described in its docstring.
- A production server (`gunicorn app:server`) is planned for NERSC Spin.
  gunicorn isn't in `requirements.txt` yet.

Claude built this code with human input.
