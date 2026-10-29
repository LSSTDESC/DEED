"""Smoke tests for the DEED front ends, driven through Flask's test client.

No server, port or browser: each check builds the app in-process and POSTs to
Dash's callback endpoint the way the browser would (see deed_notes/
troubleshooting.md, "Driving Dash callbacks without a server or a browser").

Usage (from the repo root, venv active):
    python tests/smoke_test.py
    DEED_TEST_CHAIN_DIR=/path/to/chains python tests/smoke_test.py

The .sacc checks use the example file in the repo. The chain checks need
chain_2x2pt_lcdm_SR_maglim.txt and chain_3x2pt_lcdm_SR_maglim.txt (DES Y3) in
$DEED_TEST_CHAIN_DIR (default ~/DEED/example_analysis_files), and the registry
checks need a readable dataregistry config (i.e. a DESC account at NERSC);
whichever is missing is reported as skipped rather than failed.
"""
import json
import os
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

SACC_PATH = os.path.join(REPO_DIR, "paul_auto_forecast_fid_3x2pt_linear_sys_limber_20_log_bins.sacc")
CHAIN_DIR = os.path.expanduser(os.environ.get("DEED_TEST_CHAIN_DIR", "~/DEED/example_analysis_files"))
CHAIN_PATHS = [os.path.join(CHAIN_DIR, f) for f in ("chain_2x2pt_lcdm_SR_maglim.txt", "chain_3x2pt_lcdm_SR_maglim.txt")]
CHAIN_PARAMS = ["cosmological_parameters--omega_m", "cosmological_parameters--sigma_8"]
HAVE_CHAINS = all(os.path.isfile(p) for p in CHAIN_PATHS)

### Where dataregistry looks for its config when none is given (dataregistry 1.7.0, db_basic.py)
REGISTRY_CONFIGS = [os.environ.get("DATAREG_CONFIG", ""),
                    "/global/common/software/lsst/dbaccess/dataregistry/data/.writer_config",
                    os.path.expanduser("~/.config_reg_access")]
HAVE_REGISTRY = any(p and os.access(p, os.R_OK) for p in REGISTRY_CONFIGS)

FAILURES = []
SKIPS    = []


def check(name, ok, detail = ""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def skip(name, reason):
    print(f"  [skip] {name} -- {reason}")
    SKIPS.append(name)


NO_CHAINS   = f"chain files not found in {CHAIN_DIR} (set DEED_TEST_CHAIN_DIR)"
NO_REGISTRY = "no readable dataregistry config (needs DESC access at NERSC)"


def post_callback(client, outputs, inputs, state = (), changed = None):
    """ post_callback

        Inputs:
        - client: Flask test client
        - outputs (list of (id, property)): the callback's outputs
        - inputs (list of (id, property, value)): the callback's inputs; a value of
        ("ALL", [ {id, property, value}, ... ]) stands for a pattern-matching input
        - state (list of (id, property, value))
        - changed (list of str or None): changedPropIds, defaulting to the first input

        Outputs:
        - (status code, parsed JSON response or None)
    """
    def encode(item):
        cid, prop, value = item
        if isinstance(value, tuple) and value and value[0] == "ALL":
            return value[1]
        return {"id": cid, "property": prop, "value": value}

    def key(cid):
        return json.dumps(cid, separators = (",", ":"), sort_keys = True) if isinstance(cid, dict) else cid

    output = (
        f"{key(outputs[0][0])}.{outputs[0][1]}" if len(outputs) == 1
        else ".." + "...".join(f"{key(c)}.{p}" for c, p in outputs) + ".."
    )
    body = {
        "output":         output,
        "outputs":        [{"id": c, "property": p} for c, p in outputs] if len(outputs) > 1 else {"id": outputs[0][0], "property": outputs[0][1]},
        "inputs":         [encode(i) for i in inputs],
        "state":          [encode(s) for s in state],
        "changedPropIds": changed or [f"{key(inputs[0][0])}.{inputs[0][1]}"],
    }
    r = client.post("/_dash-update-component", json = body)
    return r.status_code, (r.get_json() if r.status_code == 200 else None)


def layout_ids(client, path = "/_dash-layout"):
    """All component ids in a served layout (json walk)."""
    found = set()

    def walk(node):
        if isinstance(node, dict):
            if "props" in node and isinstance(node["props"], dict) and "id" in node["props"]:
                found.add(json.dumps(node["props"]["id"], sort_keys = True) if isinstance(node["props"]["id"], dict) else node["props"]["id"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    r = client.get(path)
    walk(r.get_json())
    return r.status_code, found


def test_plot_many():
    print("plot_many.py")
    import deed_core
    import plot_many

    datasets = deed_core.normalize_datasets([SACC_PATH] + (CHAIN_PATHS if HAVE_CHAINS else []))
    client = plot_many.build_app(datasets).server.test_client()

    status, ids = layout_ids(client)
    check("layout served", status == 200)
    want = {"data-type-dropdown", "tracer-checklist", "dataset-store"} | ({"chain-param-checklist"} if HAVE_CHAINS else set())
    check("layout has sacc + chain ids" if HAVE_CHAINS else "layout has sacc ids", want <= ids, str(sorted(ids))[:200])

    status, resp = post_callback(client, [("data-vector-graph", "figure")],
                                 [("data-type-dropdown", "value", "galaxy_shear_cl_ee"), ("dataset-store", "data", datasets)])
    check("data vector", status == 200 and len(resp["response"]["data-vector-graph"]["figure"]["data"]) > 0)

    status, resp = post_callback(client, [("cov-graph", "figure")],
                                 [("cov-mode", "value", "correlation"), ("cov-dataset", "value", SACC_PATH), ("dataset-store", "data", datasets)])
    check("covariance", status == 200 and resp["response"]["cov-graph"]["figure"]["data"][0]["type"] == "heatmap")

    if not HAVE_CHAINS:
        skip("chain plot", NO_CHAINS)
        return

    t0 = time.time()
    status, resp = post_callback(client, [("chain-graph", "src")],
                                 [("chain-param-checklist", "value", CHAIN_PARAMS), ("dataset-store", "data", datasets)])
    check("chain plot", status == 200 and resp["response"]["chain-graph"]["src"].startswith("data:image/png;base64,"), f"{time.time() - t0:.1f} s")


def find_ids(node, found = None):
    """All component ids inside a callback's returned children."""
    found = set() if found is None else found
    if isinstance(node, dict):
        props = node.get("props")
        if isinstance(props, dict) and "id" in props:
            found.add(json.dumps(props["id"], sort_keys = True) if isinstance(props["id"], dict) else props["id"])
        for v in node.values():
            find_ids(v, found)
    elif isinstance(node, list):
        for v in node:
            find_ids(v, found)
    return found


def png_alpha_fraction(uri):
    """Fraction of fully transparent pixels in a base64 PNG data URI."""
    import base64
    import io
    import numpy as np
    from PIL import Image
    pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGBA"))
    return float((pixels[..., 3] == 0).mean())


def test_app():
    print("app.py")
    import deed_core
    import app as deed_app

    client   = deed_app.server.test_client()
    datasets = deed_core.normalize_datasets([SACC_PATH] + (CHAIN_PATHS if HAVE_CHAINS else []))
    store    = ("dataset-store", "data", datasets)

    for path in ("/", "/explore", "/registry", "/compare"):
        check(f"GET {path}", client.get(path).status_code in (200, 301))
    status, ids = layout_ids(client)
    check("shell ids", {"dataset-store", "theme-switch", "file-path-input", "registry-id-input", "dataset-panel", "_pages_content"} <= ids,
          str(sorted(ids))[:200])

    ### Sidebar
    status, resp = post_callback(client, [("dataset-panel", "children"), ("dataset-count", "children")], [store])
    check("sidebar lists datasets", status == 200 and resp["response"]["dataset-count"]["children"] == str(len(datasets)))

    ### /compare
    status, resp = post_callback(client, [("cmp-content", "children")], [store])
    cmp_ids = find_ids(resp["response"]["cmp-content"]["children"]) if status == 200 else set()
    want    = {"cmp-data-type", "cmp-dv-graph", "cmp-tracers", "cmp-nz-graph", "cmp-cov-mode", "cmp-cov-graph"}
    check("compare content", (want | ({"cmp-chain-params", "cmp-chain-img"} if HAVE_CHAINS else set())) <= cmp_ids, str(sorted(cmp_ids))[:200])

    for dark in (False, True):
        theme = "dark" if dark else "light"
        status, resp = post_callback(client, [("cmp-dv-graph", "figure")],
                                     [("cmp-data-type", "value", "galaxy_shear_cl_ee"), ("theme-switch", "value", dark), store])
        fig = resp["response"]["cmp-dv-graph"]["figure"] if status == 200 else {}
        template_bg = fig.get("layout", {}).get("template", {}).get("layout", {}).get("paper_bgcolor")
        check(f"compare data vector ({theme})", status == 200 and len(fig["data"]) > 0 and template_bg == "rgba(0,0,0,0)")

    status, resp = post_callback(client, [("cmp-nz-graph", "figure")],
                                 [("cmp-tracers", "value", ["src0", "lens0"]), ("theme-switch", "value", True), store])
    check("compare n(z) (dark)", status == 200 and len(resp["response"]["cmp-nz-graph"]["figure"]["data"]) > 0)

    status, resp = post_callback(client, [("cmp-cov-graph", "figure")],
                                 [("cmp-cov-mode", "value", "precision"), ("cmp-cov-dataset", "value", SACC_PATH),
                                  ("theme-switch", "value", True), store])
    check("compare covariance (dark)", status == 200 and "data" in resp["response"]["cmp-cov-graph"]["figure"])

    if not HAVE_CHAINS:
        skip("compare chains, interactive chains, FoM", NO_CHAINS)
    else:
        for dark in (False, True):
            t0 = time.time()
            status, resp = post_callback(client, [("cmp-chain-img", "src")],
                                         [("cmp-chain-params", "value", CHAIN_PARAMS), ("cmp-chain-mode", "value", "static"),
                                          ("theme-switch", "value", dark), store])
            uri = resp["response"]["cmp-chain-img"]["src"] if status == 200 else ""
            alpha = png_alpha_fraction(uri) if uri.startswith("data:image/png") else -1
            ok = alpha > 0.5 if dark else alpha == 0.0
            check(f"compare chains ({'dark' if dark else 'light'})", status == 200 and ok, f"transparent {alpha:.0%}, {time.time() - t0:.1f} s")

        ### Interactive chains: figure + trace map, pin by click, FoM table
        chains   = [d for d in datasets if d["kind"] == "chain"]
        PARAMS3  = CHAIN_PARAMS + ["cosmological_parameters--h0"]
        t0 = time.time()
        status, resp = post_callback(
            client, [("cmp-chain-graph", "figure"), ("cmp-chain-traces", "data"), ("cmp-chain-warning", "children")],
            [("cmp-chain-params", "value", PARAMS3), ("cmp-chain-mode", "value", "interactive"),
             ("cmp-chain-ref", "value", chains[1]["path"]), ("theme-switch", "value", True), store],
        )
        fig    = resp["response"]["cmp-chain-graph"]["figure"] if status == 200 else {}
        traces = resp["response"]["cmp-chain-traces"]["data"] if status == 200 else []
        fills  = [t for t in fig.get("data", []) if t.get("fill") == "toself"]
        check("interactive chain figure", status == 200 and len(traces) == len(fig["data"]) and len(fills) >= 2 * 2 * 3,
              f"{len(fig.get('data', []))} traces, {len(fills)} filled regions, {time.time() - t0:.1f} s")
        check("hover text has FoM ratio to reference", any("×" in (t.get("text") or "") for t in fills))

        pair_curve = next(k for k, tag in enumerate(traces) if len(tag) == 2)
        status, resp = post_callback(client, [("cmp-chain-pin", "data")],
                                     [("cmp-chain-graph", "clickData", {"points": [{"curveNumber": pair_curve}]}),
                                      ("cmp-chain-params", "value", PARAMS3), store],
                                     state = [("cmp-chain-traces", "data", traces)])
        pin = resp["response"]["cmp-chain-pin"]["data"] if status == 200 else None
        check("click pins a 2D panel", pin == traces[pair_curve] and len(pin) == 2, str(pin))

        status, resp = post_callback(client, [("cmp-chain-pinned", "children")],
                                     [("cmp-chain-pin", "data", pin), ("cmp-chain-ref", "value", chains[1]["path"]), store],
                                     state = [("cmp-chain-params", "value", PARAMS3)])
        check("pinned card shows rho and FoM", status == 200 and "FoM" in json.dumps(resp) and "reference" in json.dumps(resp))

        status, resp = post_callback(client, [("cmp-chain-pin", "data")],
                                     [("cmp-chain-graph", "clickData", {"points": [{"curveNumber": pair_curve}]}),
                                      ("cmp-chain-params", "value", CHAIN_PARAMS), store],
                                     state = [("cmp-chain-traces", "data", traces)], changed = ["cmp-chain-params.value"])
        check("changing parameters clears the pin", status == 200 and resp["response"]["cmp-chain-pin"]["data"] is None)

        status, resp = post_callback(client, [("cmp-chain-fom", "children")],
                                     [("cmp-chain-params", "value", PARAMS3), ("cmp-chain-ref", "value", chains[0]["path"]), store])
        body = json.dumps(resp)
        check("FoM table: one row per pair", status == 200 and body.count('"Tr"') == 1 + 3 and "(ref)" in body)

        import deed_core
        many = sorted(deed_core.shared_chain_params(chains)[0])[:9]
        status, resp = post_callback(
            client, [("cmp-chain-graph", "figure"), ("cmp-chain-traces", "data"), ("cmp-chain-warning", "children")],
            [("cmp-chain-params", "value", many), ("cmp-chain-mode", "value", "interactive"),
             ("cmp-chain-ref", "value", chains[0]["path"]), ("theme-switch", "value", False), store],
        )
        check("more than 8 parameters warns", status == 200 and "slow" in json.dumps(resp["response"]["cmp-chain-warning"]))

    ### /explore
    status, resp = post_callback(client, [("exp-dataset", "options"), ("exp-dataset", "value")], [store], state = [("exp-dataset", "value", None)])
    check("explore selector", status == 200 and resp["response"]["exp-dataset"]["value"] == datasets[0]["path"])

    EXPLORE = [(datasets[0], {"exp-data-type", "exp-dv-graph", "exp-cov-graph"})]
    if HAVE_CHAINS:
        EXPLORE.append((datasets[1], {"exp-chain-params", "exp-chain-img", "exp-chain-summary"}))
    for entry, want in EXPLORE:
        status, resp = post_callback(client, [("exp-content", "children")], [("exp-dataset", "value", entry["path"]), store])
        exp_ids = find_ids(resp["response"]["exp-content"]["children"]) if status == 200 else set()
        check(f"explore content ({entry['kind']})", want <= exp_ids, str(sorted(exp_ids))[:200])

    status, resp = post_callback(client, [("exp-dv-graph", "figure")],
                                 [("exp-data-type", "value", "galaxy_density_cl"), ("theme-switch", "value", False), store,
                                  ("exp-dataset", "value", datasets[0]["path"])])
    check("explore data vector", status == 200 and len(resp["response"]["exp-dv-graph"]["figure"]["data"]) > 0)

    if not HAVE_CHAINS:
        skip("explore chain content and summary", NO_CHAINS)
    else:
        status, resp = post_callback(client, [("exp-chain-summary", "children")],
                                     [("exp-chain-params", "value", CHAIN_PARAMS), ("exp-dataset", "value", datasets[1]["path"])])
        check("explore chain summary", status == 200 and "68% interval" in json.dumps(resp))

        ### The chain -> sacc switch: the old checklist fires with the new, sacc, path
        status, resp = post_callback(client, [("exp-chain-summary", "children")],
                                     [("exp-chain-params", "value", CHAIN_PARAMS), ("exp-dataset", "value", datasets[0]["path"])])
        check("explore chain summary ignores a sacc path", status in (200, 204))

    ### Sidebar add / remove, and /registry
    def update_datasets(trigger, store_data, remove = (), registry_add = (), file_path = None, dataset_id = None):
        return post_callback(
            client,
            [("dataset-store", "data"), ("registry-status", "children")],
            [("file-add-button", "n_clicks", 1 if trigger == "file-add-button.n_clicks" else 0),
             ("registry-load-button", "n_clicks", 1 if trigger == "registry-load-button.n_clicks" else 0),
             ("file-path-input", "n_submit", None), ("registry-id-input", "n_submit", None),
             (None, None, ("ALL", list(remove))), (None, None, ("ALL", list(registry_add)))],
            state   = [("file-path-input", "value", file_path), ("registry-id-input", "value", dataset_id), ("dataset-store", "data", store_data)],
            changed = [trigger],
        )

    status, resp = update_datasets("file-add-button.n_clicks", [], file_path = SACC_PATH)
    added = resp["response"]["dataset-store"]["data"] if status == 200 else []
    check("add by path", len(added) == 1 and added[0]["kind"] == "sacc")

    status, resp = update_datasets("file-add-button.n_clicks", added, file_path = "/no/such/file.sacc")
    check("add missing file -> error", status == 200 and "no such file" in json.dumps(resp["response"]["registry-status"]))

    after = added
    if not HAVE_REGISTRY:
        skip("registry search, results, add from registry", NO_REGISTRY)
    else:
        ### dataset_id 46 is chain_3x2pt_lcdm_SR_maglim v1.0.0, registered by sajkov in the working schema
        status, resp = post_callback(client, [("reg-results-store", "data")],
                                     [("reg-search", "n_clicks", 1), ("reg-name", "n_submit", None), ("reg-owner", "n_submit", None)],
                                     state = [("reg-name", "value", "chain_3x2pt_lcdm"), ("reg-owner", "value", "sajkov"), ("reg-type", "value", "loadable")])
        results = resp["response"]["reg-results-store"]["data"] if status == 200 else {}
        found = [r["dataset_id"] for r in results.get("rows", [])]
        check("registry search", 46 in found, results.get("error", str(found)))

        status, resp = post_callback(client, [("reg-results", "children")], [("reg-results-store", "data", results), ("dataset-store", "data", added)])
        check("registry results render Add buttons", status == 200 and '"registry-add"' in json.dumps(resp))

        add_button = {"id": {"type": "registry-add", "index": 46}, "property": "n_clicks", "value": 1}
        status, resp = update_datasets('{"index":46,"type":"registry-add"}.n_clicks', added, registry_add = [add_button])
        after = resp["response"].get("dataset-store", {}).get("data", added) if status == 200 else added
        check("add from registry page", [d["origin"] for d in after] == ["file", "registry:46"], json.dumps(resp)[:200] if resp else str(status))

    remove = [{"id": {"type": "dataset-remove", "index": i}, "property": "n_clicks", "value": 1 if i == 0 else 0} for i in range(len(after))]
    status, resp = update_datasets('{"index":0,"type":"dataset-remove"}.n_clicks', after, remove = remove)
    left = resp["response"]["dataset-store"]["data"] if status == 200 else []
    check("remove", [d["origin"] for d in left] == [d["origin"] for d in after[1:]])


if __name__ == "__main__":
    tests = [name for name in sys.argv[1:]] or [n for n in dir() if n.startswith("test_")]
    for name in tests:
        globals()[name]()
    print(f"\n{len(FAILURES)} failure(s){': ' + ', '.join(FAILURES) if FAILURES else ''}")
    if SKIPS:
        print(f"{len(SKIPS)} skipped: {'; '.join(SKIPS)}")
    sys.exit(1 if FAILURES else 0)
