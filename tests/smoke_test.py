"""Smoke tests for the DEED front ends, driven through Flask's test client.

No server, port or browser: each check builds the app in-process and POSTs to
Dash's callback endpoint the way the browser would (see deed_notes/
troubleshooting.md, "Driving Dash callbacks without a server or a browser").

Usage (from the repo root, venv active):
    python tests/smoke_test.py
"""
import json
import os
import sys
import time

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

SACC_PATH = os.path.join(REPO_DIR, "paul_auto_forecast_fid_3x2pt_linear_sys_limber_20_log_bins.sacc")
CHAIN_DIR = os.path.expanduser("~/DEED/example_analysis_files")
CHAIN_PATHS = [os.path.join(CHAIN_DIR, f) for f in ("chain_2x2pt_lcdm_SR_maglim.txt", "chain_3x2pt_lcdm_SR_maglim.txt")]
CHAIN_PARAMS = ["cosmological_parameters--omega_m", "cosmological_parameters--sigma_8"]

FAILURES = []


def check(name, ok, detail = ""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


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

    datasets = deed_core.normalize_datasets([SACC_PATH] + CHAIN_PATHS)
    client = plot_many.build_app(datasets).server.test_client()

    status, ids = layout_ids(client)
    check("layout served", status == 200)
    check("layout has sacc + chain ids", {"data-type-dropdown", "tracer-checklist", "chain-param-checklist", "dataset-store"} <= ids, str(sorted(ids))[:200])

    status, resp = post_callback(client, [("data-vector-graph", "figure")],
                                 [("data-type-dropdown", "value", "galaxy_shear_cl_ee"), ("dataset-store", "data", datasets)])
    check("data vector", status == 200 and len(resp["response"]["data-vector-graph"]["figure"]["data"]) > 0)

    status, resp = post_callback(client, [("cov-graph", "figure")],
                                 [("cov-mode", "value", "correlation"), ("cov-dataset", "value", SACC_PATH), ("dataset-store", "data", datasets)])
    check("covariance", status == 200 and resp["response"]["cov-graph"]["figure"]["data"][0]["type"] == "heatmap")

    t0 = time.time()
    status, resp = post_callback(client, [("chain-graph", "src")],
                                 [("chain-param-checklist", "value", CHAIN_PARAMS), ("dataset-store", "data", datasets)])
    check("chain plot", status == 200 and resp["response"]["chain-graph"]["src"].startswith("data:image/png;base64,"), f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    tests = [name for name in sys.argv[1:]] or [n for n in dir() if n.startswith("test_")]
    for name in tests:
        globals()[name]()
    print(f"\n{len(FAILURES)} failure(s){': ' + ', '.join(FAILURES) if FAILURES else ''}")
    sys.exit(1 if FAILURES else 0)
