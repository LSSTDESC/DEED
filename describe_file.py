"""Identify a ceci / TXPipe / RAIL output file and say how DEED could plot it.

Given a path, works out what kind of product the file is -- from its contents
(magic bytes, HDF5 group layout, FITS header keys), not just its extension --
and returns its file type, what the data represent, and a short note on how
the dashboard could plot it, including whether DEED can already do so.

Usage:
    python describe_file.py path/to/file [path/to/another ...] [--json]

Readers are imported lazily, so a missing optional package (h5py, pyarrow, qp)
only affects the formats that need it. Pickle files are never unpickled.
"""
import argparse
import json
import os
import pickletools
from dataclasses import asdict, dataclass, field

import numpy as np

### Values for FileDescription.deed_support
SUPPORTED = "supported"           # plot.py / plot_many.py can plot it today
PLANNED = "new loader needed"     # plottable, DEED doesn't read it yet
DISPLAY_ONLY = "display only"     # already a finished picture or a table of values
NOT_PLOTTABLE = "not plottable"   # nothing to plot, or unsafe to open

### A qp ensemble with at most this many rows is treated as a tomographic n(z)
### stack (TXPipe QPNOfZFile: one row per bin plus a final 2D row) rather than a
### catalog of per-object photo-z PDFs, which has one row per galaxy
MAX_NZ_ROWS = 50

MAGIC = {
    b"\x89HDF\r\n\x1a\n": "hdf5",
    b"SIMPLE  =": "fits",
    b"\x89PNG\r\n\x1a\n": "png",
    b"PAR1": "parquet",
}


@dataclass
class FileDescription:
    path: str
    file_type: str
    represents: str
    plotting: str
    deed_support: str
    details: dict = field(default_factory = dict)


def _missing_reader(path, file_type, package):
    """ _missing_reader

        Inputs:
        - path (str): file being described
        - file_type (str): what the magic bytes say it is
        - package (str): reader package that failed to import

        Outputs:
        - FileDescription saying the contents couldn't be inspected
    """
    return FileDescription(
        path = path,
        file_type = file_type,
        represents = f"Unknown -- `{package}` isn't installed, so the contents couldn't be inspected.",
        plotting = f"Install `{package}` in this environment and re-run to classify it.",
        deed_support = PLANNED,
    )


def sniff_format(path):
    """ sniff_format

        Inputs:
        - path (str): path to a file or directory

        Process:
        - Directories are reported as such. Otherwise the first bytes are
        matched against the HDF5 / FITS / PNG / Parquet signatures, falling
        back to the extension for text-based and pickle formats

        Outputs:
        - format name (str): "directory", "hdf5", "fits", "png", "parquet",
          "pickle", "yaml", "text" or "unknown"
    """
    if os.path.isdir(path):
        return "directory"
    with open(path, "rb") as f:
        head = f.read(16)
    for magic, name in MAGIC.items():
        if head.startswith(magic):
            return name

    ext = os.path.splitext(path)[1].lower()
    if ext in (".pkl", ".pickle") or head[:1] == b"\x80":
        return "pickle"
    if ext in (".yml", ".yaml"):
        return "yaml"
    if ext in (".txt", ".dat", ".csv", ".tsv"):
        return "text"
    return "unknown"


### ---------------------------------------------------------------- FITS / sacc

def describe_sacc(path):
    """ describe_sacc

        Inputs:
        - path (str): path to a sacc file (FITS on disk)

        Process:
        - Loads it with sacc and summarizes data types, tracers and covariance.
        Also checks the two assumptions plot.py's build_data_vector_df makes --
        exactly two tracers per data point, and an "ell" or "theta" tag -- so
        the blurb can say whether DEED's existing views will cope

        Outputs:
        - FileDescription
    """
    import sacc

    s = sacc.Sacc.load_fits(path)
    data_types = sorted(s.get_data_types())
    tracer_types = {}
    for t in s.tracers.values():
        tracer_types[type(t).__name__] = tracer_types.get(type(t).__name__, 0) + 1

    n_points = len(s.data)
    bad_tracers = sum(1 for dp in s.data if len(dp.tracers) != 2)
    no_x = sum(1 for dp in s.data if "ell" not in dp.tags and "theta" not in dp.tags)
    has_cov = s.covariance is not None
    n_nz = sum(1 for t in s.tracers.values() if hasattr(t, "z"))

    spaces = []
    if any("_cl" in dt for dt in data_types):
        spaces.append("Fourier-space C_ell")
    if any("_xi" in dt or "xi_" in dt for dt in data_types):
        spaces.append("real-space correlation functions")
    space_text = " and ".join(spaces) if spaces else "summary statistics"

    represents = (
        f"A sacc summary-statistics file: {n_points} data points of {space_text} "
        f"across {len(data_types)} data type(s), {len(s.tracers)} tracer(s)"
        f"{', with a covariance matrix' if has_cov else ', with no covariance'}. "
        "TXPipe writes these for its two-point data, theory predictions, null tests and delta-sigma."
    )

    if bad_tracers or no_x:
        problems = []
        if bad_tracers:
            problems.append(f"{bad_tracers} point(s) don't have exactly two tracers (build_data_vector_df raises)")
        if no_x:
            problems.append(f"{no_x} point(s) have neither an 'ell' nor a 'theta' tag (they'd have no x position)")
        plotting = (
            "Partly supported. The n(z) and covariance tabs should work, but the data-vector tab won't: "
            + "; ".join(problems)
            + ". This is typical of cluster-count, bispectrum or other non-two-point data, which needs its own view."
        )
        support = PLANNED
    else:
        plotting = (
            "Supported today: `plot.py` / `plot_many.py` show the data vector as a triangular "
            "grid of tracer pairs, the n(z) of each tracer, and the covariance as a correlation, "
            "covariance or precision heatmap. `plot_many.py` overlays several files, e.g. a TXPipe "
            "theory file (twopoint_theory_*) on its measurement."
        )
        support = SUPPORTED

    return FileDescription(
        path = path,
        file_type = "sacc (FITS)",
        represents = represents,
        plotting = plotting,
        deed_support = support,
        details = {
            "data_types": data_types,
            "tracer_types": tracer_types,
            "tracers_with_nz": n_nz,
            "n_points": n_points,
            "has_covariance": has_cov,
        },
    )


def describe_fits(path):
    """ describe_fits

        Inputs:
        - path (str): path to a FITS file

        Process:
        - Reads only the headers. Extensions carrying SACCTYPE mean sacc (handed
        to describe_sacc); 2PTDATA / COVDATA / NZDATA mean the DES "twopoint"
        format; otherwise binary tables are a catalog and images are images

        Outputs:
        - FileDescription
    """
    from astropy.io import fits

    with fits.open(path, memmap = True) as hdus:
        headers = [h.header for h in hdus]
        kinds = [type(h).__name__ for h in hdus]
        names = [h.name for h in hdus]

    if any("SACCTYPE" in h for h in headers):
        return describe_sacc(path)

    twopt = [n for n, h in zip(names, headers) if h.get("2PTDATA")]
    nz = [n for n, h in zip(names, headers) if h.get("NZDATA")]
    cov = [n for n, h in zip(names, headers) if h.get("COVDATA")]
    if twopt or nz:
        ### DES files can carry ~1000 n(z) realizations; name the main tables, count the rest
        realizations = [n for n in nz if "realisation" in n.lower() or "realization" in n.lower()]
        main_nz = [n for n in nz if n not in realizations]
        nz_text = f"n(z) tables {main_nz}" + (f" plus {len(realizations)} n(z) realizations" if realizations else "")
        return FileDescription(
            path = path,
            file_type = "twopoint FITS (DES format)",
            represents = (
                f"A DES-style two-point data file: data vectors {twopt}, {nz_text}"
                f"{', covariance ' + str(cov) if cov else ''}. It's the predecessor of sacc "
                "and holds the same three products."
            ),
            plotting = (
                "Needs a loader, but no new views: the content maps one-to-one onto the sacc tabs "
                "(data vector by bin pair, n(z), covariance). The simplest route is to convert it "
                "into a sacc.Sacc in memory and reuse plot.py's figures."
                + (" The realizations could be drawn as a spread band around each bin's n(z)." if realizations else "")
            ),
            deed_support = PLANNED,
            details = {"data_extensions": twopt, "nz_extensions": main_nz, "n_nz_realizations": len(realizations),
                       "covariance_extensions": cov},
        )

    tables = [(n, h.get("NAXIS2"), h.get("TFIELDS")) for n, h, k in zip(names, headers, kinds) if k == "BinTableHDU"]
    images = [(n, tuple(h.get(f"NAXIS{i}") for i in range(1, h.get("NAXIS", 0) + 1))) for n, h, k in zip(names, headers, kinds)
              if k in ("PrimaryHDU", "ImageHDU") and h.get("NAXIS", 0) > 0]
    if tables:
        return FileDescription(
            path = path,
            file_type = "FITS table",
            represents = (
                f"A catalog / table: {len(tables)} table extension(s), the largest with "
                f"{max(t[1] or 0 for t in tables):,} rows. RAIL's FitsHandle and TXPipe's "
                "desi_catalog_selected are written this way."
            ),
            plotting = (
                "Generic table view: pick columns for histograms, 2D density (hexbin) plots or "
                "RA/Dec sky scatter. Large catalogs need subsampling before they go to Plotly. "
                "Readable with astropy or tables_io."
            ),
            deed_support = PLANNED,
            details = {"tables": [{"name": n, "rows": r, "columns": c} for n, r, c in tables]},
        )
    if images:
        return FileDescription(
            path = path,
            file_type = "FITS image",
            represents = f"Image data: {images}.",
            plotting = "A 2D heatmap (plotly imshow); a sky image would want its WCS for axis labels.",
            deed_support = PLANNED,
            details = {"images": images},
        )
    return FileDescription(path, "FITS", "A FITS file with no data extensions.", "Nothing to plot.", NOT_PLOTTABLE)


### ---------------------------------------------------------------- HDF5

def _h5_columns(group):
    """ _h5_columns

        Inputs:
        - group (h5py.Group)

        Outputs:
        - (list of dataset names directly in group, row count of the first one or None)
    """
    import h5py

    cols = [k for k, v in group.items() if isinstance(v, h5py.Dataset)]
    rows = group[cols[0]].shape[0] if cols and group[cols[0]].shape else None
    return cols, rows


def _describe_qp(path, f):
    """ _describe_qp

        Inputs:
        - path (str): file being described
        - f (h5py.File): open file with qp's meta / data layout

        Process:
        - Reads the parameterization (pdf_name) and the number of rows. A short
        ensemble is a tomographic n(z) stack; a long one is per-object PDFs

        Outputs:
        - FileDescription
    """
    meta = f["meta"]
    pdf_name = meta["pdf_name"][0] if "pdf_name" in meta else b"unknown"
    pdf_name = pdf_name.decode() if isinstance(pdf_name, bytes) else str(pdf_name)
    data_cols, n_rows = _h5_columns(f["data"])
    grid_key = next((k for k in ("xvals", "bins") if k in meta), None)
    grid = meta[grid_key][()].squeeze() if grid_key else None
    grid_text = f", on a grid of {grid.size} points from {grid.min():.3g} to {grid.max():.3g}" if grid is not None and grid.size else ""
    ancil = list(f["ancil"].keys()) if "ancil" in f else []
    details = {"pdf_type": pdf_name, "n_pdfs": n_rows, "data_columns": data_cols, "ancil_columns": ancil}

    if n_rows is not None and n_rows <= MAX_NZ_ROWS:
        return FileDescription(
            path = path,
            file_type = "qp ensemble (HDF5) -- n(z) stack",
            represents = (
                f"Redshift distributions: {n_rows} n(z) curves of type '{pdf_name}'{grid_text}. "
                "In TXPipe's QPNOfZFile (photoz_stack, lens_photoz_stack, shear_photoz_stack) each "
                "row is one tomographic bin, and the last row is the 2D (non-tomographic) n(z)."
            ),
            plotting = (
                "Fits DEED's existing n(z) view: one curve per bin, colored like the sacc n(z) tab, "
                "with the 2D row drawn separately. It can also be overlaid on the n(z) stored in a "
                "sacc file, to check that the two agree. Read with qp.read and evaluate with .pdf(z)."
            ),
            deed_support = PLANNED,
            details = details,
        )
    return FileDescription(
        path = path,
        file_type = "qp ensemble (HDF5) -- per-object PDFs",
        represents = (
            f"Photo-z PDFs for {n_rows:,} objects, of type '{pdf_name}'{grid_text}. "
            "These come from RAIL estimators (QPHandle) or TXPipe's QPPDFFile (photoz_pdfs)."
            + (f" Point estimates / extra columns: {ancil}." if ancil else "")
        ),
        plotting = (
            "Per-object view: a handful of example p(z) curves picked by index or at random. "
            "Population view: a histogram of point estimates (the ancil columns, e.g. zmode), "
            "and the stacked sum of the PDFs as an n(z). Evaluate in chunks (qp.iterator); "
            "don't load a large file whole."
        ),
        deed_support = PLANNED,
        details = details,
    )


def _describe_maps(path, f):
    """ _describe_maps

        Inputs:
        - path (str): file being described
        - f (h5py.File): open file with a top-level "maps" group

        Process:
        - Finds every map the way TXPipe's MapsFile.list_maps does: a group
        holding "pixel" + "value" datasets (legacy healpix) or a "healsparse" subgroup

        Outputs:
        - FileDescription
    """
    import h5py

    found = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group):
            keys = obj.keys()
            if ("pixel" in keys and "value" in keys) or "healsparse" in keys:
                found.append((name, "healsparse" if "healsparse" in keys else "healpix"))

    f["maps"].visititems(visit)
    attrs = {k: (v.item() if hasattr(v, "item") else v) for k, v in f["maps"].attrs.items()}
    nside = attrs.get("nside")
    noise = "realization" in " ".join(n for n, _ in found) or any("realization" in k for k in attrs)

    return FileDescription(
        path = path,
        file_type = "TXPipe maps file (HDF5)",
        represents = (
            f"{'Noise-realization sky' if noise else 'Sky'} maps: {len(found)} map(s)"
            f"{f' at nside {nside}' if nside else ''}, e.g. {[n for n, _ in found[:6]]}. "
            "TXPipe's MapsFile covers depth, masks, PSF, flags, bright objects, source/lens "
            "density and shear, convergence, and LSS weights."
        ),
        plotting = (
            "Sky map: a map selector plus a Mollweide or zoomed Cartesian projection of the "
            "footprint. Plotly has no HEALPix projection, so the simplest route matches the chain "
            "plots: render with healpy (healsparse maps degraded to healpix) into a PNG for an "
            "<img>. Alternatively, draw pixel centers as a Plotly scattergeo, which keeps hover. "
            "Also useful: a histogram of pixel values, and the mask's sky fraction."
        ),
        deed_support = PLANNED,
        details = {"maps": [n for n, _ in found], "map_formats": sorted({k for _, k in found}), "attrs": attrs},
    )


### HDF5 top-level group -> (what it is, how to plot it), for TXPipe catalogs and
### diagnostics. Group names come from the create_group calls in TXPipe's source.
CATALOG_GROUPS = {
    "shear": (
        "a shear catalog (TXPipe ShearCatalog: metacal / metadetect / lensfit / HSC / simple layouts)",
        "Histograms of g1, g2, S/N, size T and PSF quantities; mean shear in bins of PSF ellipticity "
        "(a null test); RA/Dec density. Subsample first: these catalogs run to millions of rows.",
    ),
    "photometry": (
        "a photometry catalog (TXPipe PhotometryCatalog), mag_<band> columns per band",
        "Magnitude histograms per band, color-color and color-magnitude 2D density, and redshift "
        "vs magnitude if truth redshifts are present. Subsample first.",
    ),
    "tomography": (
        "tomographic bin assignments (TXPipe TomographyCatalog): a bin index per object, with the bin edges as attributes",
        "Bar chart of counts per bin with the z edges marked; joined with a photo-z or truth "
        "catalog, this gives each bin's redshift distribution.",
    ),
    "randoms": (
        "a randoms catalog (TXPipe RandomsCatalog): positions sampling the survey footprint",
        "RA/Dec density map of the footprint (and z histogram, if there's a z column). Subsample first.",
    ),
    "lens": (
        "a lens-sample catalog, possibly split into tomographic bins",
        "Counts per bin, RA/Dec density, and weight / magnitude histograms.",
    ),
    "stars": (
        "a star catalog (PSF stars)",
        "Star density on the sky, and histograms of PSF size and ellipticity residuals.",
    ),
    "clusters": (
        "a cluster catalog (e.g. redMaPPer)",
        "Richness vs redshift scatter, and counts in the richness x redshift bins.",
    ),
}

DIAGNOSTIC_GROUPS = {
    "rowe_statistics": "Rowe statistics rho_0..rho_5 (PSF-residual correlation functions) vs theta",
    "tau_statistics": "tau statistics tau_0, tau_2, tau_5 (PSF-shear cross-correlations) vs theta",
    "star_cross_galaxy": "galaxy-star shear cross-correlation vs theta (a systematics null test)",
    "star_cross_star": "star-star shear correlation vs theta",
    "star_density": "galaxy-star density correlation vs theta",
    "moment_statistics": "PSF moment correlations",
    "brighter_fatter": "PSF size / ellipticity vs magnitude (brighter-fatter effect)",
    "quantiles": "quantiles of shear-catalog columns, used to bin diagnostics",
    "exposures": "per-exposure metadata (pointing, seeing, airmass ...)",
    "magnification": "magnification coefficients from SSI",
    "lognormal_cl": "input C_ell for the lognormal (GLASS) simulation",
    "tracers": "per-tracer metadata (number densities, shape noise, areas)",
}


def describe_hdf5(path):
    """ describe_hdf5

        Inputs:
        - path (str): path to an HDF5 file

        Process:
        - Identifies the layout from the top-level groups, in priority order:
        qp ensemble (meta + data, or qps for several), TXPipe maps, TXPipe
        catalogs, TXPipe diagnostics, then a generic table / tree fallback.
        Notes ceci provenance when present

        Outputs:
        - FileDescription
    """
    try:
        import h5py
    except ImportError:
        return _missing_reader(path, "HDF5", "h5py")

    with h5py.File(path, "r") as f:
        top = set(f.keys())
        provenance = {k: str(v) for k, v in f["provenance"].attrs.items()} if "provenance" in f else {}

        if {"meta", "data"} <= top:
            desc = _describe_qp(path, f)
        elif "qps" in top:
            names = list(f["qps"].keys())
            desc = FileDescription(
                path = path,
                file_type = "several qp ensembles (HDF5, TXPipe QPMultiFile)",
                represents = (
                    f"{len(names)} named qp ensembles under qps/: {names[:6]}. TXPipe writes this "
                    "for n(z) realizations (photoz_realizations), so each is one draw of the "
                    "tomographic n(z)."
                ),
                plotting = (
                    "Overlay the realizations per bin as thin, translucent curves, with the mean and "
                    "a 68% band on top -- shows how uncertain each bin's n(z) is. Mean z per bin "
                    "across realizations gives a compact scatter plot."
                ),
                deed_support = PLANNED,
                details = {"ensembles": names},
            )
        elif "maps" in top:
            desc = _describe_maps(path, f)
        elif top & CATALOG_GROUPS.keys():
            kinds = sorted(top & CATALOG_GROUPS.keys())
            parts = {}
            for k in kinds:
                cols, rows = _h5_columns(f[k])
                parts[k] = {"columns": cols[:40], "n_columns": len(cols), "rows": rows,
                            "attrs": {a: str(v) for a, v in f[k].attrs.items()}}
            binned = any("nbin" in f[k].attrs and any(a.startswith("bin_") for a in f[k].attrs) for k in kinds)
            main = kinds[0]
            rows = parts[main]["rows"]
            desc = FileDescription(
                path = path,
                file_type = f"TXPipe {'binned ' if binned else ''}catalog (HDF5)",
                represents = (
                    f"Object catalog: {CATALOG_GROUPS[main][0]}"
                    + (f"; {rows:,} rows" if rows else "")
                    + (f"; also contains {kinds[1:]}" if len(kinds) > 1 else "")
                    + (". Objects are split into per-bin subgroups." if binned else ".")
                ),
                plotting = "Generic catalog view. " + CATALOG_GROUPS[main][1],
                deed_support = PLANNED,
                details = parts,
            )
        elif top & DIAGNOSTIC_GROUPS.keys():
            kinds = sorted(top & DIAGNOSTIC_GROUPS.keys())
            desc = FileDescription(
                path = path,
                file_type = "TXPipe diagnostic statistics (HDF5)",
                represents = "Systematics / diagnostic measurements: " + "; ".join(DIAGNOSTIC_GROUPS[k] for k in kinds) + ".",
                plotting = (
                    "Mostly curves vs angular scale with error bars -- the same kind of figure as "
                    "the sacc data vector, so it can reuse that styling. Each statistic has its own "
                    "dataset names, so each needs a small dedicated reader. TXPipe already renders "
                    "these to PNG (e.g. rowe0, tau0), and those could be shown in the meantime."
                ),
                deed_support = PLANNED,
                details = {k: list(f[k].keys())[:40] for k in kinds},
            )
        else:
            cols, rows = _h5_columns(f)
            groups = sorted(k for k in top if isinstance(f[k], h5py.Group) and k != "provenance")
            if cols and rows and all(f[c].shape[:1] == (rows,) for c in cols):
                desc = FileDescription(
                    path = path,
                    file_type = "HDF5 table",
                    represents = (
                        f"A flat table: {len(cols)} columns x {rows:,} rows (tables_io layout, as "
                        "written by RAIL's Hdf5Handle / TableHandle)."
                    ),
                    plotting = "Generic table view: column histograms and 2D density plots; subsample if large.",
                    deed_support = PLANNED,
                    details = {"columns": cols[:40], "rows": rows},
                )
            else:
                desc = FileDescription(
                    path = path,
                    file_type = "HDF5 (unrecognized layout)",
                    represents = f"HDF5 with top-level groups {groups} and datasets {cols[:20]}; not a layout this script knows.",
                    plotting = (
                        "Needs a look by hand. A generic HDF5 tree browser (pick a dataset; 1D -> "
                        "line or histogram, 2D -> heatmap) would cover it as a fallback."
                    ),
                    deed_support = PLANNED,
                    details = {"groups": groups, "datasets": cols[:40]},
                )

    if provenance:
        desc.details["ceci_provenance"] = provenance
    return desc


### ---------------------------------------------------------------- other formats

def describe_parquet(path):
    """ describe_parquet

        Inputs:
        - path (str): path to a Parquet file

        Process:
        - Reads the footer only (schema and row count), not the data

        Outputs:
        - FileDescription
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return _missing_reader(path, "Parquet", "pyarrow")
    meta = pq.ParquetFile(path)
    cols = meta.schema_arrow.names
    return FileDescription(
        path = path,
        file_type = "Parquet table",
        represents = (
            f"A columnar table: {len(cols)} columns x {meta.metadata.num_rows:,} rows. RAIL writes "
            "these (PqHandle); ceci can only read them."
        ),
        plotting = (
            "Generic table view: column histograms, 2D density and RA/Dec scatter. Parquet can "
            "read just the columns asked for, which suits large catalogs."
        ),
        deed_support = PLANNED,
        details = {"columns": cols[:40], "rows": meta.metadata.num_rows},
    )


def describe_text(path):
    """ describe_text

        Inputs:
        - path (str): path to a plain-text file

        Process:
        - A first line of '#'-prefixed, tab-separated "section--name" columns
        is a CosmoSIS-style MCMC chain (what plot.py reads). Otherwise the
        numeric body is loaded to report its shape

        Outputs:
        - FileDescription
    """
    with open(path) as f:
        first = f.readline().strip()
        comments, n_rows = [], 0
        for line in f:
            if line.startswith("#"):
                comments.append(line.strip())
            elif line.strip():
                n_rows += 1

    if first.startswith("#") and "--" in first:
        cols = first.lstrip("#").split()
        sampler = next((c.split("=", 1)[1] for c in comments if c.startswith("#sampler=")), "unknown")
        params = [c for c in cols if "--" in c]
        return FileDescription(
            path = path,
            file_type = "MCMC chain (CosmoSIS text)",
            represents = (
                f"Posterior samples: {n_rows:,} samples of {len(params)} parameters, sampler '{sampler}'. "
                "Each row is one sample with its weight and likelihood."
            ),
            plotting = (
                "Supported today: `plot.py` / `plot_many.py` draw getdist triangle plots of the "
                "selected parameters, with several chains overlaid in `plot_many.py`. Rendered as a "
                "static PNG, so there's no hover."
            ),
            deed_support = SUPPORTED,
            details = {"n_samples": n_rows, "parameters": params, "sampler": sampler},
        )

    try:
        arr = np.loadtxt(path, ndmin = 2, delimiter = "," if path.endswith(".csv") else None)
        shape = arr.shape
    except ValueError:
        shape = None
    header = first if first.startswith("#") else ""
    return FileDescription(
        path = path,
        file_type = "text table" if shape else "text",
        represents = (
            f"A numeric array of shape {shape}{f', header {header!r}' if header else ''}. TXPipe's text "
            "outputs are of this kind: diagnostic arrays (g_psf_T_out, g_snr_out ...), jackknife "
            "patch centers, the rlens measurement and the CMB-cross Monte Carlo correction."
            if shape else "Non-numeric text (a log or report)."
        ),
        plotting = (
            "Small and simple: column 0 against the others as lines or points. The meaning of each "
            "column depends on which stage wrote it, so label axes from the header or by hand."
            if shape else "Show as text."
        ),
        deed_support = PLANNED if shape else DISPLAY_ONLY,
        details = {"shape": shape, "header": header},
    )


### Keys that mark a YAML file as a cosmology (TXPipe FiducialCosmology / CCL dump)
COSMOLOGY_KEYS = {"Omega_c", "Omega_b", "h", "n_s"}


def describe_yaml(path):
    """ describe_yaml

        Inputs:
        - path (str): path to a YAML file

        Process:
        - safe_load (never the unsafe loaders); separates ceci provenance from content

        Outputs:
        - FileDescription
    """
    import yaml

    with open(path) as f:
        content = yaml.safe_load(f) or {}
    if not isinstance(content, dict):
        content = {"value": content}
    provenance = content.pop("provenance", None)
    keys = list(content.keys())

    if COSMOLOGY_KEYS <= set(keys):
        represents = "A cosmology parameter set (TXPipe FiducialCosmology / CCL YAML): " + ", ".join(
            f"{k}={content[k]}" for k in ("Omega_c", "Omega_b", "h", "n_s", "sigma8", "w0", "wa") if k in content
        ) + "."
        plotting = (
            "A key/value table. Also worth marking these as reference lines on the chain triangle "
            "plot, to show where the fiducial sits in the posterior."
        )
    else:
        represents = (
            f"Key/value metadata with keys {keys[:15]}. TXPipe's YAML outputs are tracer metadata "
            "(tracer_metadata_yml) and PSF star statistics (star_psf_stats)."
        )
        plotting = "A key/value table, with nested values flattened; numeric per-bin lists could go in a small bar chart."
    return FileDescription(
        path = path,
        file_type = "YAML",
        represents = represents,
        plotting = plotting,
        deed_support = DISPLAY_ONLY,
        details = {"keys": keys[:40], "ceci_provenance": provenance},
    )


def describe_png(path):
    """ describe_png

        Inputs:
        - path (str): path to a PNG file

        Process:
        - Reads size and text metadata (ceci's PNGFile writes provenance into it)

        Outputs:
        - FileDescription
    """
    from PIL import Image

    with Image.open(path) as im:
        size = im.size
        meta = {k: str(v) for k, v in im.info.items() if isinstance(v, (str, bytes))}
    return FileDescription(
        path = path,
        file_type = "PNG image",
        represents = (
            f"A finished plot, {size[0]}x{size[1]} px. TXPipe writes 64 of these (two-point plots, "
            "map plots, PSF and source diagnostics, n(z) plots)."
        ),
        plotting = (
            "Display only: embed as a base64 <img>, as the chain plots already are, or in a gallery "
            "of a pipeline's plots. The underlying data usually lives in a sibling sacc / HDF5 output "
            "that DEED could plot interactively instead."
        ),
        deed_support = DISPLAY_ONLY,
        details = {"size": size, "metadata": meta},
    )


def describe_pickle(path):
    """ describe_pickle

        Inputs:
        - path (str): path to a pickle file

        Process:
        - Does NOT unpickle: loading a pickle can execute arbitrary code. Instead
        scans the opcode stream with pickletools (which only parses) for the
        module.class names it would import

        Outputs:
        - FileDescription
    """
    classes = []
    try:
        with open(path, "rb") as f:
            strings = []
            for op, arg, _ in pickletools.genops(f):
                if op.name == "GLOBAL":
                    classes.append(arg.replace(" ", "."))
                elif op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                    strings.append(arg)
                elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
                    classes.append(f"{strings[-2]}.{strings[-1]}")
                if len(classes) >= 10:
                    break
    except Exception:
        pass
    return FileDescription(
        path = path,
        file_type = "Python pickle",
        represents = (
            "Serialized Python objects"
            + (f" referencing {sorted(set(classes))}" if classes else "")
            + ". In TXPipe these are a trained tomography classifier (shear_tomography_classifier) "
            "and cluster profiles; in RAIL, trained estimator models (ModelHandle)."
        ),
        plotting = (
            "Don't: unpickling runs arbitrary code, which is unacceptable on a public Spin "
            "deployment, and models hold no plottable data. Plot the outputs the model produced "
            "(the qp PDFs or tomography catalog) instead."
        ),
        deed_support = NOT_PLOTTABLE,
        details = {"referenced_classes": sorted(set(classes))},
    )


def describe_directory(path):
    """ describe_directory

        Inputs:
        - path (str): path to a directory

        Process:
        - Recognizes ceci's Directory (provenance.yml) and FileCollection
        (txpipe_listing.txt), and counts the files by extension (top level only)

        Outputs:
        - FileDescription
    """
    entries = os.listdir(path)
    exts = {}
    for e in entries:
        ext = os.path.splitext(e)[1].lower() or "(none)"
        exts[ext] = exts.get(ext, 0) + 1
    kind = (
        "ceci FileCollection" if "txpipe_listing.txt" in entries
        else "ceci Directory" if "provenance.yml" in entries
        else "directory"
    )
    pngs = exts.get(".png", 0)
    return FileDescription(
        path = path,
        file_type = kind,
        represents = (
            f"A bundle of {len(entries)} files {exts}. TXPipe's FileCollection outputs are mostly "
            "plot folders (lss_density_plots, map_systematic_correlations, lss_weight_summary) or "
            "per-band survey-property maps."
        ),
        plotting = (
            "A gallery of the PNGs." if pngs else
            "Describe each member file with this script and plot them by their own types."
        ),
        deed_support = DISPLAY_ONLY if pngs else PLANNED,
        details = {"n_files": len(entries), "extensions": exts},
    )


def describe_file(path):
    """ describe_file

        Inputs:
        - path (str): path to any ceci / TXPipe / RAIL output, or a directory

        Process:
        - sniff_format picks the format from the contents; the matching
        describe_* function classifies the product within that format

        Outputs:
        - FileDescription: file_type, represents, plotting, deed_support, details
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    fmt = sniff_format(path)
    handlers = {
        "directory": describe_directory,
        "hdf5": describe_hdf5,
        "fits": describe_fits,
        "png": describe_png,
        "parquet": describe_parquet,
        "pickle": describe_pickle,
        "yaml": describe_yaml,
        "text": describe_text,
    }
    if fmt not in handlers:
        return FileDescription(
            path = path,
            file_type = "unknown",
            represents = "Not a format produced by ceci, TXPipe or RAIL.",
            plotting = "Nothing to suggest.",
            deed_support = NOT_PLOTTABLE,
        )
    return handlers[fmt](path)


def format_description(d):
    """ format_description

        Inputs:
        - d (FileDescription)

        Outputs:
        - human-readable multi-line string
    """
    return (
        f"{d.path}\n"
        f"  File type:    {d.file_type}\n"
        f"  DEED support: {d.deed_support}\n"
        f"  Represents:   {d.represents}\n"
        f"  Plotting:     {d.plotting}\n"
    )


def main():
    parser = argparse.ArgumentParser(description = "Describe a ceci / TXPipe / RAIL output and how DEED could plot it")
    parser.add_argument("paths", nargs = "+", help = "file(s) or directory(ies) to describe")
    parser.add_argument("--json", action = "store_true", help = "print JSON, including the details dict")
    args = parser.parse_args()

    results = [describe_file(p) for p in args.paths]
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent = 2, default = str))
    else:
        print("\n".join(format_description(r) for r in results))


if __name__ == "__main__":
    main()
