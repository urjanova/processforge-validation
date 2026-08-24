from loguru import logger
import logging
import json
import os
import argparse
import tempfile
import shutil
import urllib.parse
import zipfile
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zarr
import boto3
import requests
import json


# helper utilities -----------------------------------------------------------


# Variables that are per-stream scalars / metadata rather than composition
# components.  Used when a store does not explicitly tag its composition.
_KNOWN_STREAM_SCALAR_KEYS = {
    "time",
    "phase",
    "Phase",
    "T",
    "P",
    "flowrate",
    "VaporFrac",
    "H",
    "Cp",
    "K_values",
    "beta",
    "rho",
}


def _ensure_parent_dir(path: str) -> None:
    """Make parent directories for *path* if they don't exist."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _convert_value(value):
    """Convert numpy scalars/arrays to plain Python objects."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return _convert_value(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_pfarchive(path: str) -> bool:
    """Return True if *path* looks like a ProcessStateArchive directory."""
    return (
        path.endswith(".pfarchive")
        or os.path.isdir(os.path.join(path, "outputs", "streams"))
    )


def _normalize_stream_data(data: dict) -> dict:
    """Flatten the optional ``z`` composition dict into top-level columns."""
    normalized = dict(data)
    composition = normalized.pop("z", None)
    if isinstance(composition, dict):
        normalized.update(composition)
    return normalized


def _load_streams_from_zarr(store_path: str) -> tuple[dict, str]:
    """Load stream result dicts from a flattened ProcessForge Zarr store.

    The new (v0.3.1+) layout writes composition arrays directly in the
    stream group and tags them with the ``composition`` group attribute.
    Solver-unit results live in groups with no arrays (attrs only) and are
    ignored.  If a sibling ``<store>.schema.json`` exists, it is used to
    identify stream groups and skip solver-unit groups.
    """
    store = zarr.storage.LocalStore(store_path)
    root = zarr.open(store=store, mode="r")
    mode = root.attrs.get("mode", "steady")

    schema = None
    schema_path = store_path + ".schema.json"
    if os.path.isfile(schema_path):
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
        except Exception as exc:
            logger.warning("Failed to read schema {}: {}", schema_path, exc)

    stream_names = None
    if schema is not None:
        stream_names = set(schema.get("streams", {}).keys())

    streams = {}
    for name in sorted(root.group_keys()):
        if name == "run_info":
            continue
        if stream_names is not None and name not in stream_names:
            continue

        group = root[name]
        arrays = list(group.array_keys())
        if not arrays:
            # Solver-unit groups contain only attributes.
            continue

        composition = set(group.attrs.get("composition", []))
        if not composition:
            # Infer composition from arrays that are not known stream scalars.
            composition = {
                k for k in arrays if k not in _KNOWN_STREAM_SCALAR_KEYS
            }

        data = {}
        for key in arrays:
            value = group[key][:]
            if key == "time":
                data["time"] = value
            else:
                data[key] = value
        streams[name] = data

    return streams, mode


def _load_streams_from_pfarchive(archive_path: str) -> tuple[dict, str]:
    """Load stream result dicts from a ProcessStateArchive directory.

    The canonical layout is ``<base>.pfarchive/outputs/streams/<name>.json``.
    If stream JSONs are absent, falls back to parsing the latest
    ``RunManifest`` in ``runs/<run_id>.json``.
    """
    streams_dir = os.path.join(archive_path, "outputs", "streams")
    if os.path.isdir(streams_dir):
        streams = {}
        for fname in sorted(os.listdir(streams_dir)):
            if not fname.endswith(".json"):
                continue
            stream_name = fname[:-5]
            file_path = os.path.join(streams_dir, fname)
            with open(file_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            streams[stream_name] = _normalize_stream_data(raw)

        mode = "steady"
        for data in streams.values():
            time_values = data.get("time")
            if isinstance(time_values, (list, tuple, np.ndarray)) and len(time_values) > 1:
                mode = "dynamic"
                break
        return streams, mode

    # Fallback: parse the latest RunManifest.
    latest_path = os.path.join(archive_path, "latest_run")
    runs_dir = os.path.join(archive_path, "runs")
    if os.path.isfile(latest_path) and os.path.isdir(runs_dir):
        with open(latest_path, encoding="utf-8") as f:
            run_id = f.read().strip()
        run_path = os.path.join(runs_dir, run_id + ".json")
        if os.path.isfile(run_path):
            with open(run_path, encoding="utf-8") as f:
                manifest = json.load(f)
            streams = {}
            for stream_name, stream_out in manifest.get("streams", {}).items():
                data = {}
                for field in stream_out.get("fields", []):
                    q = field.get("quantity", {})
                    data[field["name"]] = q.get("value")
                streams[stream_name] = _normalize_stream_data(data)
            return streams, manifest.get("mode", "steady")

    raise ValueError(f"Could not find stream results in archive: {archive_path}")


def _streams_to_dataframe(streams: dict, mode: str) -> pd.DataFrame:
    """Convert a ``{stream_name: {var: array|scalar}}`` mapping to a DataFrame."""
    rows = []
    components = set()

    for stream_name, data in sorted(streams.items()):
        time_values = data.get("time")
        if isinstance(time_values, (list, tuple, np.ndarray)):
            n_rows = len(time_values)
        else:
            n_rows = 1

        for idx in range(n_rows):
            row = {"stream": stream_name}
            if time_values is not None:
                row["time"] = float(time_values[idx])

            for var, value in data.items():
                if var == "time":
                    continue
                if isinstance(value, (list, tuple, np.ndarray)):
                    row[var] = _convert_value(value[idx])
                else:
                    row[var] = _convert_value(value)
            rows.append(row)

        for var in data.keys():
            if var not in _KNOWN_STREAM_SCALAR_KEYS:
                components.add(var)

    df = pd.DataFrame(rows)
    df.attrs["components"] = sorted(components)
    return df


# utilities for fetching zarr stores ----------------------------------------

_S3_ENV_VARS = (
    "S3_ACCESS_KEY",
    "S3_SECRET_KEY",
    "S3_REGION_NAME",
    "S3_ENDPOINT_URL",
    "S3_BUCKET_NAME",
)


def _check_s3_env_vars():
    missing = [v for v in _S3_ENV_VARS if not os.environ.get(v)]
    if missing:
        logger.warning(
            "S3 environment variables not set: {}. "
            "S3 access may fail. Consider setting LOCAL_ZARR_DIR instead.",
            ", ".join(missing),
        )
    return missing


def _load_schema(store_path: str) -> dict | None:
    """Load a schema JSON file that sits alongside a zarr store.

    Convention: ``<store_path>.schema.json`` is loaded if it exists.
    Returns ``None`` when no schema file is found.
    """
    schema_path = store_path + ".schema.json"
    if os.path.exists(schema_path):
        logger.info("Found schema: {}", schema_path)
        with open(schema_path) as f:
            return json.load(f)
    return None


def download_zarr_from_s3(s3_url: str, dest_dir: str) -> str:
    """Mirror the contents of an S3 prefix to a local directory.

    ``s3_url`` should be of the form ``s3://bucket/prefix``; the
    entire prefix (including nested subdirectories) will be downloaded
    under ``dest_dir`` preserving the relative structure.  The
    returned string is ``dest_dir`` itself.
    """
    _check_s3_env_vars()

    parsed = urllib.parse.urlparse(s3_url)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
        region_name=os.environ.get("S3_REGION_NAME"),
        endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
    )
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = os.path.relpath(key, prefix)
            target = os.path.join(dest_dir, rel_path)
            target_dir = os.path.dirname(target)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            s3.download_file(bucket, key, target)
    return dest_dir


def fetch_zarr_store(source: str | None = None):
    """Return a local path to a store, downloading if necessary.

    Parameters
    ----------
    source : str or None
        Local filesystem path, or a URL with scheme ``s3://``, ``http://``,
        or ``https://``.  If *None*, the function falls back to the
        ``LOCAL_ZARR_DIR`` environment variable, then to an S3 URL built
        from ``S3_BUCKET_NAME``.

    Returns
    -------
    tuple[str, str|None]
        ``(path, tmpdir)`` where ``path`` points at the directory stored
        on disk, and ``tmpdir`` is a temporary directory that should be
        removed by the caller (or ``None`` if the path is the original
        ``source`` and does not need cleanup).
    """
    if source is None:
        local_dir = os.environ.get("LOCAL_ZARR_DIR")
        s3_bucket = os.environ.get("S3_BUCKET_NAME")
        if local_dir:
            logger.info("Using LOCAL_ZARR_DIR: {}", local_dir)
            source = local_dir
        elif s3_bucket:
            source = f"s3://{s3_bucket}"
        else:
            raise ValueError(
                "No source provided. Pass a path/URL argument or set "
                "LOCAL_ZARR_DIR or S3_BUCKET_NAME environment variables."
            )

    # local directory
    if os.path.isdir(source):
        return source, None

    parsed = urllib.parse.urlparse(source)
    scheme = parsed.scheme.lower()

    if scheme == "s3":
        tmp = tempfile.mkdtemp(prefix="pfzarr-")
        download_zarr_from_s3(source, tmp)
        return tmp, tmp

    elif scheme in ("http", "https"):  # download file
        response = requests.get(source, stream=True)
        response.raise_for_status()
        fname = os.path.basename(parsed.path)
        tmpfile = tempfile.NamedTemporaryFile(
            delete=False, prefix="pfzarr-", suffix=fname
        )
        with tmpfile as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        # if zip archive, unpack
        if fname.lower().endswith(".zip"):
            extract_dir = tempfile.mkdtemp(prefix="pfzarr-")
            with zipfile.ZipFile(tmpfile.name, "r") as zf:
                zf.extractall(extract_dir)
            os.unlink(tmpfile.name)
            return extract_dir, extract_dir
        else:
            # if it is a bare zarr directory packed into a tarball etc we
            # would need additional logic; for now assume zip only.
            raise ValueError(
                "HTTP source must be a .zip archive containing a zarr store"
            )
    else:
        raise ValueError(f"Unsupported URL scheme: {scheme}")


class ProcessForgeValidator:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def _load_dataframe_from_zarr(self, store_path, schema=None):
        store = zarr.storage.LocalStore(store_path)
        root = zarr.open(store=store, mode="r")

        if schema is not None:
            # Schema-driven: use schema for stream/variable discovery
            stream_defs = schema.get("streams", {})
            rows = []
            comp_set = set()
            std_vars = {"P", "T", "flowrate", "Phase", "VaporFrac"}

            for stream_name in sorted(stream_defs):
                info = stream_defs[stream_name]
                variables = info.get("variables", [])
                has_time = info.get("has_time", False)
                comp_names = sorted(v for v in variables if v not in std_vars)
                comp_set.update(comp_names)

                group = root.get(stream_name)
                if group is None:
                    continue

                length = group["time"].shape[0] if has_time and "time" in group else 1
                for idx in range(length):
                    row = {"stream": stream_name}
                    if has_time and "time" in group:
                        try:
                            row["time"] = float(group["time"][idx])
                        except Exception:
                            row["time"] = group["time"][idx]

                    for var in variables:
                        if var == "time":
                            continue
                        arr = group.get(var)
                        if arr is None:
                            continue
                        try:
                            v = arr[idx] if hasattr(arr, "shape") and arr.shape else arr
                            row[var] = v.item() if hasattr(v, "item") else v
                        except Exception:
                            row[var] = arr

                    for comp in comp_names:
                        row.setdefault(comp, 0.0)

                    rows.append(row)

            df = pd.DataFrame(rows)
            for comp in sorted(comp_set):
                if comp in df:
                    df[comp] = df[comp].fillna(0.0)
                else:
                    df[comp] = 0.0
            df.attrs["components"] = sorted(comp_set)
            return df

        # Fallback: heuristic discovery of streams and composition.
        streams, mode = _load_streams_from_zarr(store_path)
        return _streams_to_dataframe(streams, mode)

    def _load_dataframe(self, source):
        """Load a validation DataFrame from a zarr store or .pfarchive path."""
        if _is_pfarchive(source):
            streams, mode = _load_streams_from_pfarchive(source)
        else:
            streams, mode = _load_streams_from_zarr(source)
        return _streams_to_dataframe(streams, mode)

    def _validate_against_schema(self, store_path, schema):
        issues = []
        store = zarr.storage.LocalStore(store_path)
        root = zarr.open(store=store, mode="r")

        for stream_name in schema.get("streams", {}):
            if stream_name not in root:
                issues.append(f"Missing stream: '{stream_name}'")

        for stream_name, stream_info in schema.get("streams", {}).items():
            if stream_name not in root:
                continue
            group = root[stream_name]
            for var_name in stream_info.get("variables", []):
                if var_name not in group:
                    issues.append(
                        f"Stream '{stream_name}': missing variable '{var_name}'"
                    )

        zarr_mode = root.attrs.get("mode")
        schema_mode = schema.get("mode")
        if zarr_mode and schema_mode and zarr_mode != schema_mode:
            issues.append(
                f"Mode mismatch: schema='{schema_mode}', zarr='{zarr_mode}'"
            )

        return issues

    def generate_validation_excel(self, data_source, output_filename):
        """
        Generate a multi-sheet Excel validation report from simulation results.

        data_source: path to a Zarr store or a directory containing one;
                     path to a CSV file; or a pandas DataFrame.
                     When a directory is given, a ``.schema.json`` file is
                     used to discover the matching ``.zarr`` store inside it.
        output_filename: path for the output .xlsx file.
        """
        schema = None
        if isinstance(data_source, str) and os.path.isdir(data_source):
            schema_files = sorted(
                glob.glob(os.path.join(data_source, "*.schema.json"))
            )
            if len(schema_files) == 0:
                # No schemas found inside the directory.  Fall back to
                # looking for a sidecar schema alongside data_source
                # (backward compat for direct .zarr paths).
                schema = _load_schema(data_source)
                store_path = data_source
            elif len(schema_files) == 1:
                schema_path = schema_files[0]
                basename = os.path.basename(schema_path)
                store_name = basename.removesuffix(".schema.json")
                store_path = os.path.join(data_source, store_name)
                schema = _load_schema(store_path)
                if not os.path.isdir(store_path):
                    raise FileNotFoundError(
                        f"Schema {basename} expects a Zarr store at "
                        f"{store_name}, but that directory does not exist."
                    )
            else:
                raise ValueError(
                    f"Multiple schema files found in {data_source}. "
                    f"Please point directly to the desired .zarr directory."
                )
            df = self._load_dataframe_from_zarr(store_path, schema=schema)
        elif isinstance(data_source, str):
            df = pd.read_csv(data_source)
        else:
            df = data_source.copy()

        if "Stream" in df.columns and "stream" not in df.columns:
            df.rename(columns={"Stream": "stream"}, inplace=True)

        # Schema-aware column renaming: P -> P [Pa], flowrate -> flowrate [mol/s], ...
        rename_map = {}
        if schema is not None:
            units_map = {}
            for s_info in schema.get("streams", {}).values():
                for var, unit in s_info.get("units", {}).items():
                    if var not in units_map:
                        units_map[var] = unit
            for var, unit in units_map.items():
                if unit:
                    col = f"{var} [{unit}]"
                    if var in df.columns and col not in df.columns:
                        rename_map[var] = col
        else:
            if "P" in df.columns and "P [Pa]" not in df.columns:
                rename_map["P"] = "P [Pa]"
            if "T" in df.columns and "T [K]" not in df.columns:
                rename_map["T"] = "T [K]"
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        if schema is not None:
            known_cols = {"time", "stream"}
            for s_info in schema.get("streams", {}).values():
                for var, unit in s_info.get("units", {}).items():
                    if var in ("P", "T", "flowrate", "Phase", "VaporFrac"):
                        known_cols.add(f"{var} [{unit}]" if unit else var)
        else:
            known_cols = {
                "time",
                "stream",
                "T [K]",
                "P [Pa]",
                "Phase",
                "phase",
                "VaporFrac",
                "flowrate",
            }
        comp_cols = df.attrs.get("components") or [
            c for c in df.columns if c not in known_cols
        ]
        numeric_comp = (
            df[comp_cols].apply(pd.to_numeric, errors="coerce")
            if comp_cols
            else pd.DataFrame()
        )
        if not numeric_comp.empty:
            df["Total_Fraction"] = numeric_comp.sum(axis=1)
            df["Composition_Alert"] = np.where(
                np.isclose(df["Total_Fraction"], 1.0, atol=1e-5),
                "OK",
                "MASS LEAK",
            )
        else:
            df["Total_Fraction"] = ""
            df["Composition_Alert"] = ""
        mass_ok = (df["Composition_Alert"] == "OK").all()

        pump_check = pd.DataFrame()
        pump_ok = True
        temp_ok = True
        unpaired_pumps = []

        if "stream" in df.columns and "time" in df.columns:
            stream_names = df["stream"].unique()
            pump_ins = sorted([s for s in stream_names if "before_pump" in str(s)])
            pump_outs = sorted([s for s in stream_names if "after_pump" in str(s)])

            paired_outs = set()
            for p_in, p_out in zip(pump_ins, pump_outs):
                df_in = df[df["stream"] == p_in].set_index("time")
                df_out = df[df["stream"] == p_out].set_index("time")
                common_idx = df_in.index.intersection(df_out.index)
                if common_idx.empty:
                    continue
                paired_outs.add(p_out)
                pc = pd.DataFrame(index=common_idx)
                pc["Pump"] = f"{p_in} -> {p_out}"
                pc["Pressure_Gain_Pa"] = (
                    df_out.loc[common_idx, "P [Pa]"].values
                    - df_in.loc[common_idx, "P [Pa]"].values
                )
                pc["Temp_Rise_K"] = (
                    df_out.loc[common_idx, "T [K]"].values
                    - df_in.loc[common_idx, "T [K]"].values
                )
                pc["Pump_Status"] = np.where(
                    pc["Pressure_Gain_Pa"] > 0,
                    "Functional",
                    "Broken",
                )
                pump_check = pd.concat([pump_check, pc])

            if not pump_check.empty:
                pump_ok = (pump_check["Pump_Status"] == "Functional").all()
                temp_ok = (pump_check["Temp_Rise_K"] >= 0).all()

            unpaired_pumps = sorted(set(pump_outs) - paired_outs)
            if unpaired_pumps:
                logger.warning(
                    "No inlet stream found for {}: pump checks skipped. "
                    "Pump inlets are matched by name, so each '{}' outlet needs a "
                    "matching 'before_pump*' inlet stream.",
                    ", ".join(unpaired_pumps),
                    "after_pump*",
                )

        # Schema validation against the store
        schema_issues = []
        schema_ok = True
        if schema is not None and isinstance(data_source, str) and os.path.isdir(data_source):
            schema_issues = self._validate_against_schema(store_path, schema)
            schema_ok = not schema_issues

        summary_rows = []
        if schema is not None:
            summary_rows.append(
                {
                    "Physical Law": "Schema Compliance",
                    "Logic": "Does the zarr store match its schema?",
                    "Status": "PASS" if schema_ok else "FAIL",
                }
            )
            for issue in schema_issues:
                summary_rows.append(
                    {
                        "Physical Law": "Schema Issue",
                        "Logic": issue,
                        "Status": "INFO",
                    }
                )
            summary_rows.append(
                {
                    "Physical Law": "Simulation Mode",
                    "Logic": schema.get("mode", "unknown"),
                    "Status": "INFO",
                }
            )
            summary_rows.append(
                {
                    "Physical Law": "ProcessForge Version",
                    "Logic": schema.get("processforge_version", "unknown"),
                    "Status": "INFO",
                }
            )
        summary_rows.append(
            {
                "Physical Law": "Conservation of Mass",
                "Logic": "Do chemical fractions add to 1.0?",
                "Status": "PASS" if mass_ok else "FAIL",
            }
        )
        if not pump_check.empty:
            summary_rows.append(
                {
                    "Physical Law": "Pump Work (Pressure)",
                    "Logic": "Does the pump increase pressure?",
                    "Status": "PASS" if pump_ok else "FAIL",
                }
            )
            summary_rows.append(
                {
                    "Physical Law": "Thermal Direction",
                    "Logic": "Is the outlet temperature >= inlet?",
                    "Status": "PASS" if temp_ok else "WARNING",
                }
            )
        if unpaired_pumps:
            summary_rows.append(
                {
                    "Physical Law": "Pump Work (Pressure)",
                    "Logic": (
                        "No inlet stream found for "
                        f"{', '.join(unpaired_pumps)}; expected a 'before_pump*' "
                        "stream to compare against"
                    ),
                    "Status": "SKIPPED",
                }
            )
        summary_df = pd.DataFrame(summary_rows)

        _ensure_parent_dir(output_filename)
        with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
            if schema is not None:
                schema_info = pd.DataFrame(
                    [
                        {"Property": "Mode", "Value": schema.get("mode", "")},
                        {
                            "Property": "ProcessForge Version",
                            "Value": schema.get("processforge_version", ""),
                        },
                        {
                            "Property": "Backend",
                            "Value": schema.get("provenance", {}).get("backend", ""),
                        },
                        {
                            "Property": "Git Hash",
                            "Value": schema.get("provenance", {}).get("git_hash", ""),
                        },
                        {
                            "Property": "Created",
                            "Value": schema.get("created", ""),
                        },
                    ]
                )
                schema_info.to_excel(writer, sheet_name="0_SCHEMA_INFO", index=False)
            summary_df.to_excel(writer, sheet_name="1_EXECUTIVE_SUMMARY", index=False)
            if not pump_check.empty:
                pump_check.to_excel(writer, sheet_name="2_PUMP_PERFORMANCE")
            df.to_excel(writer, sheet_name="3_RAW_DATA_CHECKED", index=False)

        logger.info(f"Validation Report Generated: {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate a ProcessForge output store and produce an Excel report."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help=(
            "Local directory or URL (s3://, http(s)://) pointing to a zarr store "
            "or a .pfarchive directory. If omitted, LOCAL_ZARR_DIR or "
            "S3_BUCKET_NAME env vars are used."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="validation.xlsx",
        help="path for the generated Excel file",
    )
    args = parser.parse_args()

    store_path, tmpdir = fetch_zarr_store(args.source)
    try:
        validator = ProcessForgeValidator()
        validator.generate_validation_excel(store_path, args.output)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
