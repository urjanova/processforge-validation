from loguru import logger
import logging
import os
import argparse
import tempfile
import shutil
import urllib.parse
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zarr
import boto3
import requests


# helper utilities -----------------------------------------------------------


def _ensure_parent_dir(path: str) -> None:
    """Make parent directories for *path* if they don't exist."""
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _build_dataframe_row(group, stream, idx, comp_names, has_time):
    """Construct a single row from a zarr group.

    Parameters
    ----------
    group : zarr.hierarchy.Group
        Zarr group corresponding to one stream.
    stream : str
        Name of the stream.
    idx : int
        Time index (or 0 for steady).  If ``has_time`` is False this is ignored.
    comp_names : list[str]
        List of component names present in ``__composition__`` sub‑group.
    has_time : bool
        Whether the group contains a ``time`` dataset.
    """
    row = {"stream": stream}
    if has_time:
        try:
            row["time"] = float(group["time"][idx])
        except Exception:
            row["time"] = group["time"][idx]

    # copy all top‑level datasets except the composition group
    for key, val in group.members():
        if key in ("time", "__composition__"):
            continue
        try:
            # handle indexed arrays; zarr v3 returns 0-d ndarray on scalar index
            v = val[idx] if hasattr(val, "shape") and val.shape else val
            row[key] = v.item() if hasattr(v, "item") else v
        except Exception:
            row[key] = val

    comp_group = group.get("__composition__")
    if comp_group is not None:
        for comp in comp_names:
            arr = comp_group.get(comp)
            if arr is None:
                row[comp] = 0.0
            else:
                try:
                    v = arr[idx] if has_time else arr
                    row[comp] = v.item() if hasattr(v, "item") else v
                except Exception:
                    row[comp] = arr
    return row


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
    """Return a local path to a Zarr store, downloading if necessary.

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

    def _load_dataframe_from_zarr(self, store_path):
        store = zarr.storage.LocalStore(store_path)
        root = zarr.open(store=store, mode="r")
        streams = sorted(root.group_keys())
        rows = []
        components = set()
        mode = root.attrs.get("mode", "steady")
        for stream in streams:
            group = root[stream]
            comp_group = group.get("__composition__")
            comp_names = sorted(comp_group.keys()) if comp_group is not None else []
            components.update(comp_names)
            has_time = "time" in group and mode == "dynamic"
            length = group["time"].shape[0] if has_time else 1
            for idx in range(length):
                rows.append(
                    _build_dataframe_row(group, stream, idx, comp_names, has_time)
                )
        df = pd.DataFrame(rows)
        for comp in sorted(components):
            if comp in df:
                df[comp] = df[comp].fillna(0.0)
            else:
                df[comp] = 0.0
        return df

    def generate_validation_excel(self, data_source, output_filename):
        """
        Generate a multi-sheet Excel validation report from simulation results.

        data_source: path to a Zarr store, CSV file, or pandas DataFrame.
        output_filename: path for the output .xlsx file.
        """
        if isinstance(data_source, str) and os.path.isdir(data_source):
            df = self._load_dataframe_from_zarr(data_source)
        elif isinstance(data_source, str):
            df = pd.read_csv(data_source)
        else:
            df = data_source.copy()

        if "Stream" in df.columns and "stream" not in df.columns:
            df.rename(columns={"Stream": "stream"}, inplace=True)

        # normalise bare unit-less column names produced by zarr stores
        rename_map = {}
        if "P" in df.columns and "P [Pa]" not in df.columns:
            rename_map["P"] = "P [Pa]"
        if "T" in df.columns and "T [K]" not in df.columns:
            rename_map["T"] = "T [K]"
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        known_cols = {
            "time",
            "stream",
            "T [K]",
            "P [Pa]",
            "Phase",
            "VaporFrac",
            "flowrate",
        }
        comp_cols = [c for c in df.columns if c not in known_cols]
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

        if "stream" in df.columns and "time" in df.columns:
            stream_names = df["stream"].unique()
            pump_ins = sorted([s for s in stream_names if "before_pump" in str(s)])
            pump_outs = sorted([s for s in stream_names if "after_pump" in str(s)])

            for p_in, p_out in zip(pump_ins, pump_outs):
                df_in = df[df["stream"] == p_in].set_index("time")
                df_out = df[df["stream"] == p_out].set_index("time")
                common_idx = df_in.index.intersection(df_out.index)
                if common_idx.empty:
                    continue
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

        summary_rows = [
            {
                "Physical Law": "Conservation of Mass",
                "Logic": "Do chemical fractions add to 1.0?",
                "Status": "PASS" if mass_ok else "FAIL",
            }
        ]
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
        summary_df = pd.DataFrame(summary_rows)

        _ensure_parent_dir(output_filename)
        with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="1_EXECUTIVE_SUMMARY", index=False)
            if not pump_check.empty:
                pump_check.to_excel(writer, sheet_name="2_PUMP_PERFORMANCE")
            df.to_excel(writer, sheet_name="3_RAW_DATA_CHECKED", index=False)

        logger.info(f"Validation Report Generated: {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate a ProcessForge zarr store and produce an Excel report."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=None,
        help=(
            "Local directory or URL (s3://, http(s)://) pointing to a zarr store. "
            "If omitted, LOCAL_ZARR_DIR or S3_BUCKET_NAME env vars are used."
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
