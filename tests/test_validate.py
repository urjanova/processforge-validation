import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import zarr

# Add the project source directory to the path so we import the local module.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"),
)

from validate import ProcessForgeValidator, fetch_zarr_store


class TestProcessForgeValidator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.test_dir, "test_store.zarr")
        self.output_file = os.path.join(self.test_dir, "validation.xlsx")

        # Create a dummy zarr store using the v0.3.1+ flattened layout.
        store = zarr.storage.LocalStore(self.store_path)
        root = zarr.open_group(store=store)
        root.attrs["mode"] = "steady"

        def _make_stream(group, t, p, water_frac):
            group.create_array("T", data=np.array([t]))
            group.create_array("P", data=np.array([p]))
            group.create_array("phase", data=np.array(["Liquid"], dtype="<U10"))
            group.create_array("VaporFrac", data=np.array([0.0]))
            group.create_array("flowrate", data=np.array([10.0]))
            group.create_array("Water", data=np.array([water_frac]))
            group.attrs["composition"] = ["Water"]

        s1 = root.create_group("stream1")
        _make_stream(s1, 300.0, 101325.0, 1.0)

        s2 = root.create_group("stream2_after_pump")
        _make_stream(s2, 305.0, 200000.0, 1.0)

        s3 = root.create_group("stream3")
        _make_stream(s3, 300.0, 101325.0, 0.5)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_local_store_loading(self):
        validator = ProcessForgeValidator()
        df = validator._load_dataframe_from_zarr(self.store_path)
        self.assertEqual(len(df), 3)
        self.assertIn("stream", df.columns)
        self.assertIn("Water", df.columns)

        row1 = df[df["stream"] == "stream1"].iloc[0]
        self.assertAlmostEqual(row1["Water"], 1.0)

        row3 = df[df["stream"] == "stream3"].iloc[0]
        self.assertAlmostEqual(row3["Water"], 0.5)

    def test_validation_report_generation(self):
        validator = ProcessForgeValidator()
        validator.generate_validation_excel(self.store_path, self.output_file)
        self.assertTrue(os.path.exists(self.output_file))

        xl = pd.ExcelFile(self.output_file)
        self.assertIn("1_EXECUTIVE_SUMMARY", xl.sheet_names)
        self.assertIn("3_RAW_DATA_CHECKED", xl.sheet_names)

        summary = pd.read_excel(xl, "1_EXECUTIVE_SUMMARY")
        mass_row = summary[summary["Physical Law"] == "Conservation of Mass"].iloc[0]
        self.assertEqual(mass_row["Status"], "FAIL")


class TestProcessForgeValidatorPfarchive(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.archive_path = os.path.join(self.test_dir, "run.pfarchive")
        self.output_file = os.path.join(self.test_dir, "validation.xlsx")

        streams_dir = os.path.join(self.archive_path, "outputs", "streams")
        os.makedirs(streams_dir, exist_ok=True)

        def _stream_json(t, p, water_frac):
            return {
                "T": [t],
                "P": [p],
                "phase": ["Liquid"],
                "VaporFrac": [0.0],
                "flowrate": [10.0],
                "z": {"Water": [water_frac]},
            }

        for name, payload in [
            ("stream1", _stream_json(300.0, 101325.0, 1.0)),
            ("stream2_after_pump", _stream_json(305.0, 200000.0, 1.0)),
            ("stream3", _stream_json(300.0, 101325.0, 0.5)),
        ]:
            with open(
                os.path.join(streams_dir, f"{name}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(payload, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_pfarchive_loading(self):
        validator = ProcessForgeValidator()
        df = validator._load_dataframe(self.archive_path)
        self.assertEqual(len(df), 3)
        self.assertIn("Water", df.columns)
        self.assertAlmostEqual(df[df["stream"] == "stream1"].iloc[0]["Water"], 1.0)

    def test_pfarchive_report_generation(self):
        validator = ProcessForgeValidator()
        validator.generate_validation_excel(self.archive_path, self.output_file)
        self.assertTrue(os.path.exists(self.output_file))

        xl = pd.ExcelFile(self.output_file)
        summary = pd.read_excel(xl, "1_EXECUTIVE_SUMMARY")
        mass_row = summary[summary["Physical Law"] == "Conservation of Mass"].iloc[0]
        self.assertEqual(mass_row["Status"], "FAIL")


class TestProcessForgeValidatorSchema(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.test_dir, "test_store.zarr")
        self.schema_path = self.store_path + ".schema.json"
        self.output_file = os.path.join(self.test_dir, "validation.xlsx")

        # Build a zarr store with both stream scalars (incl. lowercase
        # "phase" and "time") and a real composition component ("Water").
        store = zarr.storage.LocalStore(self.store_path)
        root = zarr.open_group(store=store)
        root.attrs["mode"] = "steady"

        def _make_stream(group, t, p, water_frac):
            group.create_array("T", data=np.array([t]))
            group.create_array("P", data=np.array([p]))
            group.create_array("phase", data=np.array(["Liquid"], dtype="<U10"))
            group.create_array("VaporFrac", data=np.array([0.0]))
            group.create_array("flowrate", data=np.array([10.0]))
            group.create_array("Water", data=np.array([water_frac]))
            group.attrs["composition"] = ["Water"]

        s1 = root.create_group("stream1")
        _make_stream(s1, 300.0, 101325.0, 1.0)
        s2 = root.create_group("stream2_after_pump")
        _make_stream(s2, 305.0, 200000.0, 1.0)
        s3 = root.create_group("stream3")
        _make_stream(s3, 300.0, 101325.0, 0.5)

        # A ResultSchema-shaped sidecar describing the store.  Includes the
        # scalar "phase" variable, which must NOT be treated as a component.
        schema = {
            "version": 1,
            "store_type": "simulation_results",
            "created": "2026-01-01T00:00:00+00:00",
            "mode": "steady",
            "processforge_version": "0.3.15",
            "provenance": {"backend": "scipy", "git_hash": "abc123"},
            "streams": {
                name: {
                    "variables": [
                        "T",
                        "P",
                        "phase",
                        "VaporFrac",
                        "flowrate",
                        "Water",
                    ],
                    "dtypes": {},
                    "units": {
                        "T": "K",
                        "P": "Pa",
                        "flowrate": "mol/s",
                        "Water": "",
                    },
                    "shape": [1, 6],
                    "has_time": False,
                    "has_phase": True,
                }
                for name in ("stream1", "stream2_after_pump", "stream3")
            },
            "solver_units": {},
        }
        with open(self.schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_schema_driven_components(self):
        # Point at the directory holding both the .zarr and .schema.json.
        validator = ProcessForgeValidator()
        df = validator._load_dataframe_from_zarr(self.store_path, schema=json.load(open(self.schema_path)))

        self.assertEqual(df.attrs["components"], ["Water"])
        self.assertIn("Water", df.columns)
        self.assertIn("phase", df.columns)
        # Scalars / metadata must not leak into the composition set.
        self.assertNotIn("phase", df.attrs["components"])
        self.assertNotIn("time", df.attrs["components"])
        self.assertNotIn("T", df.attrs["components"])

    def test_schema_driven_report(self):
        validator = ProcessForgeValidator()
        validator.generate_validation_excel(self.test_dir, self.output_file)
        self.assertTrue(os.path.exists(self.output_file))

        xl = pd.ExcelFile(self.output_file)
        self.assertIn("0_SCHEMA_INFO", xl.sheet_names)
        self.assertIn("1_EXECUTIVE_SUMMARY", xl.sheet_names)
        self.assertIn("3_RAW_DATA_CHECKED", xl.sheet_names)

        summary = pd.read_excel(xl, "1_EXECUTIVE_SUMMARY")
        comp = summary[summary["Physical Law"] == "Conservation of Mass"].iloc[0]
        self.assertEqual(comp["Status"], "FAIL")  # stream3 Water=0.5

        schema_row = summary[summary["Physical Law"] == "Schema Compliance"].iloc[0]
        self.assertEqual(schema_row["Status"], "PASS")

        info = pd.read_excel(xl, "0_SCHEMA_INFO")
        self.assertEqual(
            info[info["Property"] == "ProcessForge Version"]["Value"].iloc[0],
            "0.3.15",
        )


class TestFetchZarrStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.zip_path = os.path.join(self.test_dir, "store.zip")

        # Build a real zarr v3 directory store and zip it up manually.
        store_path = os.path.join(self.test_dir, "store.zarr")
        store = zarr.storage.LocalStore(store_path)
        root = zarr.open_group(store=store)
        root.create_group("s1")

        with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root_dir, _dirs, files in os.walk(store_path):
                for file in files:
                    file_path = os.path.join(root_dir, file)
                    arcname = os.path.relpath(file_path, store_path)
                    zf.write(file_path, arcname)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_local_directory(self):
        path, tmp = fetch_zarr_store(self.test_dir)
        self.assertEqual(path, self.test_dir)
        self.assertIsNone(tmp)

    @patch("requests.get")
    def test_http_zip_download(self, mock_get):
        with open(self.zip_path, "rb") as f:
            zip_content = f.read()

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [zip_content]
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        url = "http://example.com/data.zip"
        path, tmp = fetch_zarr_store(url)

        self.assertTrue(os.path.isdir(path))
        self.assertTrue(os.path.exists(os.path.join(path, "s1")))
        self.assertIsNotNone(tmp)

        shutil.rmtree(tmp)

    @patch("boto3.client")
    def test_s3_download(self, mock_boto):
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3

        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "prefix/s1/.zgroup"}]}
        ]

        url = "s3://bucket/prefix"
        path, tmp = fetch_zarr_store(url)

        self.assertTrue(os.path.isdir(path))
        self.assertIsNotNone(tmp)
        mock_s3.download_file.assert_called()

        shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
