import os
import shutil
import tempfile
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import zarr

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validate import ProcessForgeValidator, fetch_zarr_store

class TestProcessForgeValidator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.test_dir, "test_store.zarr")
        self.output_file = os.path.join(self.test_dir, "validation.xlsx")

        # Create a dummy zarr store (v3 API)
        store = zarr.storage.LocalStore(self.store_path)
        root = zarr.open_group(store=store, mode='w')
        
        # Attribute 'mode'
        root.attrs["mode"] = "steady"

        # Stream 1
        s1 = root.create_group("stream1")
        s1.create_array("T [K]", data=np.array([300.0]))
        s1.create_array("P [Pa]", data=np.array([101325.0]))
        s1.create_array("Phase", data=np.array(["Liquid"], dtype='<U10'))
        s1.create_array("VaporFrac", data=np.array([0.0]))
        s1.create_array("flowrate", data=np.array([10.0]))
        comp1 = s1.create_group("__composition__")
        comp1.create_array("Water", data=np.array([1.0]))

        # Stream 2 (pump outlet)
        s2 = root.create_group("stream2_after_pump")
        s2.create_array("T [K]", data=np.array([305.0]))
        s2.create_array("P [Pa]", data=np.array([200000.0])) # higher pressure
        s2.create_array("Phase", data=np.array(["Liquid"], dtype='<U10'))
        s2.create_array("VaporFrac", data=np.array([0.0]))
        s2.create_array("flowrate", data=np.array([10.0]))
        comp2 = s2.create_group("__composition__")
        comp2.create_array("Water", data=np.array([1.0]))
        
        # Stream 3 (failed mass balance)
        s3 = root.create_group("stream3")
        s3.create_array("T [K]", data=np.array([300.0]))
        s3.create_array("P [Pa]", data=np.array([101325.0]))
        s3.create_array("Phase", data=np.array(["Liquid"], dtype='<U10'))
        s3.create_array("VaporFrac", data=np.array([0.0]))
        s3.create_array("flowrate", data=np.array([5.0]))
        comp3 = s3.create_group("__composition__")
        comp3.create_array("Water", data=np.array([0.5])) # incomplete fraction

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_local_store_loading(self):
        validator = ProcessForgeValidator()
        df = validator._load_dataframe_from_zarr(self.store_path)
        self.assertEqual(len(df), 3)
        self.assertIn("stream", df.columns)
        self.assertIn("Water", df.columns)
        
        # check specific values
        row1 = df[df["stream"] == "stream1"].iloc[0]
        self.assertAlmostEqual(row1["Water"], 1.0)
        
        row3 = df[df["stream"] == "stream3"].iloc[0]
        self.assertAlmostEqual(row3["Water"], 0.5)

    def test_validation_report_generation(self):
        validator = ProcessForgeValidator()
        validator.generate_validation_excel(self.store_path, self.output_file)
        self.assertTrue(os.path.exists(self.output_file))
        
        # verify content using pandas
        xl = pd.ExcelFile(self.output_file)
        self.assertIn("1_EXECUTIVE_SUMMARY", xl.sheet_names)
        self.assertIn("3_RAW_DATA_CHECKED", xl.sheet_names)
        
        summary = pd.read_excel(xl, "1_EXECUTIVE_SUMMARY")
        # Mass balance fails because of stream3
        mass_row = summary[summary["Physical Law"] == "Conservation of Mass"].iloc[0]
        self.assertEqual(mass_row["Status"], "FAIL")


class TestFetchZarrStore(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.zip_path = os.path.join(self.test_dir, "store.zip")
        
        # Create dummy zip with a file inside (zarr v3 API)
        with zarr.storage.ZipStore(self.zip_path, mode='w') as store:
            root = zarr.open_group(store=store, mode='w')
            root.create_group("s1")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_local_directory(self):
        path, tmp = fetch_zarr_store(self.test_dir)
        self.assertEqual(path, self.test_dir)
        self.assertIsNone(tmp)

    @patch("requests.get")
    def test_http_zip_download(self, mock_get):
        # Mock response content
        with open(self.zip_path, "rb") as f:
            zip_content = f.read()

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [zip_content]
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        url = "http://example.com/data.zip"
        path, tmp = fetch_zarr_store(url)
        
        # Should have unzipped to a temp dir
        self.assertTrue(os.path.isdir(path))
        self.assertTrue(os.path.exists(os.path.join(path, "s1")))
        self.assertIsNotNone(tmp)
        
        # Cleanup
        shutil.rmtree(tmp)

    @patch("boto3.client")
    def test_s3_download(self, mock_boto):
        # Setup mock s3 client
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        
        # Mock list_objects_v2 pagination
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "prefix/s1/.zgroup"}]}
        ]
        
        url = "s3://bucket/prefix"
        path, tmp = fetch_zarr_store(url)
        
        self.assertTrue(os.path.isdir(path))
        self.assertIsNotNone(tmp)
        
        # Verify download_file was called
        mock_s3.download_file.assert_called()
        
        # Cleanup
        shutil.rmtree(tmp)

if __name__ == "__main__":
    unittest.main()
