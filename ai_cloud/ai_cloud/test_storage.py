import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from storage import store


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.temp_path = tempfile.mkdtemp()
        self.model_name = "example_model"
        self.bucket = "example_bucket"
        self.key = "example_key"
        self.model = {}

    def tearDown(self):
        shutil.rmtree(self.temp_path, ignore_errors=False, onerror=None)

    @patch("boto3.client")
    def test_s3_storage(self, mock_client):
        mock_s3_client = MagicMock()
        s3_store = store(
            backend="s3",
            model_name=self.model_name,
            path=self.temp_path,
            bucket=self.bucket,
            key=self.key,
        )
        pass
        #TODO: add tests

    @patch("joblib.dump")
    @patch("joblib.load")
    def test_disk_storage(self, mock_load, mock_dump):
        mock_load.return_value = self.model
        disk_store = store(
            backend="disk", model_name=self.model_name, path=self.temp_path
        )
        disk_store.put(self.model)
        mock_dump.assert_called_with(
            self.model, str(Path(self.temp_path) / self.model_name)
        )
        model = disk_store.get()
        mock_load.assert_called_with(str(Path(self.temp_path) / self.model_name))
        self.assertEqual(model, self.model)


if __name__ == "__main__":
    unittest.main()
