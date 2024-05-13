import unittest
from unittest.mock import patch, MagicMock
import plmfit.functions.extract_embeddings as extract_embeddings

class TestExtractEmbeddings(unittest.TestCase):
    def setUp(self):
        # Mock the logger and model
        self.mock_logger = MagicMock()
        self.mock_model = MagicMock()

        # Patch the init_plm function to return the mock model
        self.patcher_init_plm = patch('plmfit.shared_utils.utils.init_plm', return_value=self.mock_model)
        self.mock_init_plm = self.patcher_init_plm.start()

        # Setup arguments
        self.args = MagicMock()
        self.args.plm = 'proteinbert'
        self.args.data_type = 'testing'
        self.args.layer = 'last'
        self.args.reduction = 'mean'
        self.args.experimenting = 'True'

    def tearDown(self):
        self.patcher_init_plm.stop()

    def test_extract_embeddings_success(self):
        # Run the function with the setup arguments and logger
        extract_embeddings(self.args, self.mock_logger)

        # Assert init_plm was called correctly
        self.mock_init_plm.assert_called_once_with(self.args.plm, self.mock_logger)

        # Assert the model's extract_embeddings was called correctly
        self.mock_model.extract_embeddings.assert_called_once_with(
            data_type=self.args.data_type, layer=self.args.layer, reduction=self.args.reduction
        )

    def test_extract_embeddings_failure(self):
        # Setup the init_plm to return None to simulate model not initializing
        self.mock_init_plm.return_value = None

        # Run the function and expect an assertion error
        with self.assertRaises(AssertionError) as context:
            extract_embeddings(self.args, self.mock_logger)
        
        # Check that the error message is as expected
        self.assertEqual(str(context.exception), 'Model is not initialized')

if __name__ == '__main__':
    unittest.main()
