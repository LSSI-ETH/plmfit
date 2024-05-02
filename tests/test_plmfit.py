import unittest
from unittest.mock import patch, mock_open, MagicMock
import sys

import plmfit.__main__ as plmfit_main

class TestPLMFitMainInvocation(unittest.TestCase):
    def setUp(self):
        # Mock os.makedirs to prevent actual file system modifications
        self.patcher_makedirs = patch('os.makedirs')
        self.mock_makedirs = self.patcher_makedirs.start()

        # Mock open to prevent actual file operations
        self.patcher_open = patch('builtins.open', mock_open())
        self.mock_open = self.patcher_open.start()

    def tearDown(self):
        self.patcher_makedirs.stop()
        self.patcher_open.stop()
        

    # Test running of functions and fine tuning methods
    def test_run_extract_embeddings(self):
        test_args = [
            'plmfit', '--function=extract_embeddings', '--experiment_dir=/fake/dir', '--logger=local'
        ]
        with patch('sys.argv', test_args), patch('plmfit.__main__.run_extract_embeddings') as mock_run:
            plmfit_main.main()
            mock_run.assert_called_once()
            self.mock_makedirs.assert_any_call('/fake/dir', exist_ok=True)

    def test_run_feature_extraction(self):
        test_args = [
            'plmfit', '--function=fine_tuning', '--ft_method=feature_extraction', '--experiment_dir=/fake/dir',
            '--logger=local'
        ]
        with patch('sys.argv', test_args), patch('plmfit.__main__.run_feature_extraction') as mock_run:
            plmfit_main.main()
            mock_run.assert_called_once()
            self.mock_makedirs.assert_any_call('/fake/dir', exist_ok=True)

    def test_run_lora(self):
        test_args = [
            'plmfit', '--function=fine_tuning', '--ft_method=lora', '--experiment_dir=/fake/dir',
            '--logger=local'
        ]
        with patch('sys.argv', test_args), patch('plmfit.__main__.run_lora') as mock_run:
            plmfit_main.main()
            mock_run.assert_called_once()
            self.mock_makedirs.assert_any_call('/fake/dir', exist_ok=True)

    def test_run_onehot(self):
        test_args = [
            'plmfit', '--function=one_hot', '--experiment_dir=/fake/dir', '--logger=local'
        ]
        with patch('sys.argv', test_args), patch('plmfit.__main__.run_onehot') as mock_run:
            plmfit_main.main()
            mock_run.assert_called_once()
            self.mock_makedirs.assert_any_call('/fake/dir', exist_ok=True)

    

if __name__ == '__main__':
    unittest.main()
