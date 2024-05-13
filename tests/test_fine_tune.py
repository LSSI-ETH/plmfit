import unittest
from unittest.mock import patch, MagicMock
import plmfit.functions.fine_tune as fine_tune
import tempfile

class TestFineTune(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock()
        self.args = MagicMock()
        self.args.layer = 'first'
        self.args.plm = 'proteinbert'
        self.args.data_type = 'testing-aav'
        self.args.reduction = 'mean'
        self.args.experimenting = 'True'
        self.args.split = None

    def run_fine_tune_tests(self, ft_method, heads_to_test):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.args.output_dir = temp_dir
            self.args.experiment_dir = temp_dir
            self.mock_logger.base_dir = temp_dir
            for head_config in heads_to_test:
                with self.subTest(ft_method=ft_method, head_config=head_config):
                    self.args.ft_method = ft_method
                    self.args.head_config = head_config
                    with self.assertRaises(SystemExit) as context:
                        fine_tune(self.args, self.mock_logger)
                    self.assertEqual(str(context.exception), 'Experiment over', 
                                     f"Failed for ft_method={ft_method}, head_config={head_config}")

    def test_fine_tune_full(self):
        heads_to_test = [
            'mlm_head_config.json',
            'full_linear_head_config.json',
            'full_linear_classification_head_config.json'
        ]
        self.run_fine_tune_tests('full', heads_to_test)

    def test_fine_tune_lora(self):
        heads_to_test = [
            'mlm_head_config.json',
            'lora_linear_head_config.json',
            'lora_linear_classification_head_config.json'
        ]
        self.run_fine_tune_tests('lora', heads_to_test)

    def test_fine_tune_adapters(self):
        heads_to_test = [
            'mlm_head_config.json',
            'bottleneck_linear_head_config.json',
            'bottleneck_linear_classification_head_config.json'
        ]
        self.run_fine_tune_tests('bottleneck_adapters', heads_to_test)

if __name__ == '__main__':
    unittest.main()
