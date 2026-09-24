"""CPU-only checks for relocation, output isolation, and explicit protocols."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from path_config import (PROJECT_ROOT, project_path, resolve_data_paths,
                         resolve_pretrained_path, resolve_output_path, resolve_rgb_path)
from early_stopping import ConsecutiveDeclineStopper
from experiment_summary import write_experiment_summary


class PortablePathsTests(unittest.TestCase):
    def test_defaults_and_cwd_independence(self):
        config = dict(rgb_dir='rgb', lmdb_dir='lmdb', dataset_json_folder='json')
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            try:
                os.chdir(directory)
                resolve_data_paths(config)
                self.assertEqual(config['lmdb_dir'], str(PROJECT_ROOT / 'datasets/lmdb'))
                self.assertEqual(project_path('training/train.py'), PROJECT_ROOT / 'training/train.py')
                self.assertEqual(resolve_pretrained_path({}, 'clip_path', 'clip'), str(PROJECT_ROOT / 'pretrained/clip'))
            finally:
                os.chdir(previous)

    def test_environment_overrides_config(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(DEEPFAKE_DATA_ROOT=directory, PRD_PRETRAINED_ROOT=directory,
                       PRD_OUTPUT_ROOT=directory)
            with patch.dict(os.environ, env):
                config = dict(data_root='ignored', rgb_dir='rgb', lmdb_dir='lmdb', dataset_json_folder='json')
                resolve_data_paths(config)
                self.assertEqual(config['data_root'], str(Path(directory).resolve()))
                self.assertEqual(resolve_output_path('logs/run-a'), Path(directory) / 'logs/run-a')
                self.assertNotEqual(resolve_output_path('logs/run-a'), resolve_output_path('logs/run-b'))

    def test_rgb_legacy_prefix_is_relative_to_data_root(self):
        root = PROJECT_ROOT / 'datasets/rgb'
        for name in ['FF/real/1.png', './FF/real/1.png', './datasets\\FF\\real\\1.png']:
            self.assertEqual(Path(resolve_rgb_path(name, root)), root / 'FF/real/1.png')

    def test_decline_rule_not_best_score_patience(self):
        stopper = ConsecutiveDeclineStopper()
        self.assertEqual([stopper.update(x) for x in [.9, .8, .85, .8, .7, .6]],
                         [False, False, False, False, False, True])

    def test_summary_stays_with_run(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'training.log'
            log.write_text('smoke log\n', encoding='utf-8')
            summary = write_experiment_summary(log, 'portability test')
            self.assertEqual(summary.parent, log.parent)


if __name__ == '__main__':
    unittest.main()
