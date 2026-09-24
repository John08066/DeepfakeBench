"""CPU-only preflight contract tests; torch and dataset imports are mocked."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_prd_environment as check


FFPP_NAMES = ['FaceForensics++', 'FaceShifter', 'DeepFakeDetection',
              'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']


class PreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = patch.dict(os.environ, {
            'DEEPFAKE_DATA_ROOT': str(self.root / 'data'),
            'PRD_PRETRAINED_ROOT': str(self.root / 'weights'),
        })
        environment.start()
        self.addCleanup(environment.stop)

        self.torch = ModuleType('torch')
        self.torch.__version__ = 'mock'
        self.torch.version = SimpleNamespace(cuda='mock')
        self.torch.cuda = SimpleNamespace(is_available=lambda: True,
                                          get_device_name=lambda index: 'mock GPU')
        abstract = ModuleType('dataset.abstract_dataset')
        abstract.FFpp_pool = FFPP_NAMES
        detectors = ModuleType('detectors')
        # Registry deliberately exposes .data, not mapping membership.
        detectors.DETECTOR = SimpleNamespace(data={'lora': object()})
        modules = patch.dict(sys.modules, {
            'torch': self.torch,
            'dataset': ModuleType('dataset'),
            'dataset.abstract_dataset': abstract,
            'detectors': detectors,
        })
        modules.start()
        self.addCleanup(modules.stop)

    def load(self, probe=None, filename='prd_probe_ablation.yaml'):
        source = check.ROOT / 'training/config/detector' / filename
        if probe is None:
            return check.load_config(source)
        import yaml
        config = yaml.safe_load(source.read_text(encoding='utf-8'))
        config['probe'] = probe
        source = self.root / 'detector.json'  # JSON is also valid YAML.
        source.write_text(json.dumps(config), encoding='utf-8')
        return check.load_config(source)

    def touch(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')

    def prepare(self, config, vae_files=('config.json', 'model.safetensors')):
        for filename in ('config.json', 'model.safetensors'):
            self.touch(Path(config['clip_path']) / filename)
        probe_type = config.get('probe', {}).get('type', 'sd15_vae')
        if probe_type in ('sd15_vae', 'sdvae_alt'):
            vae = config['probe']['vae_path'] if probe_type == 'sdvae_alt' else config['vae_path']
            for filename in vae_files:
                self.touch(Path(vae) / filename)
        for name in set(config['train_dataset'] + config['test_dataset']):
            self.touch(Path(config['dataset_json_folder']) / (name + '.json'))
            lmdb_name = 'FaceForensics++' if name in FFPP_NAMES else name
            self.touch(Path(config['lmdb_dir']) / (lmdb_name + '_lmdb') / 'data.mdb')

    def run_preflight(self, config):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            check.preflight(config)
        return json.loads(output.getvalue())

    def test_explicit_baseline_protocols_survive_config_merge(self):
        for filename, test_batch, early_stop in [('prd_probe_ablation.yaml', 64, True),
                                                  ('lora.yaml', 16, False)]:
            with self.subTest(filename=filename):
                config = self.load(filename=filename)
                self.assertEqual(config['train_batchSize'], 32)
                self.assertEqual(config['test_batchSize'], test_batch)
                self.assertEqual(config['early_stopping'],
                                 {'enabled': early_stop, 'max_declines': 3})

    def test_ffpp_datasets_use_the_shared_lmdb(self):
        config = self.load()
        config['test_dataset'] = FFPP_NAMES + ['Celeb-DF-v2']
        self.prepare(config)
        self.assertFalse((Path(config['lmdb_dir']) / 'DeepFakeDetection_lmdb').exists())
        self.assertFalse((Path(config['lmdb_dir']) / 'FF-DF_lmdb').exists())
        self.assertEqual(self.run_preflight(config)['preflight'], 'PASS')

    def test_alternate_vae_uses_its_own_resolved_directory(self):
        config = self.load({'type': 'sdvae_alt', 'vae_path': 'alternative'})
        alternative = self.root / 'weights/alternative'
        self.assertEqual(Path(config['probe']['vae_path']), alternative)
        self.prepare(config)
        self.assertFalse(Path(config['vae_path']).exists())
        report = self.run_preflight(config)
        self.assertEqual(report['model_directories'], [config['clip_path'], str(alternative)])

    def test_alternate_vae_missing_weights_is_rejected(self):
        config = self.load({'type': 'sdvae_alt', 'vae_path': 'alternative'})
        self.prepare(config, vae_files=('config.json',))
        # A complete default VAE must not mask an incomplete alternate VAE.
        for filename in ('config.json', 'model.safetensors'):
            self.touch(Path(config['vae_path']) / filename)
        with self.assertRaises(FileNotFoundError) as error:
            self.run_preflight(config)
        self.assertIn(config['probe']['vae_path'], str(error.exception))

    def test_alternate_vae_missing_config_is_rejected(self):
        config = self.load({'type': 'sdvae_alt', 'vae_path': 'alternative'})
        self.prepare(config, vae_files=('model.safetensors',))
        with self.assertRaises(FileNotFoundError) as error:
            self.run_preflight(config)
        self.assertIn(str(Path(config['probe']['vae_path']) / 'config.json'), str(error.exception))

    def test_alternate_vae_requires_an_explicit_path(self):
        with self.assertRaisesRegex(ValueError, 'requires probe.vae_path'):
            self.load({'type': 'sdvae_alt'})

    def test_non_vae_probes_do_not_require_vae_files(self):
        for probe_type in ('identity', 'gaussian_blur'):
            with self.subTest(probe_type=probe_type):
                config = self.load({'type': probe_type})
                self.prepare(config)
                self.assertFalse(Path(config['vae_path']).exists())
                self.assertEqual(self.run_preflight(config)['model_directories'], [config['clip_path']])

    def test_unknown_model_is_rejected(self):
        config = self.load()
        self.prepare(config)
        config['model_name'] = 'unknown_model'
        with self.assertRaisesRegex(ValueError, 'Unknown model unknown_model'):
            self.run_preflight(config)

    def test_unavailable_cuda_is_rejected_without_touching_a_gpu(self):
        config = self.load()
        self.prepare(config)
        self.torch.cuda.is_available = lambda: False
        with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
            self.run_preflight(config)


if __name__ == '__main__':
    unittest.main()
