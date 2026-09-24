"""Linux launcher tests with temporary checkouts; never imports ML or trains."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
FAKE_PYTHON = r'''import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
if args[0] == '-c':
    sys.path.insert(0, os.getcwd())
    sys.argv = ['-c', *args[2:]]
    exec(args[1], {'__name__': '__main__'})
else:
    assert args[:2] == ['-u', 'training/train.py'], args
    print(json.dumps({'cwd': os.getcwd(), 'args': args,
                      'gpu': os.environ['CUDA_VISIBLE_DEVICES'],
                      'data': os.environ.get('DEEPFAKE_DATA_ROOT')}), flush=True)
    if os.environ.get('FAKE_READY'):
        Path(os.environ['FAKE_READY']).touch()
        deadline = time.monotonic() + 10
        while not Path(os.environ['FAKE_RELEASE']).exists():
            if time.monotonic() > deadline:
                raise RuntimeError('test did not release fake training')
            time.sleep(.05)
    sys.exit(int(os.environ.get('FAKE_EXIT_CODE', '0')))
'''


@unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('bash')
                     and shutil.which('flock'), 'requires Linux bash and flock')
class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='prd launcher ')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.env = os.environ.copy()
        for key in ('PRD_OUTPUT_ROOT', 'FAKE_READY', 'FAKE_RELEASE', 'FAKE_EXIT_CODE'):
            self.env.pop(key, None)
        self.fake_python = self.directory / 'fakepython'
        self.fake_python.write_text('#!' + sys.executable + '\n' + FAKE_PYTHON)
        self.fake_python.chmod(0o700)
        self.env['PYTHON_BIN'] = str(self.fake_python)
        self.env['CUDA_VISIBLE_DEVICES'] = '3'
        self.env['DEEPFAKE_DATA_ROOT'] = 'machine-specific-data'
        self.checkout = self.make_checkout('checkout one')

    def make_checkout(self, name):
        checkout = self.directory / name
        (checkout / 'scripts').mkdir(parents=True)
        (checkout / 'training').mkdir()
        (checkout / 'training/__init__.py').touch()
        # Normalize source checkout CRLF only inside the disposable Linux fixture.
        (checkout / 'scripts/launch_prd_training.sh').write_text(
            (ROOT / 'scripts/launch_prd_training.sh').read_text())
        shutil.copyfile(ROOT / 'training/path_config.py', checkout / 'training/path_config.py')
        (checkout / 'scripts/watch_prd_early_stop.py').write_text(
            'def project_training():\n    return []\n')
        (checkout / 'detector.yaml').write_text('model_name: lora\n')
        return checkout

    def command(self, checkout, task='test_run'):
        return ['bash', str(checkout / 'scripts/launch_prd_training.sh'), 'detector.yaml', task]

    def run_launcher(self, checkout=None, env=None, task='test_run'):
        return subprocess.run(self.command(checkout or self.checkout, task),
                              cwd=self.directory, env=env or self.env,
                              text=True, capture_output=True, timeout=10)

    def test_self_location_and_configured_output_root(self):
        output = self.directory / 'external output'
        self.env['PRD_OUTPUT_ROOT'] = str(output)
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        console = output / 'logs/test_run/console.log'
        record = next(json.loads(line) for line in console.read_text().splitlines()
                      if line.startswith('{'))
        self.assertEqual(record['cwd'], str(self.checkout))
        self.assertEqual(record['gpu'], '3')
        self.assertEqual(record['data'], 'machine-specific-data')
        self.assertEqual(record['args'][-4:], ['--detector_path', 'detector.yaml',
                                              '--task_target', 'test_run'])
        self.assertFalse((self.checkout / 'logs').exists())
        self.assertTrue((self.checkout / '.state/prd_training.lock').exists())

    def test_training_failure_exit_code_is_preserved(self):
        self.env['FAKE_EXIT_CODE'] = '7'
        result = self.run_launcher()
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn('TRAINING_EXIT_CODE=7',
                      (self.checkout / 'logs/test_run/console.log').read_text())

    def test_existing_console_is_not_overwritten(self):
        first = self.run_launcher()
        self.assertEqual(first.returncode, 0, first.stderr)
        console = self.checkout / 'logs/test_run/console.log'
        original = console.read_bytes()
        second = self.run_launcher()
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(console.read_bytes(), original)

    def start_waiting_launcher(self):
        ready, release = self.directory / 'ready', self.directory / 'release'
        env = self.env | {'FAKE_READY': str(ready), 'FAKE_RELEASE': str(release)}
        process = subprocess.Popen(self.command(self.checkout), cwd=self.directory,
                                   env=env, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        def finish():
            release.touch()
            process.communicate(timeout=15)
        self.addCleanup(finish)
        deadline = time.monotonic() + 5
        while not ready.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                self.fail('fake training failed to acquire the launcher lock')
            time.sleep(.05)
        return process, release

    def test_same_checkout_cannot_launch_twice(self):
        _, release = self.start_waiting_launcher()
        result = self.run_launcher(task='second_run')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('holds the training lock', result.stdout)
        self.assertFalse((self.checkout / 'logs/second_run').exists())
        release.touch()

    def test_two_checkouts_can_launch_independently(self):
        process, release = self.start_waiting_launcher()
        other = self.make_checkout('checkout two')
        result = self.run_launcher(checkout=other)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(process.poll())
        self.assertTrue((other / 'logs/test_run/console.log').exists())
        self.assertTrue((self.checkout / 'logs/test_run/console.log').exists())
        release.touch()


if __name__ == '__main__':
    unittest.main()
