"""Read-only PRD preflight; --smoke explicitly runs one tiny train/eval batch."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from path_config import project_path, resolve_data_paths, resolve_pretrained_path, resolve_output_path


def load_config(path):
    import yaml
    with project_path(path).open(encoding='utf-8') as file:
        config = yaml.safe_load(file)
    with (ROOT / 'training/config/train_config.yaml').open(encoding='utf-8') as file:
        shared = yaml.safe_load(file)
    if 'label_dict' in config:
        shared['label_dict'] = config['label_dict']
    config.update(shared)
    resolve_data_paths(config)
    resolve_pretrained_path(config, 'clip_path', 'clip-vit-large-patch14')
    resolve_pretrained_path(config, 'vae_path', 'vae_original')
    if (config.get('probe') or {}).get('type') == 'sdvae_alt':
        alt_path = config['probe'].get('vae_path')
        if not alt_path:
            raise ValueError('probe.type=sdvae_alt requires probe.vae_path')
        config['probe']['vae_path'] = resolve_pretrained_path(dict(config, vae_path=alt_path), 'vae_path', alt_path)
    return config


def preflight(config):
    import torch
    from dataset.abstract_dataset import FFpp_pool
    required = [Path(config['clip_path']) / 'config.json']
    model_dirs = [Path(config['clip_path'])]
    probe_type = (config.get('probe') or {}).get('type', 'sd15_vae')
    if probe_type in ['sd15_vae', 'sdvae_alt']:
        vae_path = config['probe']['vae_path'] if probe_type == 'sdvae_alt' else config['vae_path']
        model_dirs.append(Path(vae_path))
        required.append(Path(vae_path) / 'config.json')
    datasets = sorted(set(config['train_dataset'] + config['test_dataset']))
    for name in datasets:
        required.append(Path(config['dataset_json_folder']) / (name + '.json'))
        if config['lmdb']:
            lmdb_name = 'FaceForensics++' if name in FFpp_pool else name
            required.append(Path(config['lmdb_dir']) / (lmdb_name + '_lmdb') / 'data.mdb')
        else:
            required.append(Path(config['rgb_dir']))
    for directory in model_dirs:
        if not any(directory.glob('*.bin')) and not any(directory.glob('*.safetensors')):
            raise FileNotFoundError(f'Missing local model weights in {directory}')
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing prepared data/weights:\n' + '\n'.join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install the documented environment and a compatible driver.')
    from detectors import DETECTOR
    if config['model_name'] not in DETECTOR.data:
        raise ValueError(f"Unknown model {config['model_name']}")
    print(json.dumps(dict(preflight='PASS', python=sys.version.split()[0], torch=torch.__version__,
                         cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                         data_root=config['data_root'], clip_path=config['clip_path'],
                         model_directories=[str(path) for path in model_dirs], early_stopping=config.get('early_stopping'),
                         train_batch=config['train_batchSize'], test_batch=config['test_batchSize']), indent=2), flush=True)


def smoke(config):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from detectors import DETECTOR
    from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
    from experiment_metadata import write_run_metadata
    random.seed(config['manualSeed'])
    np.random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    torch.cuda.manual_seed_all(config['manualSeed'])
    config = copy.deepcopy(config)
    config['save_tsne'] = False
    config['analysis'] = {}
    parent = resolve_output_path('logs/portability-smoke')
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=parent))
    config['output_dir'] = str(output)

    def loader(mode):
        cfg = copy.deepcopy(config)
        if mode == 'test':
            cfg['test_dataset'] = config['test_dataset'][0]
        dataset = DeepfakeAbstractBaseDataset(config=cfg, mode=mode)
        # One real and one fake sample, retaining the actual dataset/transforms.
        indices = [next(i for i, label in enumerate(dataset.label_list) if (label != 0) == fake)
                   for fake in [False, True]]
        subset = Subset(dataset, indices)
        return DataLoader(subset, batch_size=2, num_workers=0, collate_fn=dataset.collate_fn), dataset, indices

    train_loader, train_set, _ = loader('train')
    batch = next(iter(train_loader))
    for key in ['image', 'label']:
        batch[key] = batch[key].cuda()
    batch['label'] = (batch['label'] != 0).long()
    model = DETECTOR[config['model_name']](config).cuda()
    model.train()
    adam = config['optimizer']['adam']
    optimizer = torch.optim.Adam(model.parameters(), lr=adam['lr'], betas=(adam['beta1'], adam['beta2']),
                                 eps=adam['eps'], weight_decay=adam['weight_decay'], amsgrad=adam['amsgrad'])
    optimizer.zero_grad()
    predictions = model(batch)
    logits_shape = list(predictions['cls'].shape)
    if logits_shape != [2, 2]:
        raise RuntimeError(f'Unexpected classifier shape: {logits_shape}')
    loss = model.get_losses(batch, predictions)['overall']
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite smoke loss')
    loss.backward()
    optimizer.step()
    checkpoint = output / 'smoke.pth'
    torch.save(model.state_dict(), checkpoint)
    restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(restored, strict=True)
    del restored, optimizer, predictions
    model.eval()
    test_loader, test_set, indices = loader('test')
    saved_args = sys.argv
    try:
        sys.argv = ['test.py']
        from test import test_one_dataset, save_metrics_report
    finally:
        sys.argv = saved_args
    pred, labels, _, _ = test_one_dataset(model, test_loader)
    from metrics.utils import get_test_metrics
    metrics = get_test_metrics(pred, labels, [test_set.image_list[i] for i in indices])
    save_metrics_report({config['test_dataset'][0]: metrics}, str(output), config, str(checkpoint))
    write_run_metadata(output, config, dict(smoke=True, physical_batch=2, train_loss=loss.item(),
                                          checkpoint=str(checkpoint), not_a_benchmark=True))
    print(json.dumps(dict(smoke='PASS', loss=loss.item(), checkpoint_bytes=checkpoint.stat().st_size,
                          logits_shape=logits_shape, output_dir=str(output), samples_tested=len(labels)), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='training/config/detector/prd_probe_ablation.yaml')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        preflight(cfg)
        if args.smoke:
            smoke(cfg)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        sys.exit(1)
