"""Shared paths, checks and process execution for CHARM commands."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent
def path_setting(name, default):
    return Path(os.environ.get(name, str(default))).expanduser()


DEFAULT_DATA = path_setting('CHARM_DATA_ROOT', REPO / 'data' / 'MAGB')
DEFAULT_MODEL = path_setting('CHARM_MODEL', REPO / 'models' / 'base_model')
DEFAULT_CLIP = path_setting('CHARM_CLIP_MODEL', REPO / 'models' / 'clip-vit-large-patch14-336')
DEFAULT_GRAPH = path_setting('CHARM_SYNTHETIC_ROOT', DEFAULT_DATA / 'realnode_kmeans_product3_g24_topm3_semantic_cliptext_bridge')
DEFAULT_INIT = REPO / 'charm_pipeline' / 'weights' / 'initial' / 'mm_projector.bin'


def fail_missing(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError('Missing required files:\n  ' + '\n  '.join(missing))


def validate_domains(sources, target):
    if len(sources) != 2 or len(set(sources)) != 2:
        raise ValueError('Provide exactly two distinct source domains.')
    if target in sources:
        raise ValueError('The target domain must differ from both source domains.')
    for domain in [*sources, target]:
        if not domain or Path(domain).name != domain or domain in ('.', '..'):
            raise ValueError(f'Invalid domain name: {domain!r}')


def require_fresh(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Refusing to overwrite non-empty output: {path}')


def run(command, *, dry_run=False, gpu=None, env_extra=None):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env.setdefault(key, '2')
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if env_extra:
        env.update(env_extra)
    print('+', shlex.join([str(part) for part in command]), flush=True)
    if not dry_run:
        subprocess.run([str(part) for part in command], check=True, cwd=REPO, env=env)


def python(*args):
    return [sys.executable, *map(str, args)]

