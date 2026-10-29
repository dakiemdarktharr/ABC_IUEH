"""Explicit entry point; construction alone does not read quaternion values."""
import argparse
import json
from pathlib import Path
from .evaluator import evaluate_fold
from .source import FixedValidationSegments
from .fit_identity import verify_fit_identities


def main():
    parser = argparse.ArgumentParser(description='Evaluate one held-out train fold after all 18 fits')
    parser.add_argument('--fold', required=True, type=int, choices=(0, 1, 2))
    parser.add_argument('--checkpoints', required=True, type=Path,
                        help='private JSON list of 18 fold_id, arm_id, training_seed, path, sha256 records')
    parser.add_argument('--output', required=True, type=Path, help='new private report path')
    parser.add_argument('--fit-manifests', required=True, nargs=3, type=Path, help='signed fit manifest paths for folds 0, 1, 2')
    parser.add_argument('--fit-public-keys', required=True, nargs=3, type=Path, help='independently pinned raw Ed25519 public keys for folds 0, 1, 2 in order')
    parser.add_argument('--fit-key-ids', required=True, nargs=3, help='independently pinned key IDs for folds 0, 1, 2 in order')
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    records = json.loads(args.checkpoints.read_text(encoding='utf-8'))
    if type(records) is not list: parser.error('checkpoint input must be a list')
    checkpoints = {}
    for row in records:
        if type(row) is not dict or set(row) != {'fold_id','arm_id','training_seed','path','sha256'}:
            parser.error('unexpected checkpoint record schema')
        key = (row['fold_id'], row['arm_id'], row['training_seed'])
        if key in checkpoints: parser.error('duplicate checkpoint cell')
        checkpoints[key] = (row['path'], row['sha256'])
    fit_identities = verify_fit_identities(dict(enumerate(args.fit_manifests)),
        public_keys={fold: path.read_bytes() for fold, path in enumerate(args.fit_public_keys)},
        key_ids=dict(enumerate(args.fit_key_ids)))
    segments = FixedValidationSegments(args.fold)
    evaluate_fold(fold_id=args.fold, checkpoints=checkpoints, fit_identities=fit_identities, segments=segments,
                  output_path=args.output, device=args.device, batch_size=args.batch_size)
    print(f'Private fold-{args.fold} evaluation report written to {args.output}')

if __name__ == '__main__': main()
