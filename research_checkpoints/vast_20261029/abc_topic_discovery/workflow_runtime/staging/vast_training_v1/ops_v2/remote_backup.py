"""Immutable-intent off-host checkpoint copy with remote byte readback.

A single writer must own the remote prefix. Rclone's backend-independent
copy flags do not provide atomic create-if-absent across concurrent writers.
No network access occurs until an explicit backup call is made.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from . import checkpoint_store as store


class RemoteBackupError(ValueError):
    pass


def _remote_prefix(value):
    if type(value) is not str or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*:[A-Za-z0-9_./-]+', value):
        raise RemoteBackupError('remote must be a configured rclone name and nonempty safe directory')
    path = value.split(':', 1)[1]
    if path.startswith('/') or any(p in ('', '.', '..') for p in path.rstrip('/').split('/')):
        raise RemoteBackupError('remote directory must have safe relative components')
    return value.rstrip('/')


class RcloneTransport:
    """Copy existing-safe and read back bytes; configured remote and creds stay external."""
    def __init__(self, executable='rclone', timeout_seconds=1800):
        if not isinstance(executable, str) or not executable or type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise RemoteBackupError('invalid rclone executable or timeout')
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def check(self, remote_prefix):
        prefix=_remote_prefix(remote_prefix)
        remote=prefix.split(':',1)[0] + ':'
        try:
            subprocess.run([self.executable,'lsf',remote,'--max-depth','1'],check=True,
                           timeout=self.timeout_seconds,stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except (OSError,subprocess.SubprocessError):
            raise RemoteBackupError('rclone preflight failed or timed out') from None

    def put_if_absent(self, local_path, remote_path):
        # Existing visible objects are skipped and subsequently read back.
        try:
            subprocess.run([self.executable, 'copyto', str(local_path), remote_path,
                            '--ignore-existing', '--immutable'], check=True,
                           timeout=self.timeout_seconds, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            raise RemoteBackupError('rclone upload failed or timed out') from None

    def download_to(self, remote_path, destination):
        try:
            with Path(destination).open('xb') as output:
                subprocess.run([self.executable, 'cat', remote_path], check=True,
                               timeout=self.timeout_seconds, stdin=subprocess.DEVNULL,
                               stdout=output, stderr=subprocess.DEVNULL)
                output.flush(); os.fsync(output.fileno())
        except (OSError, subprocess.SubprocessError):
            raise RemoteBackupError('rclone download failed or timed out') from None

    def read_digest(self, remote_path):
        with tempfile.TemporaryDirectory(prefix='abc-remote-read-') as tmp:
            local=Path(tmp)/'object'
            self.download_to(remote_path, local)
            return store.sha256_file(local)


def _snapshot(source, target, expected):
    with store._open_nofollow(source) as (src, before), target.open('xb') as out:
        shutil.copyfileobj(src, out, length=1024*1024)
        after=store._signature(os.fstat(src.fileno()), source)
        out.flush(); os.fsync(out.fileno())
    store._verify_path_still_same(source, before)
    if before != after or store.sha256_file(target) != expected or store.sha256_file(source) != expected:
        raise RemoteBackupError('source changed during checkpoint snapshot')


def backup_checkpoint(checkpoint, metadata, remote_prefix, *, transport=None):
    """Copy one finalized trainer checkpoint and its sidecar; commit manifest last.

    Returns remote manifest path/hash only after all three remote objects are
    downloaded and SHA-256/size checked. Requires an exclusive remote writer.
    """
    prefix=_remote_prefix(remote_prefix)
    checkpoint, metadata = Path(checkpoint), Path(metadata)
    if checkpoint.is_symlink() or metadata.is_symlink() or checkpoint.parent.resolve(strict=True) != metadata.parent.resolve(strict=True):
        raise RemoteBackupError('checkpoint and metadata must be nonsymlink siblings')
    if metadata.name != f'{checkpoint.stem}.meta.json':
        raise RemoteBackupError('checkpoint metadata filename is not its sidecar')
    meta, meta_hash, meta_size = store._load_json_with_digest(metadata)
    store.validate_meta(meta)
    payload_hash, payload_size = store.sha256_file(checkpoint)
    match = re.fullmatch(r'update-(\d+)-([0-9a-f]{64})\.pt', checkpoint.name)
    if (match is None or match.group(2) != payload_hash or int(match.group(1)) != meta['global_step'] or
            (meta['payload_sha256'], meta['payload_size_bytes']) != (payload_hash, payload_size)):
        raise RemoteBackupError('checkpoint name, metadata and actual bytes must agree')
    payload_name=f'checkpoint-{payload_hash}.pt'
    meta_name=f'checkpoint-{payload_hash}.meta.json'
    manifest_name=f'checkpoint-{payload_hash}.manifest.json'
    manifest={'schema':'abc-checkpoint-copy-v1', 'payload_file':payload_name,
              'payload_sha256':payload_hash, 'payload_size_bytes':payload_size,
              'metadata_file':meta_name, 'metadata_sha256':meta_hash, 'metadata_size_bytes':meta_size}
    encoded=(json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode('utf-8')
    manifest_digest=(hashlib.sha256(encoded).hexdigest(),len(encoded))
    transport=transport if transport is not None else RcloneTransport()
    with tempfile.TemporaryDirectory(prefix='abc-remote-backup-') as tmp:
        root=Path(tmp)
        payload_copy=root/payload_name; metadata_copy=root/meta_name; manifest_copy=root/manifest_name
        _snapshot(checkpoint, payload_copy, (payload_hash,payload_size))
        _snapshot(metadata, metadata_copy, (meta_hash,meta_size))
        manifest_copy.write_bytes(encoded)
        for local, expected in ((payload_copy,(payload_hash,payload_size)),
                                (metadata_copy,(meta_hash,meta_size)),
                                (manifest_copy,manifest_digest)):
            remote=f'{prefix}/{local.name}'
            transport.put_if_absent(local, remote)
            if transport.read_digest(remote) != expected:
                raise RemoteBackupError(f'remote object differs from expected bytes: {local.name}')
    return {'manifest_remote':f'{prefix}/{manifest_name}', 'manifest_sha256':manifest_digest[0],
            'payload_sha256':payload_hash, 'metadata_sha256':meta_hash}


def _pin(value):
    if type(value) is not str or not store.SHA_RE.fullmatch(value):
        raise RemoteBackupError('independent lowercase SHA-256 pin required')
    return value


def verify_remote_copy(remote_prefix, local_manifest, manifest_sha256, *, transport=None):
    """Re-read a remote manifest and both objects against a locally verified copy."""
    prefix=_remote_prefix(remote_prefix); manifest_sha256=_pin(manifest_sha256)
    local_manifest=Path(local_manifest)
    manifest=store.verify_copy(local_manifest)
    name=f"checkpoint-{manifest['payload_sha256']}.manifest.json"
    if local_manifest.name != name or store.sha256_file(local_manifest)[0] != manifest_sha256:
        raise RemoteBackupError('local manifest does not match the independent pin')
    if (manifest['payload_file'] != f"checkpoint-{manifest['payload_sha256']}.pt" or
        manifest['metadata_file'] != f"checkpoint-{manifest['payload_sha256']}.meta.json"):
        raise RemoteBackupError('unexpected remote object names')
    transport=transport if transport is not None else RcloneTransport()
    for filename, expected in ((name,store.sha256_file(local_manifest)),
             (manifest['payload_file'],(manifest['payload_sha256'],manifest['payload_size_bytes'])),
             (manifest['metadata_file'],(manifest['metadata_sha256'],manifest['metadata_size_bytes']))):
        if transport.read_digest(f'{prefix}/{filename}') != expected:
            raise RemoteBackupError('remote verification failed')
    return manifest


def restore_checkpoint(remote_prefix, destination, *, manifest_sha256, payload_sha256,
                       expected_path, transport=None):
    """Restore to a new local directory and verify independent pins and run metadata."""
    prefix=_remote_prefix(remote_prefix); manifest_sha256=_pin(manifest_sha256); payload_sha256=_pin(payload_sha256)
    destination=Path(destination)
    if destination.exists() or destination.is_symlink(): raise RemoteBackupError('restore destination must be new')
    destination.mkdir(mode=0o700,parents=False)
    transport=transport if transport is not None else RcloneTransport()
    name=f'checkpoint-{payload_sha256}.manifest.json'
    staging=destination/'.manifest.pending'
    transport.download_to(f'{prefix}/{name}',staging)
    digest,size=store.sha256_file(staging)
    if digest != manifest_sha256 or size > 16384: raise RemoteBackupError('restored manifest differs from independent pin')
    manifest=store._load_json(staging)
    payload=f'checkpoint-{payload_sha256}.pt'; meta=f'checkpoint-{payload_sha256}.meta.json'
    if (manifest.get('schema') != 'abc-checkpoint-copy-v1' or manifest.get('payload_sha256') != payload_sha256 or
        manifest.get('payload_file') != payload or manifest.get('metadata_file') != meta):
        raise RemoteBackupError('restored manifest is outside the pinned checkpoint')
    for filename in (payload,meta): transport.download_to(f'{prefix}/{filename}',destination/filename)
    expected=store._load_json(Path(expected_path))
    if type(expected.get('expected_global_step')) is not int or expected['expected_global_step'] < 1:
        raise RemoteBackupError('restore requires an independent positive expected step')
    compared=store.compare_resume(staging,Path(expected_path))
    if compared['payload_sha256'] != payload_sha256: raise RemoteBackupError('restored payload differs from pin')
    final=destination/name
    # Publish only after validating payload, metadata and expected run pins.
    os.link(staging,final); staging.unlink()
    return {'manifest_path':str(final), 'manifest_sha256':manifest_sha256,
            'payload_sha256':payload_sha256, 'resume_path':compared['trainer_resume_argument']}


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description='Back up or restore using a configured rclone remote')
    commands=parser.add_subparsers(dest='operation',required=True)
    backup=commands.add_parser('backup')
    backup.add_argument('checkpoint', type=Path); backup.add_argument('metadata', type=Path)
    backup.add_argument('remote_prefix')
    restore=commands.add_parser('restore')
    restore.add_argument('remote_prefix'); restore.add_argument('destination',type=Path)
    restore.add_argument('--manifest-sha256',required=True); restore.add_argument('--payload-sha256',required=True)
    restore.add_argument('--expected',type=Path,required=True)
    for command in (backup,restore):
        command.add_argument('--rclone',default='rclone')
        command.add_argument('--timeout-seconds',type=int,default=1800)
    args=parser.parse_args(argv)
    transport=RcloneTransport(args.rclone,args.timeout_seconds)
    if args.operation == 'backup':
        result=backup_checkpoint(args.checkpoint,args.metadata,args.remote_prefix,transport=transport)
    else:
        result=restore_checkpoint(args.remote_prefix,args.destination,manifest_sha256=args.manifest_sha256,
                                  payload_sha256=args.payload_sha256,expected_path=args.expected,transport=transport)
    print(json.dumps(result,sort_keys=True))


if __name__ == '__main__':
    main()
