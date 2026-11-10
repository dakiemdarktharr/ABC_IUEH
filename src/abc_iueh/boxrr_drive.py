"""Native Python Drive upload/download with SHA-256 readback and a last-written index."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from filelock import FileLock
import google.auth
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from .boxrr_pilot import GIB, verify
from .boxrr_source import file_sha256


def client():
    token = os.environ.get("GOOGLE_DRIVE_ACCESS_TOKEN")
    credentials = Credentials(token) if token else google.auth.default(
        scopes=["https://www.googleapis.com/auth/drive.file"])[0]
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def private_folder(api, folder):
    info = api.files().get(fileId=folder, fields="id,mimeType,trashed,ownedByMe,permissions(type,role)").execute(num_retries=6)
    if (info["mimeType"] != "application/vnd.google-apps.folder" or info["trashed"] or
            not info["ownedByMe"] or info["permissions"] != [{"type": "user", "role": "owner"}]):
        raise ValueError("Destination must be an owner-only private Drive folder")


def get_file(api, file_id, destination):
    with destination.open("xb") as stream:
        media = MediaIoBaseDownload(stream, api.files().get_media(fileId=file_id), chunksize=8 * 1024**2)
        done = False
        while not done:
            _, done = media.next_chunk(num_retries=6)


def put_file(api, folder, path, state_path, *, remote_name=None):
    """Reuse the preallocated ID across retries. Commit only after byte readback."""
    path, state_path = Path(path), Path(state_path)
    remote_name = remote_name or path.name
    private_folder(api, folder)
    digest, size = file_sha256(path), path.stat().st_size
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(state_path) + ".lock", timeout=0):
        if state_path.exists():
            state = json.loads(state_path.read_bytes())
            if (state["folder"], state["sha256"], state["bytes"]) != (folder, digest, size):
                raise ValueError("Upload state belongs to different immutable bytes")
        else:
            file_id = api.files().generateIds(count=1, space="drive").execute(num_retries=6)["ids"][0]
            state = {"id": file_id, "folder": folder, "sha256": digest, "bytes": size, "complete": False}
            state_path.write_text(json.dumps(state), encoding="utf-8")
        file_id = state["id"]
        from googleapiclient.errors import HttpError
        fields = "id,name,size,sha256Checksum,parents,trashed,ownedByMe"
        try:
            meta = api.files().get(fileId=file_id, fields=fields).execute(num_retries=6)
        except HttpError as exc:
            if exc.resp.status != 404:
                raise
            meta = None
        matching = (meta and not meta["trashed"] and meta["parents"] == [folder] and
                    int(meta.get("size", -1)) == size and meta.get("sha256Checksum") == digest)
        if not matching:
            if state["complete"]:
                raise ValueError("Previously verified Drive file changed or disappeared")
            if meta and (meta["trashed"] or meta["parents"] != [folder] or
                         not meta["ownedByMe"] or meta["name"] != remote_name):
                raise ValueError("Incomplete upload has unexpected ownership or identity")
            media = MediaFileUpload(str(path), resumable=True, chunksize=8 * 1024**2)
            if meta:
                request = api.files().update(fileId=file_id, media_body=media, fields="id")
            else:
                request = api.files().create(body={"id": file_id, "name": remote_name, "parents": [folder]},
                                             media_body=media, fields="id")
            result = None
            while result is None:
                _, result = request.next_chunk(num_retries=6)
            meta = api.files().get(fileId=file_id, fields=fields).execute(num_retries=6)
        if (meta.get("trashed") or meta.get("parents") != [folder] or int(meta.get("size", -1)) != size or
                meta.get("sha256Checksum") != digest):
            raise ValueError("Remote object differs from pinned upload")
        if state["complete"]:
            return {"id": file_id, "sha256": digest, "bytes": size}
        with tempfile.TemporaryDirectory(prefix=".readback-", dir=state_path.parent) as tmp:
            check = Path(tmp) / "bytes"
            get_file(api, file_id, check)
            if check.stat().st_size != size or file_sha256(check) != digest:
                raise ValueError("Drive readback differs")
        state["complete"] = True
        updated = state_path.with_suffix(".part")
        updated.write_text(json.dumps(state, indent=2), encoding="utf-8")
        updated.replace(state_path)
        return {"id": file_id, "sha256": digest, "bytes": size}


def upload(api, folder, corpus, toolkit, state_dir):
    doc = verify(corpus)
    if doc["source"] != "BOXRR-23" or not doc["authorization_sha256"] or not 6 * GIB <= doc["bytes"] <= 8 * GIB:
        raise ValueError("Production upload needs an authorized, complete 6–8 GiB BOXRR pilot")
    items = [(row["path"], Path(corpus) / row["path"]) for row in doc["shards"]]
    items += [("manifest.json", Path(corpus) / "manifest.json"), ("artifacts/toolkit.zip", Path(toolkit))]
    index = {"schema": "abc-boxrr-drive-v1", "files": []}
    state_dir = Path(state_dir)
    with FileLock(str(state_dir) + ".lock", timeout=0):
        for name, path in items:
            state_name = hashlib.sha256(name.encode()).hexdigest() + ".json"
            remote_name = path.parent.name + ".h5" if path.name == "data.h5" else path.name
            result = put_file(api, folder, path, state_dir / state_name, remote_name=remote_name)
            index["files"].append({"path": name, **result})
        index_path = state_dir / "drive-index.json"
        content = json.dumps(index, sort_keys=True, indent=2).encode()
        if index_path.exists() and index_path.read_bytes() != content:
            raise ValueError("Committed release index differs")
        index_path.write_bytes(content)
        return put_file(api, folder, index_path, state_dir / "index-upload.json")


def fetch(api, index_id, index_sha256, destination):
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError("Restore requires a fresh destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.with_name(destination.name + ".partial")
    with FileLock(str(destination) + ".fetch.lock", timeout=0):
        work.mkdir(mode=0o700, exist_ok=True)
        index_file = work / "drive-index.json"
        if not index_file.exists():
            part = work / "index.part"
            part.unlink(missing_ok=True)
            get_file(api, index_id, part)
            if file_sha256(part) != index_sha256:
                raise ValueError("Index differs from the independently retained SHA-256")
            part.rename(index_file)
        if file_sha256(index_file) != index_sha256:
            raise ValueError("Index differs from the independently retained SHA-256")
        index = json.loads(index_file.read_bytes())
        if index["schema"] != "abc-boxrr-drive-v1":
            raise ValueError("Unknown Drive release schema")
        names = [row["path"] for row in index["files"]]
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("Incomplete or duplicated release paths")
        size = sum(row["bytes"] for row in index["files"])
        if not 0 < size <= 9 * GIB:
            raise OSError("Restore exceeds release bounds")
        tree = work / "corpus"
        tree.mkdir(mode=0o700, exist_ok=True)
        for row in index["files"]:
            path = PurePosixPath(row["path"])
            if path.is_absolute() or ".." in path.parts or "\\" in row["path"] or ":" in row["path"]:
                raise ValueError("Unsafe release path")
            target = tree / path
            if not target.resolve().is_relative_to(tree.resolve()) or target.is_symlink():
                raise ValueError("Unsafe cached release path")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                if shutil.disk_usage(work).free < row["bytes"] + GIB:
                    raise OSError("Insufficient disk for next shard")
                part = target.with_name(target.name + ".part")
                part.unlink(missing_ok=True)
                get_file(api, row["id"], part)
                if part.stat().st_size != row["bytes"] or file_sha256(part) != row["sha256"]:
                    raise ValueError("Downloaded file differs")
                part.rename(target)
            if target.stat().st_size != row["bytes"] or file_sha256(target) != row["sha256"]:
                raise ValueError("Downloaded file differs")
        verify(tree)
        tree.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    send = sub.add_parser("upload")
    for name in ("folder", "corpus", "toolkit", "state"):
        send.add_argument("--" + name, required=True)
    get = sub.add_parser("fetch")
    for name in ("index-id", "index-sha256", "destination"):
        get.add_argument("--" + name, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    api = client()
    if args.command == "upload":
        print(json.dumps(upload(api, args.folder, args.corpus, args.toolkit, args.state)))
    else:
        fetch(api, args.index_id, args.index_sha256, args.destination)


if __name__ == "__main__":
    main()
