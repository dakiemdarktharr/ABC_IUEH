"""Pinned official BOXRR-23 user TARs. Never extract an archive to the filesystem."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, unquote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPO = "cschell/boxrr-23"
REVISION = "af25be3ef76b176fbee0a094e82d97a611f9c950"
TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/users"
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
TAR_PATH = re.compile(rf"users/([0-9a-f]+)/({UUID})\.tar")
MAX_TAR = 4 * 1024**3


def authorization_reference(path):
    """Bind intake to the operator's explicit access authorization, not an IRB claim."""
    if path is None or not path.is_file():
        raise PermissionError("Supply the local operator authorization record before intake")
    record = json.loads(path.read_bytes())
    if record.get("dataset") != "BOXRR-23" or record.get("basis") != "explicit_user_instruction":
        raise PermissionError("The authorization record must cover BOXRR-23")
    for name in ("instruction", "recorded_on", "scope"):
        if not isinstance(record.get(name), str) or not record[name].strip():
            raise PermissionError(f"Missing authorization field: {name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class UserArchive:
    user: str
    path: str
    size: int
    sha256: str

    def __post_init__(self):
        match = TAR_PATH.fullmatch(self.path)
        if (not match or match[2] != self.user or type(self.size) is not int or
                not 0 < self.size <= MAX_TAR or not re.fullmatch("[0-9a-f]{64}", self.sha256)):
            raise ValueError("Invalid official user archive descriptor")


def http_session():
    session = requests.Session()
    retries = Retry(total=4, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _pages(session, url):
    endpoint = urlsplit(url)
    while url:
        response = session.get(url, timeout=(10, 60))
        response.raise_for_status()
        yield from response.json()
        url = response.links.get("next", {}).get("url")
        if url:
            candidate = urlsplit(url)
            if (candidate.scheme != endpoint.scheme or candidate.netloc != endpoint.netloc or
                    unquote(candidate.path) != unquote(endpoint.path) or candidate.fragment):
                raise ValueError("Unexpected pagination URL outside the pinned tree")


def official_archives(session):
    buckets = sorted(row["path"] for row in _pages(session, TREE)
                     if row["type"] == "directory" and re.fullmatch(r"users/[0-9a-f]+", row["path"]))
    # A seeded order fixes intake prospectively, independent of observed motion or errors.
    buckets.sort(key=lambda x: hashlib.sha256(("20261026:" + x).encode()).digest())
    for bucket in buckets:
        for row in _pages(session, TREE + "/" + bucket.split("/")[1]):
            match = TAR_PATH.fullmatch(row["path"])
            if row["type"] != "file" or not match:
                continue
            digest = row.get("lfs", {}).get("oid", "")
            if (not re.fullmatch("[0-9a-f]{64}", digest) or
                    row["size"] != row["lfs"]["size"] or row["size"] <= 0):
                raise ValueError("Unpinned or inconsistent official TAR metadata")
            if row["size"] <= MAX_TAR:
                yield UserArchive(match[2], row["path"], row["size"], digest)


def download_archive(session, entry, destination):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{entry.path}?download=true"
    with session.get(url, stream=True, timeout=(15, 120)) as response, destination.open("xb") as output:
        response.raise_for_status()
        for chunk in response.iter_content(1024**2):
            if output.tell() + len(chunk) > entry.size:
                raise ValueError("TAR exceeds its published length")
            output.write(chunk)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
