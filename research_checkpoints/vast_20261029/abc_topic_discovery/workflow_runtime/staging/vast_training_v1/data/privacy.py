"""Validate private export parents before creating human-data outputs."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


def require_private_parent(path: Path) -> None:
    parent = Path(path).resolve(strict=True)
    if not parent.is_dir():
        raise PermissionError("export parent must be an existing directory")
    if os.name != "nt":
        info = parent.stat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("export parent must be user-owned and mode 0700")
        return
    # 0700 is not a Windows DACL. Require every allow ACE to name the current
    # Windows SID and require full control, including inheritable child access.
    script = r'''
$ErrorActionPreference = 'Stop'
$sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$acl = (New-Object System.IO.DirectoryInfo($env:ABC_PRIVATE_PARENT)).GetAccessControl()
$rules = $acl.GetAccessRules($true, $true, [System.Security.Principal.SecurityIdentifier])
$allowed = @($rules | Where-Object { $_.AccessControlType -eq 'Allow' })
if ($allowed.Count -eq 0) { exit 3 }
if (@($allowed | Where-Object { $_.IdentityReference.Value -ne $sid }).Count -ne 0) { exit 4 }
$full = @($allowed | Where-Object {
  ($_.FileSystemRights -band [System.Security.AccessControl.FileSystemRights]::FullControl) -eq [System.Security.AccessControl.FileSystemRights]::FullControl -and
  ($_.InheritanceFlags -band [System.Security.AccessControl.InheritanceFlags]::ContainerInherit) -ne 0 -and
  ($_.InheritanceFlags -band [System.Security.AccessControl.InheritanceFlags]::ObjectInherit) -ne 0
})
if ($full.Count -eq 0) { exit 5 }
'''
    env = dict(os.environ, ABC_PRIVATE_PARENT=str(parent))
    try:
        check = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                               env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PermissionError("could not validate Windows export-parent DACL") from exc
    if check.returncode:
        raise PermissionError("Windows export parent must grant inheritable access only to the current user")
