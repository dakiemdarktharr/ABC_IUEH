# Metadata-integrity hardening — 2026-10-28

Semgrep Guardian reported a SHA-1 call in DVC pointer discovery. That call
reconstructed an upstream Git blob identifier; replacing the function with
SHA-256 while still comparing against the original Git ID would be incorrect.
[Git's hash-format documentation](https://git-scm.com/docs/hash-function-transition)
explains why SHA-1 and SHA-256 object identifiers are different formats.

The implementation now removes the SHA-1 computation entirely. It computes
SHA-256 for each raw DVC pointer, then hashes a canonical, path-sorted list of
`{"path": source_path, "sha256": pointer_sha256}` entries using UTF-8 JSON with
sorted keys and compact separators. The resulting digest must equal the
reviewed snapshot pinned alongside the source commit:

```text
a3a8cdcdf2a70ce6a28b0a09c7c688568bb457d7e2f8faae1dd3dc233932ecd9
```

The full set of 70 metadata objects at the reviewed source revision was fetched
again and matched this digest. Changes to pointer content, comments, paths,
membership or duplicate entries fail closed. Upstream Git blob IDs remain
provenance labels; they are not used as local cryptographic integrity checks.
Changing the reviewed dataset snapshot requires explicitly reviewing and updating
both pins, rather than silently trusting a new inventory.

Recording size/MD5 compatibility checks and local SHA-256 receipts remain in
place. This change does not claim that an upstream MD5 is a cryptographic
signature. No recording download, HDF5 regeneration, split change, model fitting
or forecast evaluation was required. The original preparation receipt records
its original 49-test run; the updated suite has 55 tests.

Six new regression cases cover the accepted fixture and five rejected mutations:
content change, comment-only byte change, renamed path, missing entry and duplicate
entry. CI also runs the
[reported Semgrep rule](https://github.com/semgrep/semgrep-rules/blob/16b9e6d45f10649f6a531ec4838ed9924372c3bc/python/lang/security/insecure-hash-algorithms.yaml)
using Semgrep 1.182.1, with metrics disabled, against source and tests. No rule
suppression or scanner exclusion is introduced.
