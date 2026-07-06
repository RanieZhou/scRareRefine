# Cache Provenance Audit

- Cached run directories: 76
- Required embedding files complete: 76/76
- Manifest present: 76/76
- Split hash matches cached cell IDs: 76/76
- Current git SHA: 91746c9
- Git status counts: {'different_commit': 64, 'legacy_unknown': 12}

Interpretation:
- `split_hash_ok=False` is a hard provenance problem: cached cell IDs do not match the manifest.
- `git_status=legacy_unknown` means the cache predates git-sha recording; use `--force` for strict reruns.
- `git_status=different_commit` means embeddings were generated on another commit; inspect before publication reruns.

CSV: `results\provenance\cache_audit.csv`
