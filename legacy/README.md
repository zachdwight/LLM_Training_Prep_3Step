# Legacy Scripts (v0.x)

This directory contains the original monolithic scripts from before the v1.0 refactoring.

## Files

- `pdfs_to_ideas_with_llm.py` — Step 1 (original version)
- `format_json.py` — Step 2 (original version)
- `finalize_tuning_data.py` — Step 3 (original version)

## Status

These scripts are **deprecated** but remain functional for backward compatibility.

### Migration to v1.0

**New users should use the refactored package** located in `llm_training_prep/` and `scripts/` directories instead.

**Reasons to migrate:**
- Modular, reusable classes
- Comprehensive metrics and reporting
- Centralized configuration management
- Proper logging
- Installable via `pip install -e .`

### Using the Legacy Scripts

If you prefer the old scripts, they still work unchanged:

```bash
python legacy/pdfs_to_ideas_with_llm.py --pdf-dir ./pdfs
python legacy/format_json.py --input-dir ./output
python legacy/finalize_tuning_data.py --input formatted.jsonl
```

See [../README.md](../README.md) for details on the new package structure and v1.0 improvements.
