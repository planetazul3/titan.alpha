# Model Migration Guide

## Semantic Versioning

Model versions follow semantic versioning: `MAJOR.MINOR.PATCH`

| Version Change | Meaning | Action Required |
|----------------|---------|-----------------|
| Major (1.x → 2.x) | Breaking changes | Retrain model |
| Minor (1.0 → 1.1) | New features, backward compatible | Load safely |
| Patch (1.0.0 → 1.0.1) | Bug fixes only | Load safely |

## Version Location

Current version controlled in `training/checkpoint_manifest.py`:

```python
CURRENT_MODEL_VERSION = "1.0.0"
```

## Compatibility Checking

```python
from training.checkpoint_manifest import check_model_compatibility

# Before loading a checkpoint
if check_model_compatibility("checkpoints/model.pt"):
    # Safe to load
else:
    # Major version mismatch - requires retraining
```

## Migration Steps

### Minor/Patch Updates
1. Increment version in `checkpoint_manifest.py`
2. Deploy updated code
3. Load existing checkpoints (backward compatible)

### Major Updates
1. Increment major version
2. Train new model with new architecture
3. Save checkpoint with new version
4. Test thoroughly before production

## Shadow Trade Tracking

All shadow trades include `model_version` in metadata, enabling:
- Performance comparison between model versions
- A/B testing of model improvements
- Rollback verification
