---
description: Python code style conventions
globs:
  - "**/*.py"
---

# Python Style Guide

## Type Hints

- All public functions must have type hints
- Use `Optional[]` for nullable types
- Use `list[]`, `dict[]` not `List[]`, `Dict[]` (Python 3.9+)

## Documentation

- Classes: docstring explaining purpose
- Public methods: docstring with Args/Returns
- Complex logic: inline comments

## Logging

```python
# Use logging module, never print()
import logging
logger = logging.getLogger(__name__)

logger.info("Processing %d items", count)  # NOT f-strings in logs
```

## Constants

- No magic numbers in code
- Define in `config/constants.py` or settings
- Use SCREAMING_SNAKE_CASE

## Error Handling

```python
# Specific exceptions, never bare except
try:
    result = risky_operation()
except ValueError as e:
    logger.warning("Invalid value: %s", e)
    return default
```
