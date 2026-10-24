"""Validation for `schemas/geometry-3d.schema.json` documents."""

import json
import os
from functools import lru_cache

from utils import paths

SCHEMA_PATH = os.path.join(paths.REPO_ROOT, "schemas", "geometry-3d.schema.json")
SCHEMA_VERSION = "1.0"


@lru_cache(maxsize=1)
def _validator():
    import jsonschema
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    cls = jsonschema.validators.validator_for(schema)
    return cls(schema, format_checker=cls.FORMAT_CHECKER)


def schema_problems(doc, where=""):
    out = []
    for e in sorted(_validator().iter_errors(doc), key=lambda e: list(e.absolute_path)):
        at = "/".join(str(p) for p in e.absolute_path) or "<root>"
        out.append(f"{where}{at}: {e.message}")
    return out
