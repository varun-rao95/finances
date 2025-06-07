import json, os, importlib.util, inspect
from pathlib import Path
from typing import Dict, List

# ----- internals -----------------------------------------------------------

def _load_module(file_path: str):
    spec = importlib.util.spec_from_file_location(Path(file_path).stem, file_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # type: ignore
    return mod

def parse_docstring(file_path: str) -> Dict:
    """Read top-of-file docstring and return its JSON as a dict."""
    mod = _load_module(file_path)
    raw = inspect.getdoc(mod) or "{}"
    # If you left “// comments” in the JSON, nuke them:
    cleaned = "\n".join(line.split("//")[0] for line in raw.splitlines())
    return json.loads(cleaned)

def collect_specs(root: str = "tools") -> List[Dict]:
    """Return a list of metadata dicts for every *.py under `root` (non-recursive)."""
    specs = []
    for fp in Path(root).glob("*.py"):
        if fp.name == "utils.py":
            continue
        try:
            specs.append(parse_docstring(str(fp)))
        except Exception as e:
            print(f"⚠️  {fp.name}: {e}")
    return specs

if __name__ == "__main__":
    specs = collect_specs("tools")
    Path("tool-spec.json").write_text(json.dumps(specs, indent=2))
