"""Distribution selection must cover source modules without broadening materials."""
from collections import defaultdict
import importlib.util
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aka_distribution_builder", ROOT / "tools/build_distributions.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_every_python_source_has_exactly_one_distribution():
    # Discover source independently of the build lists: checking only staged
    # members cannot detect a new module that was never selected for a wheel.
    owners = defaultdict(list)
    for distribution in builder.OWNERS:
        for path in builder.owned(distribution):
            owners[path].append(distribution)
    missing = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "src/aka").rglob("*.py")
                     if path not in owners)
    overlap = {str(path.relative_to(ROOT)): distributions for path, distributions in owners.items()
               if len(distributions) != 1}
    assert not missing, f"Python source modules omitted from all distributions: {missing}"
    assert not overlap, f"Files claimed by multiple distributions: {overlap}"


def test_schema_new_business_module_is_included_without_unlisted_materials(tmp_path, monkeypatch):
    schema = tmp_path / "src/aka/workflow/schema"
    shutil.copytree(ROOT / "src/aka/workflow/schema", schema,
                    ignore=shutil.ignore_patterns("__pycache__"))
    module = schema / "new_business_module.py"
    module.write_text("# New local runtime module, outside the material release.\n")
    unlisted = (schema / "draft.md", schema / "assets/unlisted.yaml", schema / "prompts/unlisted.py")
    for path in unlisted:
        path.write_text("unlisted material\n")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    selected = set(builder.owned("atrex-aka-schema"))
    assert module in selected
    assert selected.isdisjoint(unlisted)
    assert all(path.is_relative_to(schema) for path in selected)
