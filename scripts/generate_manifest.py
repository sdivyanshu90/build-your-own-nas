"""Regenerate the per-file source tables in ``docs/repository-manifest.md``.

A manifest maintained by hand goes stale the first time a module is added, and a stale
manifest is worse than none — it tells you a file exists that does not. So the source
tables are derived from the code itself:

* **Purpose** is each module's first docstring line.
* **Public symbols** are the top-level names not starting with ``_``.
* **Depends on** is the internal import graph, read from the AST.
* **Tests** are the test files that import the module.

Everything outside the marked region — the prose, the layering discussion, and the tables
for tests, configs, scripts, and packaging — is written by hand and left untouched.

Usage::

    python scripts/generate_manifest.py            # rewrite the tables
    python scripts/generate_manifest.py --check    # fail if they are out of date
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_tables import render as render_table

#: Region markers in the manifest. Everything between them is generated.
BEGIN = "<!-- BEGIN GENERATED SOURCE TABLES -->"
END = "<!-- END GENERATED SOURCE TABLES -->"

#: Subpackages in dependency order, shallowest first. Also the order of the tables.
PACKAGE_ORDER = (
    ".",
    "utilities",
    "observability",
    "architectures",
    "search_space",
    "models",
    "datasets",
    "training",
    "objectives",
    "evaluation",
    "search",
    "persistence",
    "config",
    "orchestration",
    "reporting",
)

#: Test directory prefixes, abbreviated to keep the table readable.
TEST_PREFIXES = {
    "unit/": "u:",
    "property/": "p:",
    "integration/": "i:",
    "end_to_end/": "e:",
    "regression/": "r:",
    "performance/": "perf:",
    "failure_recovery/": "f:",
}

#: Public symbols shown before truncating with a ``+n`` count.
MAX_SYMBOLS = 6

#: Test references shown before collapsing to a count. A module like ``exceptions`` is
#: imported by seventeen test files; listing them all produced a 275-character cell that
#: forced the whole table into horizontal scrolling and told the reader nothing useful.
MAX_TESTS = 4


def abbreviate_test(path: str) -> str:
    """Shorten a test path to its directory prefix and stem.

    Args:
        path: Test path relative to ``tests/``.

    Returns:
        For example ``unit/test_models.py`` becomes ``u:models``.
    """
    for directory, prefix in TEST_PREFIXES.items():
        if path.startswith(directory):
            stem = path[len(directory) :].removeprefix("test_").removesuffix(".py")
            return prefix + stem
    return path


def collect_test_imports(tests_root: Path) -> dict[str, set[str]]:
    """Map each ``nas_engine`` module to the test files importing it.

    Args:
        tests_root: The ``tests`` directory.

    Returns:
        Module name (without the ``nas_engine.`` prefix) to test paths.
    """
    imports: dict[str, set[str]] = defaultdict(set)
    for path in sorted(tests_root.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module.startswith("nas_engine."))
            ):
                module = node.module[len("nas_engine.") :]
                imports[module].add(path.relative_to(tests_root).as_posix())
    return imports


def public_symbols(tree: ast.Module) -> list[str]:
    """Return the module's top-level public names.

    Args:
        tree: Parsed module.

    Returns:
        Class, function, and upper-case constant names not starting with ``_``.
    """
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and not node.name.startswith("_"):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
                and target.id.isupper()
                and not target.id.startswith("_")
            )
    return names


def internal_dependencies(tree: ast.Module, own_package: str) -> list[str]:
    """Return the internal subpackages a module imports from.

    Args:
        tree: Parsed module.
        own_package: The module's own subpackage, excluded from the result.

    Returns:
        Sorted subpackage names.
    """
    found = {
        node.module[len("nas_engine.") :].split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("nas_engine.")
    }
    return sorted(found - {own_package})


def module_name(path: Path, source_root: Path) -> str:
    """Return the dotted module name relative to the package.

    Args:
        path: Module file.
        source_root: The ``src/nas_engine`` directory.

    Returns:
        For example ``architectures.spec``, or ``__init__`` for the package root.
    """
    relative = path.relative_to(source_root).with_suffix("").as_posix().replace("/", ".")
    return relative.removesuffix(".__init__") or "__init__"


def render_tables(source_root: Path, tests_root: Path) -> str:
    """Render the per-subpackage Markdown tables.

    Args:
        source_root: The ``src/nas_engine`` directory.
        tests_root: The ``tests`` directory.

    Returns:
        The generated Markdown.
    """
    tests = collect_test_imports(tests_root)
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(source_root.rglob("*.py")):
        grouped[path.relative_to(source_root).parent.as_posix()].append(path)

    unknown = set(grouped) - set(PACKAGE_ORDER)
    if unknown:
        message = f"subpackages missing from PACKAGE_ORDER: {sorted(unknown)}"
        raise SystemExit(message)

    lines: list[str] = []
    for package in PACKAGE_ORDER:
        files = grouped.get(package, [])
        if not files:
            continue
        title = "nas_engine/" if package == "." else f"nas_engine/{package}/"
        lines.append(f"### `{title}`\n")
        rows = [["File", "Lines", "Purpose", "Public symbols", "Depends on", "Tests"]]
        for path in files:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
            purpose = (ast.get_docstring(tree) or "").split("\n")[0].rstrip(".")
            symbols = public_symbols(tree)
            shown = ", ".join(f"`{name}`" for name in symbols[:MAX_SYMBOLS])
            if len(symbols) > MAX_SYMBOLS:
                shown += f" +{len(symbols) - MAX_SYMBOLS}"
            depends = ", ".join(internal_dependencies(tree, package)) or "—"
            related = sorted(tests.get(module_name(path, source_root), ()))
            rendered = ", ".join(abbreviate_test(item) for item in related[:MAX_TESTS])
            if len(related) > MAX_TESTS:
                rendered += f" +{len(related) - MAX_TESTS}"
            rows.append(
                [
                    f"`{path.relative_to(source_root).as_posix()}`",
                    str(len(text.splitlines())),
                    purpose,
                    shown or "—",
                    depends,
                    rendered or "(via package)",
                ]
            )
        # Render through the shared formatter so `make docs` does not immediately report
        # the generated tables as unaligned.
        lines.extend(render_table(rows, ["left", "right", "left", "left", "left", "left"]))
        lines.append("")
    return "\n".join(lines)


def splice(manifest: str, tables: str) -> str:
    """Replace the generated region of the manifest.

    Args:
        manifest: Current manifest contents.
        tables: Newly rendered tables.

    Returns:
        The manifest with the region replaced.

    Raises:
        SystemExit: If either marker is missing.
    """
    start = manifest.find(BEGIN)
    end = manifest.find(END)
    if start == -1 or end == -1 or end < start:
        message = f"{BEGIN} / {END} markers not found in the manifest"
        raise SystemExit(message)
    return f"{manifest[: start + len(BEGIN)]}\n\n{tables}\n{manifest[end:]}"


def main() -> int:
    """Regenerate or verify the manifest's source tables.

    Returns:
        ``0`` on success, ``1`` when ``--check`` finds the manifest out of date.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("src/nas_engine"))
    parser.add_argument("--tests", type=Path, default=Path("tests"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/repository-manifest.md"))
    parser.add_argument(
        "--check", action="store_true", help="fail instead of rewriting when out of date"
    )
    arguments = parser.parse_args()

    for path in (arguments.source, arguments.tests):
        if not path.is_dir():
            print(f"not a directory: {path}", file=sys.stderr)
            return 1
    if not arguments.manifest.is_file():
        print(f"manifest not found: {arguments.manifest}", file=sys.stderr)
        return 1

    current = arguments.manifest.read_text(encoding="utf-8")
    updated = splice(current, render_tables(arguments.source, arguments.tests))

    if updated == current:
        print(f"{arguments.manifest} is up to date")
        return 0
    if arguments.check:
        print(
            f"{arguments.manifest} is out of date; run: python scripts/generate_manifest.py",
            file=sys.stderr,
        )
        return 1
    arguments.manifest.write_text(updated, encoding="utf-8")
    print(f"{arguments.manifest} regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
