"""Regression tests pinning the factual claims the documentation makes.

Prose rots differently from code. A renamed test, a removed CLI command, or a new module
breaks nothing at runtime — it just makes a page quietly wrong, and a traceability matrix
citing tests that do not exist is worse than no matrix, because it claims verification that
is not there.

These tests check the claims that are mechanically checkable:

* every ``test_*`` and ``Test*`` name cited in the docs exists in the suite;
* the counts the docs state (CLI commands, exported symbols, tables, mutation operators,
  documentation pages) match reality;
* every source module is imported by at least one test file;
* every Markdown table is well-formed — right cell count, no unnamed column, no cell so
  wide it should have been rows.

That last one is not cosmetic. An unescaped ``|`` inside a cell silently splits it, so the
row renders with the wrong number of columns and the reader sees mangled data with no
error anywhere. An unnamed column is the same failure one level up: the table's shape stops
describing its content.

Deliberately *not* checked: whether the prose is a good explanation. That is a review
concern, and no test can stand in for it.
"""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.regression

REPOSITORY = Path(__file__).resolve().parents[2]
DOCS = REPOSITORY / "docs"
SOURCE = REPOSITORY / "src" / "nas_engine"
TESTS = REPOSITORY / "tests"

#: Matches a ``test_...`` name inside backticks in Markdown.
CITED_TEST = re.compile(r"`(test_[a-z0-9_]+)`")

#: Matches a ``TestSomething`` class name inside backticks in Markdown.
CITED_CLASS = re.compile(r"`(Test[A-Z][A-Za-z0-9_]*)`")

#: Names that appear in the docs as *illustrations*, not as citations. Each is a
#: deliberately bad example used to make a point about naming, so it must not exist.
ILLUSTRATIVE_NAMES = frozenset({"test_deque_maxlen"})


def _markdown_pages() -> list[Path]:
    """Return every documentation page, plus the README."""
    return [*sorted(DOCS.rglob("*.md")), REPOSITORY / "README.md"]


def _defined_names() -> tuple[set[str], set[str]]:
    """Return the test function and test class names the suite defines.

    Returns:
        A ``(functions, classes)`` pair.
    """
    functions: set[str] = set()
    classes: set[str] = set()
    for path in TESTS.rglob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                functions.add(node.name)
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                classes.add(node.name)
    return functions, classes


class TestCitedTestsExist:
    def test_every_cited_test_function_exists(self) -> None:
        functions, _ = _defined_names()
        missing: list[str] = []
        for page in _markdown_pages():
            text = page.read_text(encoding="utf-8")
            missing.extend(
                f"{page.relative_to(REPOSITORY)}: {name}"
                for name in sorted(set(CITED_TEST.findall(text)))
                if name not in functions and name not in ILLUSTRATIVE_NAMES
            )
        assert not missing, (
            "the documentation cites tests that do not exist, so it claims verification "
            "that is not there:\n  " + "\n  ".join(missing)
        )

    def test_illustrative_names_really_are_absent(self) -> None:
        # The allowlist above exists so bad-example names in the prose do not trip the
        # citation check. If one becomes a real test, the example stops making its point
        # and the allowlist entry should go.
        functions, _ = _defined_names()
        collisions = sorted(ILLUSTRATIVE_NAMES & functions)
        assert not collisions, (
            "these names are used in the docs as counterexamples but now exist as real "
            f"tests: {collisions}"
        )

    def test_every_cited_test_class_exists(self) -> None:
        _, classes = _defined_names()
        missing: list[str] = []
        for page in _markdown_pages():
            text = page.read_text(encoding="utf-8")
            missing.extend(
                f"{page.relative_to(REPOSITORY)}: {name}"
                for name in sorted(set(CITED_CLASS.findall(text)))
                if name not in classes
            )
        assert not missing, "the documentation cites test classes that do not exist:\n  " + (
            "\n  ".join(missing)
        )

    def test_the_scan_finds_citations_at_all(self) -> None:
        # Without this, a regex that stopped matching would make both tests above pass
        # vacuously while the documentation rotted freely.
        cited = {
            name
            for page in _markdown_pages()
            for name in CITED_TEST.findall(page.read_text(encoding="utf-8"))
        }
        assert len(cited) > 100, f"expected the docs to cite many tests, found {len(cited)}"


class TestDocumentedCountsAreTrue:
    def test_the_cli_command_count_is_right(self) -> None:
        from nas_engine import cli

        commands = [
            node
            for node in ast.walk(ast.parse(Path(cli.__file__).read_text(encoding="utf-8")))
            if isinstance(node, ast.FunctionDef)
            and any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "command"
                for decorator in node.decorator_list
            )
        ]
        assert len(commands) == 13, (
            f"the docs say 13 CLI commands, the code has {len(commands)}; "
            "update docs/traceability-matrix.md and docs/repository-manifest.md"
        )

    def test_the_exported_symbol_count_is_right(self) -> None:
        import nas_engine

        assert len(nas_engine.__all__) == 51, (
            f"the docs say 51 exported symbols, __all__ has {len(nas_engine.__all__)}"
        )

    def test_the_table_count_is_right(self) -> None:
        from nas_engine.persistence import models

        tables = [value for value in vars(models).values() if hasattr(value, "__tablename__")]
        assert len(tables) == 8, f"the docs say 8 tables, the schema has {len(tables)}"

    def test_the_mutation_operator_count_is_right(self) -> None:
        from nas_engine.search_space.mutation import DEFAULT_OPERATORS

        assert len(DEFAULT_OPERATORS) == 12, (
            f"the docs say 12 mutation operators, there are {len(DEFAULT_OPERATORS)}"
        )

    @pytest.mark.parametrize(
        ("directory", "expected"),
        [
            ("concepts", 10),
            ("architecture", 6),
            ("guides", 8),
            ("testing", 3),
            ("operations", 4),
            ("adr", 4),
        ],
    )
    def test_the_documentation_page_counts_are_right(self, directory: str, expected: int) -> None:
        actual = len(list((DOCS / directory).glob("*.md")))
        assert actual == expected, (
            f"docs/index.md and docs/repository-manifest.md say {expected} pages in "
            f"docs/{directory}/, there are {actual}"
        )


class TestEveryModuleIsReached:
    def test_every_module_is_imported_by_a_test(self) -> None:
        imported: dict[str, set[str]] = defaultdict(set)
        for path in TESTS.rglob("test_*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and (node.module.startswith("nas_engine."))
                ):
                    imported[node.module[len("nas_engine.") :]].add(path.name)

        unreached: list[str] = []
        for path in sorted(SOURCE.rglob("*.py")):
            if path.name == "__init__.py":
                # Packages are exercised through the modules they re-export, and the
                # re-exports themselves are asserted in tests/unit/test_public_api.py.
                continue
            module = path.relative_to(SOURCE).with_suffix("").as_posix().replace("/", ".")
            if module not in imported:
                unreached.append(module)

        assert not unreached, (
            "these modules are not imported by any test file, so docs/repository-manifest.md's "
            "claim that every module is reached is false:\n  " + "\n  ".join(unreached)
        )


class TestTablesAreWellFormed:
    def test_every_table_is_well_formed(self) -> None:
        checker = _load_table_checker()
        problems: list[str] = []
        for page in _markdown_pages():
            page_problems, _ = checker.process(page)
            problems.extend(page_problems)
        assert not problems, (
            "malformed documentation tables — a row with the wrong cell count renders as "
            "mangled data with no error anywhere:\n  " + "\n  ".join(problems)
        )

    def test_every_table_is_aligned(self) -> None:
        checker = _load_table_checker()
        unaligned = [
            str(page.relative_to(REPOSITORY))
            for page in _markdown_pages()
            if checker.process(page)[1] != page.read_text(encoding="utf-8")
        ]
        assert not unaligned, (
            "these pages have unaligned tables; run `make docs-fix`:\n  " + "\n  ".join(unaligned)
        )

    def test_an_unescaped_pipe_would_be_detected(self, tmp_path: Path) -> None:
        # The check exists for exactly this: a literal '|' inside a cell splits it, so the
        # row silently gains a column. Without this test a broken splitter would pass.
        checker = _load_table_checker()
        page = tmp_path / "broken.md"
        page.write_text(
            "# Broken\n\n| Name | Meaning |\n| --- | --- |\n| `a|b` | a pipe |\n",
            encoding="utf-8",
        )
        problems, _ = checker.process(page)
        assert any("3 cells, header has 2" in problem for problem in problems), problems

    def test_an_unnamed_column_would_be_detected(self, tmp_path: Path) -> None:
        checker = _load_table_checker()
        page = tmp_path / "unnamed.md"
        page.write_text("# Unnamed\n\n| Name | |\n| --- | --- |\n| a | b |\n", encoding="utf-8")
        problems, _ = checker.process(page)
        assert any("has no header" in problem for problem in problems), problems


class TestManifestLineCountsAreTrue:
    #: Directories a manifest path may be relative to. The generated source tables use
    #: paths relative to the package root; the hand-written ones use their own directory.
    ROOTS = ("src/nas_engine", "tests", "scripts", "configs", "examples", "")

    #: ``| `path` | 123 | ...`` — a manifest row stating a file's line count. Directory
    #: rows (``| `concepts/` | 10 |``) count pages, not lines, so they are excluded.
    ROW = re.compile(r"^\| `([^`]+[^`/])`\s*\|\s*(\d+)\s*\|", re.MULTILINE)

    def _candidates(self, name: str) -> list[Path]:
        """Return every file a manifest path could refer to.

        A bare name like ``__init__.py`` exists under several roots. Callers that compare
        content must refuse to guess — picking the first match silently compares against
        the wrong file, which is exactly how a bulk line-count fixer once rewrote a
        correct 152 to a wrong 5.

        Args:
            name: Path as written in the manifest.

        Returns:
            Matching files, possibly empty, possibly more than one.
        """
        return [
            candidate
            for root in self.ROOTS
            if (candidate := (REPOSITORY / root / name if root else REPOSITORY / name)).is_file()
        ]

    def test_every_stated_line_count_matches_the_file(self) -> None:
        # The generated source tables recompute their counts, but the tables for tests,
        # scripts, and packaging are written by hand — and a hand-written line count is
        # stale the moment anyone edits the file.
        manifest = (DOCS / "repository-manifest.md").read_text(encoding="utf-8")
        wrong: list[str] = []
        checked = 0
        for name, stated in self.ROW.findall(manifest):
            candidates = self._candidates(name)
            if len(candidates) != 1:
                # Missing files are the other test's job; ambiguous names cannot be
                # compared without guessing which file was meant.
                continue
            checked += 1
            actual = len(candidates[0].read_text(encoding="utf-8").splitlines())
            if actual != int(stated):
                wrong.append(f"{name}: manifest says {stated}, file has {actual}")
        assert checked > 40, f"expected to check many rows, checked {checked}"
        assert not wrong, (
            "docs/repository-manifest.md states line counts that are out of date:\n  "
            + "\n  ".join(wrong)
        )

    def test_every_named_file_exists(self) -> None:
        manifest = (DOCS / "repository-manifest.md").read_text(encoding="utf-8")
        missing = [name for name, _ in self.ROW.findall(manifest) if not self._candidates(name)]
        assert not missing, (
            "docs/repository-manifest.md lists files that do not exist:\n  " + "\n  ".join(missing)
        )


def _load_table_checker() -> ModuleType:
    """Import ``scripts/check_tables.py``, which is outside the installed package.

    Importing the real checker — rather than reimplementing its rules here — is the point:
    a bug in the script and a bug in the test cannot cancel each other out.

    Returns:
        The ``check_tables`` module.
    """
    scripts = REPOSITORY / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import check_tables

    return check_tables
