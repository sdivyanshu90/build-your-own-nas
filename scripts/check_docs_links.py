"""Verify that every relative link in the documentation resolves.

Documentation rots quietly: a file gets renamed and twenty links break with no error
anywhere. This script walks every Markdown page and checks three things:

* the file a link points at exists;
* an ``#anchor`` suffix matches a real heading on the target page;
* a bare ``#anchor`` matches a heading on the *same* page.

Anchors are checked because they are the part most likely to rot — renaming a heading
breaks every link into it, and nothing else notices. Slugs follow GitHub's rule: lowercase,
drop anything that is not alphanumeric, a space, or a hyphen, then replace spaces with
hyphens. Duplicate headings get GitHub's ``-1``, ``-2`` suffixes.

External ``http(s)`` links are skipped deliberately — checking them would make the build
depend on the network, which the project forbids.

Usage::

    python scripts/check_docs_links.py
    python scripts/check_docs_links.py --root docs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: Matches ``[text](target)``. The target keeps its ``#anchor`` so it can be checked.
LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

#: Matches an ATX heading, capturing its text.
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")

#: Matches a fenced code block delimiter.
FENCE_PATTERN = re.compile(r"^\s{0,3}(?:```|~~~)")

#: Matches inline Markdown that a heading slug ignores: links, code spans, emphasis.
_INLINE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")

#: Schemes that are not filesystem paths.
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "ftp://")


def slugify(heading: str) -> str:
    """Convert heading text to a GitHub-style anchor slug.

    Args:
        heading: Raw heading text, possibly containing inline Markdown.

    Returns:
        The anchor slug, without a leading ``#``.
    """
    text = _INLINE_LINK.sub(r"\1", heading)
    text = text.replace("`", "").replace("*", "").replace("_", "")
    text = "".join(char for char in text.lower() if char.isalnum() or char in " -")
    return "-".join(text.split())


def collect_anchors(text: str) -> set[str]:
    """Return every anchor a Markdown document defines.

    Headings inside fenced code blocks are ignored — a ``#`` in a shell snippet is a
    comment, not a heading. Duplicate slugs gain GitHub's numeric suffixes.

    Args:
        text: The document's contents.

    Returns:
        Every valid anchor for the document.
    """
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    in_fence = False
    for line in text.splitlines():
        if FENCE_PATTERN.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = HEADING_PATTERN.match(line)
        if match is None:
            continue
        slug = slugify(match.group(2))
        if not slug:
            continue
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def check(root: Path, extra: list[Path]) -> list[str]:
    """Return a list of broken link descriptions.

    Args:
        root: Directory to walk for Markdown files.
        extra: Additional files to check.

    Returns:
        Human-readable descriptions of every broken link.
    """
    broken: list[str] = []
    pages = sorted(root.rglob("*.md")) + [path for path in extra if path.is_file()]
    contents = {page: page.read_text(encoding="utf-8") for page in pages}
    anchors = {page: collect_anchors(text) for page, text in contents.items()}

    for page, text in contents.items():
        for target in LINK_PATTERN.findall(text):
            if target.startswith(EXTERNAL_PREFIXES):
                continue

            path_part, _, anchor = target.partition("#")

            if not path_part:
                # A same-page anchor.
                if anchor and anchor not in anchors[page]:
                    broken.append(f"{page}: #{anchor} (no such heading on this page)")
                continue

            resolved = (page.parent / path_part).resolve()
            if not resolved.exists():
                broken.append(f"{page}: {target}")
                continue

            if not anchor:
                continue
            if resolved.suffix != ".md":
                continue
            known = anchors.get(resolved)
            if known is None:
                # Outside the checked set (e.g. a link into the source tree); read it.
                known = collect_anchors(resolved.read_text(encoding="utf-8"))
                anchors[resolved] = known
            if anchor not in known:
                broken.append(f"{page}: {target} (no such heading in {path_part})")
    return broken


def main() -> int:
    """Check every documentation link.

    Returns:
        ``0`` when all links resolve, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("docs"))
    parser.add_argument(
        "--also",
        type=Path,
        nargs="*",
        default=[Path("README.md")],
        help="additional Markdown files to check",
    )
    arguments = parser.parse_args()

    if not arguments.root.is_dir():
        print(f"documentation root not found: {arguments.root}", file=sys.stderr)
        return 1

    broken = check(arguments.root, list(arguments.also))
    pages = len(list(arguments.root.rglob("*.md"))) + len(arguments.also)
    if broken:
        print(f"{len(broken)} broken link(s) across {pages} pages:", file=sys.stderr)
        for entry in broken:
            print(f"  {entry}", file=sys.stderr)
        return 1
    print(f"{pages} documentation pages checked, all links and anchors resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
