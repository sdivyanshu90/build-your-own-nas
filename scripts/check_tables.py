r"""Check — and optionally reformat — every Markdown table in the documentation.

A table whose source pipes do not line up still renders correctly, but it is unreadable in
an editor, a diff, or a terminal, which is where most of this documentation is actually
read. Worse problems hide in the same place: a row with the wrong number of cells, a
column with no header, or a cell so wide it forces the rendered table into horizontal
scrolling.

The checks:

* **Shape.** Every row has the same number of cells as its header.
* **Headers.** No column is unnamed — an empty header cell means the table's structure
  does not describe its data.
* **Spacing.** A table is separated from surrounding prose by blank lines, or it does not
  render as a table at all.
* **Width.** No cell exceeds :data:`MAX_CELL_WIDTH`. Prose columns are exempt; the target
  is list-like cells that should have been rows.
* **Alignment.** Pipes line up, when doing so keeps lines within
  :data:`MAX_ALIGNED_WIDTH`. Beyond that, alignment would trade one readability problem
  for a worse one, so the table is normalised to single-space padding instead.

Cells are split on *unescaped* pipes, matching GitHub. A ``|`` inside a code span still
splits a cell in GFM, so ``\\|`` is the only way to write a literal pipe — this is a real
source of silently mangled tables, and the shape check catches it.

Usage::

    python scripts/check_tables.py             # report problems, exit 1 if any
    python scripts/check_tables.py --fix       # reformat in place
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

#: Opens or closes a fenced code block. Tables inside fences are examples, not tables.
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

#: The ``| --- | :--: |`` row that turns a pipe row into a table.
SEPARATOR = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$")

#: Widest a single cell may be before it should have become rows of its own.
MAX_CELL_WIDTH = 200

#: Widest an aligned row may be before alignment stops being worth it. At 160 about three
#: quarters of the tables in this repository align; raising it further only catches tables
#: whose prose cells would push rows past 200 characters, where padding hurts more than the
#: ragged pipes it fixes.
MAX_ALIGNED_WIDTH = 160

#: Wide cells that are prose rather than packed lists, and are fine as they are.
PROSE_COLUMNS = frozenset(
    {"why", "meaning", "definition", "covers", "contents", "what it does", "notes", "detail"}
)


def split_cells(row: str) -> list[str]:
    """Split a table row into cells on unescaped pipes, as GitHub does.

    Args:
        row: One line of a Markdown table.

    Returns:
        The row's cells, without the empty leading and trailing cells that a leading or
        trailing pipe produces.
    """
    cells: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(row):
        char = row[index]
        if char == "\\" and index + 1 < len(row):
            current.append(row[index : index + 2])
            index += 2
            continue
        if char == "|":
            cells.append("".join(current))
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    cells.append("".join(current))
    if cells and not cells[0].strip():
        cells = cells[1:]
    if cells and not cells[-1].strip():
        cells = cells[:-1]
    return cells


def display_width(text: str) -> int:
    """Return the rendered width of a cell.

    Emoji used as status markers occupy two terminal columns, so counting code points
    would misalign every row that contains one.

    Args:
        text: Cell text.

    Returns:
        Width in terminal columns.
    """
    width = 0
    for char in text:
        code = ord(char)
        wide = (
            0x1100 <= code <= 0x115F
            or 0x2E80 <= code <= 0xA4CF
            or 0xAC00 <= code <= 0xD7A3
            or 0xF900 <= code <= 0xFAFF
            or 0xFE30 <= code <= 0xFE6F
            or 0xFF00 <= code <= 0xFF60
            or 0xFFE0 <= code <= 0xFFE6
            or 0x1F300 <= code <= 0x1FAFF
            or 0x2600 <= code <= 0x27BF
        )
        # Variation selectors and zero-width joiners render as part of the previous glyph.
        if code in (0xFE0F, 0xFE0E, 0x200D):
            continue
        width += 2 if wide else 1
    return width


def pad(text: str, width: int, alignment: str) -> str:
    """Pad a cell to a display width.

    Args:
        text: Cell text.
        width: Target display width.
        alignment: ``"left"``, ``"right"``, or ``"center"``.

    Returns:
        The padded cell.
    """
    filler = width - display_width(text)
    if filler <= 0:
        return text
    if alignment == "right":
        return " " * filler + text
    if alignment == "center":
        left = filler // 2
        return " " * left + text + " " * (filler - left)
    return text + " " * filler


def parse_alignments(separator: str) -> list[str]:
    """Read column alignments from a separator row.

    Args:
        separator: The ``| --- | ---: |`` row.

    Returns:
        One of ``"left"``, ``"right"``, ``"center"`` per column.
    """
    alignments: list[str] = []
    for cell in split_cells(separator):
        text = cell.strip()
        if text.startswith(":") and text.endswith(":"):
            alignments.append("center")
        elif text.endswith(":"):
            alignments.append("right")
        else:
            alignments.append("left")
    return alignments


def render(rows: list[list[str]], alignments: list[str]) -> list[str]:
    """Render a table, aligned if that keeps lines reasonable.

    Args:
        rows: Header row followed by body rows, cells already stripped.
        alignments: Per-column alignment.

    Returns:
        The rendered lines, including the separator.
    """
    columns = len(rows[0])
    widths = [max(display_width(row[index]) for row in rows) for index in range(columns)]
    # "| " + cell + " " per column, plus the closing "|".
    projected = sum(width + 3 for width in widths) + 1

    if projected > MAX_ALIGNED_WIDTH:
        header, *body = rows
        marks = {"left": "---", "right": "---:", "center": ":---:"}
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("| " + " | ".join(marks[a] for a in alignments) + " |")
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        return lines

    header, *body = rows
    lines = ["| " + " | ".join(pad(c, widths[i], "left") for i, c in enumerate(header)) + " |"]
    dashes = []
    for index, alignment in enumerate(alignments):
        width = widths[index]
        if alignment == "right":
            dashes.append("-" * (width - 1) + ":")
        elif alignment == "center":
            dashes.append(":" + "-" * (width - 2) + ":")
        else:
            dashes.append("-" * width)
    lines.append("| " + " | ".join(dashes) + " |")
    lines.extend(
        "| " + " | ".join(pad(c, widths[i], alignments[i]) for i, c in enumerate(row)) + " |"
        for row in body
    )
    return lines


def check_header(path: Path, header: list[str], alignments: list[str], line: int) -> list[str]:
    """Check a table's header and separator.

    Args:
        path: Page being checked, for the message.
        header: Header cells, stripped.
        alignments: Alignments parsed from the separator.
        line: One-based line number of the header row.

    Returns:
        Problems found.
    """
    problems = [
        f"{path}:{line}: column {position} has no header — "
        "an unnamed column means the table does not describe its own data"
        for position, name in enumerate(header, start=1)
        if not name
    ]
    if len(alignments) != len(header):
        problems.append(
            f"{path}:{line + 1}: separator has {len(alignments)} columns, header has {len(header)}"
        )
    return problems


def check_row(path: Path, header: list[str], cells: list[str], line: int, number: int) -> list[str]:
    """Check one body row against its header.

    Args:
        path: Page being checked, for the message.
        header: Header cells, stripped.
        cells: This row's cells, stripped.
        line: One-based line number of the row.
        number: One-based row number within the table.

    Returns:
        Problems found.
    """
    problems: list[str] = []
    if len(cells) != len(header):
        problems.append(
            f"{path}:{line}: row {number} has {len(cells)} cells, header has "
            f"{len(header)} — an unescaped '|' inside a cell splits it; write '\\|'"
        )
    for position, cell in enumerate(cells[: len(header)]):
        column = header[position].strip("*` ").lower()
        if len(cell) > MAX_CELL_WIDTH and column not in PROSE_COLUMNS:
            problems.append(
                f"{path}:{line}: cell in column {header[position]!r} is "
                f"{len(cell)} characters — split it into rows"
            )
    return problems


def collect_table(
    path: Path, lines: list[str], start: int
) -> tuple[list[list[str]], list[str], int, list[str]]:
    """Read one table, checking it as it goes.

    Args:
        path: Page being read, for messages.
        lines: The page's lines.
        start: Index of the header row.

    Returns:
        ``(rows, alignments, next_index, problems)``. Rows are padded to the header's
        width so the renderer always sees a rectangle, even when a row was malformed.
    """
    header = [cell.strip() for cell in split_cells(lines[start])]
    alignments = parse_alignments(lines[start + 1])
    width = len(header)
    problems = check_header(path, header, alignments, start + 1)
    alignments = (alignments + ["left"] * width)[:width]

    rows = [header]
    index = start + 2
    number = 0
    while index < len(lines):
        row = lines[index]
        if not row.strip() or "|" not in row or FENCE.match(row):
            break
        number += 1
        cells = [cell.strip() for cell in split_cells(row)]
        problems.extend(check_row(path, header, cells, index + 1, number))
        rows.append((cells + [""] * width)[:width])
        index += 1
    return rows, alignments, index, problems


def process(path: Path) -> tuple[list[str], str]:
    """Check one page, returning its problems and its reformatted text.

    Args:
        path: Markdown file.

    Returns:
        A ``(problems, formatted_text)`` pair.
    """
    problems: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    in_fence = False
    fence_char = ""
    index = 0

    while index < len(lines):
        line = lines[index]
        fence = FENCE.match(line)
        if fence:
            if not in_fence:
                in_fence, fence_char = True, fence.group(1)[0]
            elif fence.group(1)[0] == fence_char:
                in_fence = False
            output.append(line)
            index += 1
            continue
        if in_fence or "|" not in line:
            output.append(line)
            index += 1
            continue
        if index + 1 >= len(lines) or not SEPARATOR.match(lines[index + 1]):
            output.append(line)
            index += 1
            continue

        if output and output[-1].strip():
            problems.append(f"{path}:{index + 1}: table is not preceded by a blank line")
        rows, alignments, index, table_problems = collect_table(path, lines, index)
        problems.extend(table_problems)
        output.extend(render(rows, alignments))

        if index < len(lines) and lines[index].strip():
            problems.append(f"{path}:{index + 1}: table is not followed by a blank line")

    text = "\n".join(output)
    if text and not text.endswith("\n"):
        text += "\n"
    return problems, text


def main() -> int:
    """Check, or reformat, every Markdown table.

    Returns:
        ``0`` when everything is well-formed, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument("--fix", action="store_true", help="reformat tables in place")
    arguments = parser.parse_args()

    pages = [
        path
        for path in sorted(arguments.root.rglob("*.md"))
        if ".git" not in path.parts and "node_modules" not in path.parts
    ]

    problems: list[str] = []
    reformatted: list[Path] = []
    for page in pages:
        page_problems, formatted = process(page)
        problems.extend(page_problems)
        if formatted != page.read_text(encoding="utf-8"):
            if arguments.fix:
                page.write_text(formatted, encoding="utf-8")
            reformatted.append(page)

    for problem in problems:
        print(problem, file=sys.stderr)

    if arguments.fix:
        print(f"{len(pages)} pages checked, {len(reformatted)} reformatted")
        return 1 if problems else 0

    if reformatted:
        print(
            f"{len(reformatted)} page(s) have unaligned tables; run: "
            "python scripts/check_tables.py --fix",
            file=sys.stderr,
        )
        for page in reformatted:
            print(f"  {page}", file=sys.stderr)

    if problems or reformatted:
        return 1
    print(f"{len(pages)} pages checked, all tables well-formed and aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
