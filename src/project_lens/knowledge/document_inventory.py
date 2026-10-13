"""Filename-based inventory for drawing files that text retrieval cannot find."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

from project_lens.paths import DOCUMENTS_DIR

DOCUMENTS = DOCUMENTS_DIR


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("\u0111", "d")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def is_drawing_or_file_question(question: str) -> bool:
    text = normalize(question)
    return (
        "file:///" in question.lower()
        or "ban ve" in text
        or "so do" in text
        or "dwg" in text.split()
        or "dxf" in text.split()
        or ("pdf" in text.split() and any(word in text for word in ("file", "du an", "tai lieu")))
    )


def _file_uri_names(question: str) -> set[str]:
    names: set[str] = set()
    for match in re.findall(r"file:///[^\s\]\)\"']+", question, flags=re.IGNORECASE):
        parsed_path = unquote(urlparse(match).path)
        name = parsed_path.replace("\\", "/").rsplit("/", 1)[-1]
        if name:
            names.add(normalize(name))
    return names


def find_drawing_files(question: str = "") -> list[dict[str, Any]]:
    """Return likely drawing PDFs/CAD files, optionally narrowed by file URI."""
    if not DOCUMENTS.exists():
        return []
    requested_names = _file_uri_names(question)
    found = []
    for path in DOCUMENTS.rglob("*"):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        relative = path.relative_to(DOCUMENTS)
        normalized_name = normalize(path.stem)
        normalized_parts = [normalize(part) for part in relative.parts[:-1]]
        is_named_drawing = any(term in normalized_name for term in ("ban ve", "drawing"))
        is_plan_folder = any("ha tang phu tro" in part for part in normalized_parts)
        is_cad = path.suffix.lower() in {".dwg", ".dxf"}
        if not (is_named_drawing or is_plan_folder or is_cad):
            continue
        if requested_names and normalize(path.name) not in requested_names:
            continue

        relative_text = str(relative)
        plan_folder_index = next(
            (index for index, part in enumerate(normalized_parts) if "ha tang phu tro" in part),
            None,
        )
        if plan_folder_index is not None and plan_folder_index > 0:
            project = relative.parts[plan_folder_index - 1]
        else:
            project = path.stem
        found.append(
            {
                "project": project,
                "path": relative_text,
                "extension": path.suffix.lower(),
            }
        )
    return found


def drawing_inventory_answer(
    question: str, files: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    if not files:
        return (
            "Kh\u00f4ng t\u00ecm th\u1ea5y t\u1eadp tin b\u1ea3n v\u1ebd kh\u1edbp v\u1edbi c\u00e2u h\u1ecfi trong documents.",
            [],
        )
    asked_specific_file = bool(_file_uri_names(question))
    if asked_specific_file:
        lines = ["T\u00ecm th\u1ea5y t\u1eadp tin \u0111\u01b0\u1ee3c nh\u1eafc \u0111\u1ebfn trong documents:"]
    else:
        lines = ["C\u00e1c h\u1ed3 s\u01a1 c\u00f3 t\u1eadp tin b\u1ea3n v\u1ebd theo t\u00ean file/th\u01b0 m\u1ee5c:"]
    sources = []
    for index, item in enumerate(files, start=1):
        lines.append(f"- {item['project']}: {item['path']} [Ngu\u1ed3n {index}].")
        sources.append({"source_id": index, "label": item["path"], "score": 1.0})
    lines.append(
        "\u0110\u00e2y l\u00e0 kh\u1edbp theo t\u00ean v\u00e0 v\u1ecb tr\u00ed file; "
        "ch\u01b0a x\u00e1c nh\u1eadn n\u1ed9i dung s\u01a1 \u0111\u1ed3 b\u00ean trong PDF/CAD."
    )
    return "\n".join(lines), sources


def is_appendix_length_question(question: str) -> bool:
    text = normalize(question)
    asks_appendix = "phu luc" in text or "plhd" in text or "pl01" in text
    asks_length = any(term in text for term in ("dai nhat", "nhieu dong nhat", "nhieu sheet nhat"))
    return asks_appendix and asks_length


def find_longest_excel_appendix() -> dict[str, Any] | None:
    """Rank named Excel appendices by non-empty row count, keeping file provenance."""
    candidates = []
    if not DOCUMENTS.exists():
        return None
    for path in DOCUMENTS.rglob("*"):
        if (
            not path.is_file()
            or path.name.startswith("~$")
            or path.suffix.lower() not in {".xlsx", ".xls"}
        ):
            continue
        name = normalize(path.stem)
        is_appendix = "phu luc" in name or name.startswith("pl") or " pl" in name
        if "mau" in name or not is_appendix:
            continue
        try:
            sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
        except Exception:
            continue
        counts = {
            sheet_name: int(frame.notna().any(axis=1).sum())
            for sheet_name, frame in sheets.items()
        }
        total_rows = sum(counts.values())
        if not total_rows:
            continue
        largest_sheet = max(counts, key=counts.get)
        relative = path.relative_to(DOCUMENTS)
        project = relative.parts[0] if len(relative.parts) > 1 else path.stem
        candidates.append(
            {
                "project": project,
                "path": str(relative),
                "sheets": len(sheets),
                "rows": total_rows,
                "largest_sheet": largest_sheet,
                "largest_sheet_rows": counts[largest_sheet],
            }
        )
    return max(candidates, key=lambda item: item["rows"]) if candidates else None


def longest_appendix_answer(item: dict[str, Any] | None) -> tuple[str, list[dict[str, Any]]]:
    if not item:
        return "Ch\u01b0a t\u00ecm th\u1ea5y file ph\u1ee5 l\u1ee5c Excel ph\u00f9 h\u1ee3p trong documents.", []
    answer = (
        f"Theo s\u1ed1 d\u00f2ng c\u00f3 d\u1eef li\u1ec7u trong c\u00e1c ph\u1ee5 l\u1ee5c Excel, "
        f"h\u1ed3 s\u01a1 {item['project']} c\u00f3 ph\u1ee5 l\u1ee5c d\u00e0i nh\u1ea5t: "
        f"{item['path']} ({item['sheets']} sheet, {item['rows']:,} d\u00f2ng c\u00f3 d\u1eef li\u1ec7u). "
        f"Sheet d\u00e0i nh\u1ea5t l\u00e0 {item['largest_sheet']} "
        f"({item['largest_sheet_rows']:,} d\u00f2ng c\u00f3 d\u1eef li\u1ec7u). "
        "\u0110\u00e2y l\u00e0 so s\u00e1nh theo d\u00f2ng Excel, kh\u00f4ng ph\u1ea3i s\u1ed1 trang PDF; "
        "file RAR v\u00e0 c\u00e1c ph\u1ee5 l\u1ee5c kh\u00f4ng c\u00f3 t\u00ean nh\u1eadn di\u1ec7n ch\u01b0a \u0111\u01b0\u1ee3c t\u00ednh."
    )
    return answer, [{"source_id": 1, "label": item["path"], "score": 1.0}]
