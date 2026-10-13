"""Extract project-level after-tax totals from supported Excel price appendices."""
from __future__ import annotations

import numbers
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from project_lens.paths import DOCUMENTS_DIR

DOCUMENTS = DOCUMENTS_DIR


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", text.replace("\u0111", "d")).strip()


def _row_numbers(row: Any) -> list[float]:
    return [
        float(value)
        for value in row
        if isinstance(value, numbers.Number)
        and not isinstance(value, bool)
        and pd.notna(value)
        and float(value) > 0
    ]


def _project_name(sheets: dict[str, pd.DataFrame], path: Path) -> str:
    labels = ("ten du an", "ten cong trinh", "du an", "cong trinh")
    for frame in sheets.values():
        for value in frame.head(15).to_numpy().flat:
            if not isinstance(value, str):
                continue
            text = re.sub(r"\s+", " ", value).strip(" \t\r\n\"'")
            normalized = normalize(text)
            for label in labels:
                if re.search(rf"\b{label}\s*[:\uFF1A]\s*(.+)", normalized):
                    original = re.search(r"[:\uFF1A]\s*(.+)", text)
                    if original:
                        candidate = original.group(1).strip(" \t\r\n\"',")
                        if len(candidate) >= 8:
                            return candidate
    return path.stem


def _extract_totals(sheets: dict[str, pd.DataFrame]) -> list[tuple[str, float]]:
    totals: list[tuple[str, float]] = []
    for sheet_name, frame in sheets.items():
        for _, row in frame.iterrows():
            values = row.tolist()
            text = normalize(" ".join(str(v) for v in values if isinstance(v, str)))
            if "tong gia tri sau thue" in text or "tong cong sau thue" in text:
                amounts = _row_numbers(values)
                if amounts:
                    totals.append((sheet_name, amounts[-1]))
    if totals:
        # Multiple sheets can repeat the same contract total. Do not sum them.
        return [max(totals, key=lambda item: item[1])]

    # Legacy estimate workbook: dedicated summary sheet with no "after tax" label.
    for sheet_name, frame in sheets.items():
        if normalize(sheet_name) != "th chi phi xd":
            continue
        for _, row in frame.iterrows():
            values = row.tolist()
            text = normalize(" ".join(str(v) for v in values if isinstance(v, str)))
            if re.search(r"\btong cong\b", text):
                amounts = _row_numbers(values)
                if amounts:
                    return [(sheet_name, amounts[-1])]
    return []


def load_project_costs() -> list[dict[str, Any]]:
    """Read one grand after-tax total per non-template Excel price document."""
    results: list[dict[str, Any]] = []
    if not DOCUMENTS.exists():
        return results
    for path in sorted(DOCUMENTS.rglob("*")):
        if (
            not path.is_file()
            or path.name.startswith("~$")
            or path.suffix.lower() not in {".xlsx", ".xls"}
            or "mau" in normalize(path.stem)
        ):
            continue
        try:
            sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
        except Exception:
            continue
        for sheet_name, amount in _extract_totals(sheets):
            results.append(
                {
                    "project": _project_name(sheets, path),
                    "amount_vnd": amount,
                    "source": str(path.relative_to(ROOT)),
                    "sheet": sheet_name,
                }
            )
    return results


def is_max_cost_question(question: str) -> bool:
    text = normalize(question)
    asks_max = any(term in text for term in ("cao nhat", "lon nhat", "dat nhat", "nhieu nhat"))
    asks_cost = any(term in text for term in ("chi phi", "gia", "tong tien", "gia tri"))
    return asks_max and asks_cost


def highest_cost_answer(costs: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    if not costs:
        return (
            "Ch\u01b0a t\u00ecm th\u1ea5y d\u00f2ng t\u1ed5ng gi\u00e1 tr\u1ecb sau thu\u1ebf trong c\u00e1c b\u1ea3ng Excel \u0111\u01b0\u1ee3c h\u1ed7 tr\u1ee3.",
            [],
        )
    highest = max(item["amount_vnd"] for item in costs)
    winners = [item for item in costs if item["amount_vnd"] == highest]
    lines = ["M\u1ee9c t\u1ed5ng sau thu\u1ebf cao nh\u1ea5t t\u00ecm th\u1ea5y trong c\u00e1c b\u1ea3ng Excel:"]
    sources = []
    for index, item in enumerate(winners, start=1):
        lines.append(
            f"- {item['project']}: {item['amount_vnd']:,.0f} \u0111\u1ed3ng "
            f"[Ngu\u1ed3n {index}: {item['source']}, sheet {item['sheet']}]."
        )
        sources.append(
            {
                "source_id": index,
                "label": f"{item['source']} | Sheet: {item['sheet']}",
                "score": 1.0,
            }
        )
    lines.append(
        "K\u1ebft qu\u1ea3 ch\u1ec9 so s\u00e1nh c\u00e1c b\u1ea3ng Excel c\u00f3 d\u00f2ng t\u1ed5ng sau thu\u1ebf \u0111\u1ecdc \u0111\u01b0\u1ee3c; "
        "RAR, HCE v\u00e0 c\u00e1c t\u00e0i li\u1ec7u kh\u00f4ng c\u00f3 t\u1ed5ng d\u1ef1 \u00e1n t\u01b0\u01a1ng \u0111\u01b0\u01a1ng ch\u01b0a \u0111\u01b0\u1ee3c \u0111\u1ed1i chi\u1ebfu."
    )
    return "\n".join(lines), sources
