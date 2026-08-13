"""Build a compact space-group lookup index from Tables S1 and S2."""

from __future__ import annotations

import argparse
import bisect
import json
import re
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PROJECT_ROOT / "emergent particles" / "1-s2.0-S2095927321006927-mmc1.pdf"
DEFAULT_OUTPUT = PROJECT_ROOT / "emergent particles" / "emergent_particles_index.json"

PARTICLE_NAMES = {
    "C-1 WP": "Charge-1 Weyl point",
    "C-2 WP": "Charge-2 Weyl point",
    "C-3 WP": "Charge-3 Weyl point",
    "C-4 WP": "Charge-4 Weyl point",
    "TP": "Triple point",
    "C-2 TP": "Charge-2 triple point",
    "QTP": "Quadratic triple point",
    "QCTP": "Quadratic contact triple point",
    "DP": "Dirac point",
    "C-2 DP": "Charge-2 Dirac point",
    "C-4 DP": "Charge-4 Dirac point",
    "QDP": "Quadratic Dirac point",
    "C-4 QDP": "Charge-4 quadratic Dirac point",
    "CCDP": "Cubic crossing Dirac point",
    "QCDP": "Quadratic contact Dirac point",
    "CDP": "Cubic Dirac point",
    "SP": "Sextuple point",
    "C-4 SP": "Charge-4 sextuple point",
    "QCSP": "Quadratic contact sextuple point",
    "OP": "Octuple point",
    "WNL": "Weyl nodal line",
    "WNL net": "Weyl nodal-line net",
    "QNL": "Quadratic nodal line",
    "CNL": "Cubic nodal line",
    "DNL": "Dirac nodal line",
    "DNL net": "Dirac nodal-line net",
    "(one) NS": "One nodal surface",
    "(two) NSs": "Two nodal surfaces",
    "(three) NSs": "Three nodal surfaces",
}

PATH_PARTICLE_ALIASES = {
    "C-1 WP": ("C-1 WP",),
    "C-2 WP": ("C-2 WP",),
    "C-3 WP": ("C-3 WP",),
    "C-2 DP": ("C-2 DP",),
    "QDP": ("QDP",),
    "QTP": ("QTP",),
    "DP": ("DP",),
    "TP": ("TP",),
    "P-WNLs": ("WNL", "WNL net"),
    "P-WNL": ("WNL",),
    "P-DNLs": ("DNL", "DNL net"),
    "P-DNL": ("DNL",),
}

LINE_HEADER_PATTERN = re.compile(
    r"(?m)^([^\s\d{][^;,\n]{0,7}?)\s*;\s*([^;,\s{}]{1,12})\s*;"
)
PATH_PARTICLE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9-])(?:"
    + "|".join(re.escape(alias) for alias in sorted(PATH_PARTICLE_ALIASES, key=len, reverse=True))
    + r")(?![A-Za-z0-9-])"
)


def _parse_table(page_text: str) -> dict[str, dict[str, str]]:
    sections = {"essential": {}, "accidental": {}}
    section: str | None = None
    current_species: str | None = None
    species_names = sorted(PARTICLE_NAMES, key=len, reverse=True)

    for raw_line in page_text.replace("–", "-").splitlines():
        line = " ".join(raw_line.split())
        if line.startswith("The essential degeneracies"):
            section = "essential"
            current_species = None
            continue
        if line.startswith("The accidental degeneracies"):
            section = "accidental"
            current_species = None
            continue
        if section is None:
            continue

        matched_species = next(
            (species for species in species_names if line.startswith(f"{species} ")),
            None,
        )
        if matched_species:
            current_species = matched_species
            sections[section][matched_species] = line[len(matched_species) :].strip()
        elif current_species and re.fullmatch(r"[\d,\-\s]+", line):
            sections[section][current_species] += f" {line}"

    return sections


def _expand_spacegroups(expression: str) -> set[int]:
    values: set[int] = set()
    for token in expression.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    if not values or min(values) < 1 or max(values) > 230:
        raise ValueError(f"Invalid space-group expression: {expression}")
    return values


def _parse_accidental_paths(
    reader: PdfReader,
    page_indices: range,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    page_texts = [reader.pages[index].extract_text() or "" for index in page_indices]
    page_starts: list[int] = []
    combined_parts: list[str] = []
    cursor = 0
    for text in page_texts:
        page_starts.append(cursor)
        combined_parts.append(text)
        cursor += len(text) + 1
    combined_text = "\n".join(combined_parts)

    sg_matches = list(re.finditer(r"(?m)^SG (\d+)\s*$", combined_text))
    parsed: dict[str, dict[str, list[dict[str, object]]]] = {}
    for match_index, sg_match in enumerate(sg_matches):
        number = int(sg_match.group(1))
        if not 1 <= number <= 230:
            continue
        block_end = (
            sg_matches[match_index + 1].start()
            if match_index + 1 < len(sg_matches)
            else len(combined_text)
        )
        block = combined_text[sg_match.end() : block_end]
        line_matches = list(LINE_HEADER_PATTERN.finditer(block))
        sg_paths = parsed.setdefault(str(number), {})

        for line_index, line_match in enumerate(line_matches):
            line_end = (
                line_matches[line_index + 1].start()
                if line_index + 1 < len(line_matches)
                else len(block)
            )
            line_block = block[line_match.start() : line_end]
            line_label = line_match.group(1).strip()
            path = line_match.group(2).strip()
            global_offset = sg_match.end() + line_match.start()
            page_offset_index = bisect.bisect_right(page_starts, global_offset) - 1
            pdf_page = page_indices.start + page_offset_index + 1

            for alias in set(PATH_PARTICLE_PATTERN.findall(line_block)):
                for abbreviation in PATH_PARTICLE_ALIASES[alias]:
                    path_record = {
                        "line_label": line_label,
                        "path": path,
                        "source_pdf_page": pdf_page,
                    }
                    records = sg_paths.setdefault(abbreviation, [])
                    if path_record not in records:
                        records.append(path_record)

    return parsed


def _validate_path_index(
    space_groups: dict[str, object],
    mode: str,
    path_index: dict[str, dict[str, list[dict[str, object]]]],
) -> None:
    mismatches = []
    for number in range(1, 231):
        expected = {
            item["abbreviation"]
            for item in space_groups[str(number)][mode]["accidental"]
        }
        actual = set(path_index.get(str(number), {}))
        if expected != actual:
            mismatches.append(
                f"SG {number}: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
    if mismatches:
        details = "\n".join(mismatches[:20])
        raise ValueError(f"Detailed path index does not match summary table:\n{details}")


def build_index(pdf_path: Path) -> dict[str, object]:
    reader = PdfReader(str(pdf_path))
    if len(reader.pages) < 8:
        raise ValueError(f"Supplemental PDF has only {len(reader.pages)} pages")

    tables = {
        "without_soc": _parse_table(reader.pages[6].extract_text() or ""),
        "with_soc": _parse_table(reader.pages[7].extract_text() or ""),
    }
    space_groups = {
        str(number): {
            mode: {"essential": [], "accidental": []}
            for mode in tables
        }
        for number in range(1, 231)
    }

    for mode, sections in tables.items():
        for category, species_map in sections.items():
            for abbreviation, expression in species_map.items():
                particle = {
                    "abbreviation": abbreviation,
                    "name": PARTICLE_NAMES[abbreviation],
                }
                for number in _expand_spacegroups(expression):
                    space_groups[str(number)][mode][category].append(particle)

    path_indices = {
        "without_soc": _parse_accidental_paths(reader, range(344 - 1, 509)),
        "with_soc": _parse_accidental_paths(reader, range(993 - 1, 1073)),
    }
    for mode, path_index in path_indices.items():
        for number, particle_paths in path_index.items():
            known = {
                item["abbreviation"]
                for item in space_groups[number][mode]["accidental"]
            }
            for abbreviation in particle_paths.keys() - known:
                space_groups[number][mode]["accidental"].append(
                    {
                        "abbreviation": abbreviation,
                        "name": PARTICLE_NAMES[abbreviation],
                    }
                )
        _validate_path_index(space_groups, mode, path_index)
        for number, particle_paths in path_index.items():
            space_groups[number][mode]["accidental_paths"] = particle_paths
    for number in range(1, 231):
        for mode in tables:
            space_groups[str(number)][mode].setdefault("accidental_paths", {})

    return {
        "schema_version": 1,
        "source": {
            "title": "Supplemental Material for Encyclopedia of emergent particles in three-dimensional crystals",
            "filename": pdf_path.name,
            "tables": {
                "without_soc": {"table": "S1", "pdf_page": 7},
                "with_soc": {"table": "S2", "pdf_page": 8},
            },
            "path_sections": {
                "without_soc": "S7B",
                "with_soc": "S8B",
            },
        },
        "space_groups": space_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    index = build_index(args.pdf.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Indexed 230 space groups: {args.output.resolve()}")


if __name__ == "__main__":
    main()
