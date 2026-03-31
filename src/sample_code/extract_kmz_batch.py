import csv
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = SRC_DIR / "data"
LEGACY_KMZ = SRC_DIR / "sample_code" / "dc1.kmz"
BATCH_KMZ_GLOB = "*.kmz"

OUTPUT_CSV = SRC_DIR / "sample_code" / "kmz_parcels.csv"
OUTPUT_JSON = SRC_DIR / "sample_code" / "kmz_parcels.json"
OUTPUT_DEBUG_JSON = SRC_DIR / "sample_code" / "kmz_parcels_debug.json"

STANDARD_RADIUS_DEG = 0.00045
GROUPED_RADIUS_DEG = 0.00022
CEN_SEARCH_RADIUS_DEG = 0.00035

LAND_USE_PATTERN = re.compile(
    r"^(CLN|ONT|ODT|TMD|DGT|LNC|ONT\+CLN|ONT\+LNC|ODT\+CLN|SKC|TTC)$",
    re.I,
)
INTEGER_PATTERN = re.compile(r"^\d+$")
AREA_PATTERN = re.compile(r"^\d+(?:[.,]\d+)?$")
HAMLET_PATTERN = re.compile(r"^(ấp\b.*|khu phố\b.*)$", re.I)
ORG_PATTERN = re.compile(r"(cty|công ty|tnhh|ubnd|ban quản lý)", re.I)
ROAD_STREAM_PATTERN = re.compile(r"(đường|suối|ranh|kênh|mương|sông|vườn quốc gia)", re.I)
SHEET_ONLY_PATTERN = re.compile(r"\bT[ỜƠ]\s*S[ỐÔ]\s*(\d+)\b", re.I)
SHEET_TRANSITION_PATTERN = re.compile(
    r"\bT[ỜƠ]\s*S[ỐÔ]\s*(\d+).*?CHỈNH\s*L[ÝY]\s*TH[ÀA]NH\s*T[ỜƠ]\s*S[ỐÔ]\s*(\d+)",
    re.I,
)
PARCEL_TOTAL_PATTERN = re.compile(r"^Tổng\s+số\s+thửa:\s*(\d+)$", re.I)
SPLIT_SHEET_COUNT_PATTERN = re.compile(r"^THÀNH\s+(\d+)\s+T[ỜƠ]$", re.I)
PAREN_LAND_AREA_PATTERN = re.compile(
    r"^\(?(?P<land_use>CLN|ONT|ODT|TMD|DGT|LNC|ONT\+CLN|ONT\+LNC|ODT\+CLN|SKC|TTC)\s*[:)]\s*(?P<area>\d+(?:[.,]\d+)?)",
    re.I,
)
LOAIDAT_DOC_PATTERN = re.compile(
    r"^(?P<land_use>CLN|ONT|ODT|TMD|DGT|LNC|ONT\+CLN|ONT\+LNC|ODT\+CLN|SKC|TTC)\s*:",
    re.I,
)
NOTE_PATTERN = re.compile(
    r"(trùng thửa|dacapgcn|đã cấp|cấp giấy|mất gcn|thu hồi|thanh tra|qđ-|qđ|đã tl|phục đo|lý do|hủy bỏ)",
    re.I,
)
TRUNG_THUA_PATTERN = re.compile(r"^trùng thửa(?: số \d+)?$", re.I)


def repair_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    text = text.replace("\r", " ").replace("\n", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_layer_label(value: str | None) -> str | None:
    text = repair_text(value)
    if not text:
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\bELEVATION\b\s*=\s*.*$", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" -:")
    return text or None


def child_text(element: ET.Element, suffix: str) -> str | None:
    for child in element:
        if child.tag.endswith(suffix):
            return child.text
    return None


def find_descendant_text(element: ET.Element, suffix: str) -> str | None:
    for descendant in element.iter():
        if descendant.tag.endswith(suffix):
            return descendant.text
    return None


def parse_point_coordinates(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    token = text.strip().split()[0]
    parts = token.split(",")
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def parse_kmz_points(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("doc.kml"))

    points: list[dict] = []
    order = 0
    for placemark in root.iter():
        if not placemark.tag.endswith("Placemark"):
            continue
        coords_text = None
        for descendant in placemark.iter():
            if descendant.tag.endswith("Point"):
                coords_text = find_descendant_text(descendant, "coordinates")
                break
        coords = parse_point_coordinates(coords_text)
        if coords is None:
            continue
        points.append(
            {
                "name": repair_text(child_text(placemark, "name")),
                "description": repair_text(child_text(placemark, "description")),
                "layer": normalize_layer_label(child_text(placemark, "description")),
                "x": coords[0],
                "y": coords[1],
                "order": order,
            }
        )
        order += 1
    return points


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_area_value(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def parse_land_use_and_area(text: str | None) -> tuple[str | None, float | None]:
    if not text:
        return None, None
    match = PAREN_LAND_AREA_PATTERN.search(text)
    if match:
        land_use = match.group("land_use").upper()
        area = float(match.group("area").replace(",", "."))
        return land_use, area
    match = LOAIDAT_DOC_PATTERN.search(text)
    if match:
        return match.group("land_use").upper(), None
    return None, None


def normalize_sheet_number(value: str) -> str:
    stripped = value.lstrip("0")
    return stripped or "0"


def extract_metadata(points: list[dict]) -> dict:
    metadata_points = [point for point in points if point["layer"] in {"Level 63", "Level 10"}]
    labels = [point["name"] for point in metadata_points if point["name"]]

    explicit_pair = None
    explicit_pair_label = None
    for label in labels:
        match = SHEET_TRANSITION_PATTERN.search(label)
        if match:
            explicit_pair = (
                normalize_sheet_number(match.group(1)),
                normalize_sheet_number(match.group(2)),
            )
            explicit_pair_label = label
            break

    ordered_sheets: list[str] = []
    for label in labels:
        for match in SHEET_ONLY_PATTERN.finditer(label):
            sheet = normalize_sheet_number(match.group(1))
            if sheet not in ordered_sheets:
                ordered_sheets.append(sheet)

    parcel_total_labels = []
    parcel_total_values = []
    split_sheet_count_labels = []
    split_sheet_count_values = []
    address_label = explicit_pair_label

    for label in labels:
        parcel_total_match = PARCEL_TOTAL_PATTERN.fullmatch(label)
        if parcel_total_match:
            parcel_total_labels.append(label)
            parcel_total_values.append(int(parcel_total_match.group(1)))

        split_match = SPLIT_SHEET_COUNT_PATTERN.fullmatch(label)
        if split_match:
            split_sheet_count_labels.append(label)
            split_sheet_count_values.append(int(split_match.group(1)))

        if address_label is None and SHEET_TRANSITION_PATTERN.search(label):
            address_label = label

    has_before = any(label and "TRƯỚC CHỈNH LÝ" in label.upper() for label in labels)
    has_after = any(label and "SAU CHỈNH LÝ" in label.upper() for label in labels)
    if has_before and has_after:
        adjustment_stage = "mixed_adjustment"
    elif has_after:
        adjustment_stage = "after_adjustment"
    elif has_before:
        adjustment_stage = "before_adjustment"
    else:
        adjustment_stage = "unknown"

    if explicit_pair:
        sheet_before, sheet_after = explicit_pair
    elif len(ordered_sheets) >= 2:
        sheet_before, sheet_after = ordered_sheets[0], ordered_sheets[1]
    elif len(ordered_sheets) == 1:
        sheet_before = ordered_sheets[0]
        sheet_after = ordered_sheets[0]
    else:
        sheet_before = None
        sheet_after = None

    parcel_total_before = parcel_total_values[0] if parcel_total_values else None
    parcel_total_after = parcel_total_values[1] if len(parcel_total_values) >= 2 else None
    split_sheet_count_before = split_sheet_count_values[0] if split_sheet_count_values else None
    split_sheet_count_after = split_sheet_count_values[1] if len(split_sheet_count_values) >= 2 else None

    return {
        "sheet_number_before": sheet_before,
        "sheet_number_after": sheet_after,
        "adjustment_stage": adjustment_stage,
        "parcel_total_before": parcel_total_before,
        "parcel_total_after": parcel_total_after,
        "split_sheet_count_before": split_sheet_count_before,
        "split_sheet_count_after": split_sheet_count_after,
        "parcel_total_labels": parcel_total_labels,
        "parcel_total_values": parcel_total_values,
        "split_sheet_count_labels": split_sheet_count_labels,
        "split_sheet_count_values": split_sheet_count_values,
        "address_label": address_label,
        "metadata_labels": labels[:50],
    }


def detect_profile(points: list[dict]) -> str:
    counts = Counter(point["layer"] or "" for point in points)
    if counts["Level 13"] >= 20 and counts["Level 2"] >= 20 and counts["Level 4"] >= 20:
        return "standard"
    if counts["Level 49"] >= 20 and counts["Level 49"] >= counts["Level 13"] * 2:
        return "grouped_level49"
    if counts["Level 13"] > 0:
        return "standard"
    return "unsupported"


def choose_first(items: list[tuple[float, dict]]) -> tuple[str | None, float | None]:
    if not items:
        return None, None
    dist, item = min(items, key=lambda pair: pair[0])
    return item["name"], dist


def choose_item(items: list[tuple[float, dict]]) -> tuple[dict | None, float | None]:
    if not items:
        return None, None
    dist, item = min(items, key=lambda pair: pair[0])
    return item, dist


def choose_owner(items: list[tuple[float, dict]]) -> tuple[str | None, float | None]:
    filtered = []
    for dist, item in items:
        name = item["name"]
        if not name:
            continue
        if ROAD_STREAM_PATTERN.search(name):
            continue
        if NOTE_PATTERN.search(name):
            continue
        if INTEGER_PATTERN.fullmatch(name):
            continue
        if LAND_USE_PATTERN.fullmatch(name):
            continue
        if HAMLET_PATTERN.search(name):
            continue
        if len(name.split()) == 1 and not ORG_PATTERN.search(name) and "&" not in name:
            continue
        filtered.append((dist, item))
    return choose_first(filtered)


def choose_note(items: list[tuple[float, dict]], parcel_number: str) -> tuple[str | None, float | None]:
    if not items:
        return None, None

    direct_matches = []
    for dist, item in items:
        name = item["name"]
        if name and re.search(rf"\b{re.escape(parcel_number)}\b", name):
            direct_matches.append((dist, item))
    if direct_matches:
        return choose_first(direct_matches)

    trung_thua = [
        (dist, item)
        for dist, item in items
        if item["name"] and TRUNG_THUA_PATTERN.search(item["name"])
    ]
    if trung_thua:
        name, dist = choose_first(trung_thua)
        if name and "số" not in name.lower():
            return f"Trùng thửa số {parcel_number}", dist
        return name, dist

    return choose_first(items)


def nearby_points(anchor: dict, points: list[dict], radius: float) -> list[tuple[float, dict]]:
    anchor_xy = (anchor["x"], anchor["y"])
    nearby = []
    for point in points:
        if point is anchor:
            continue
        dist = distance(anchor_xy, (point["x"], point["y"]))
        if dist <= radius:
            nearby.append((dist, point))
    return nearby


def grouped_numeric_role(name: str) -> str | None:
    if not AREA_PATTERN.fullmatch(name):
        return None
    if "," in name or "." in name:
        return "area"
    value = int(name)
    if value == 0:
        return None
    if value <= 999:
        return "parcel"
    return "area"


def build_candidate(
    source_file: str,
    anchor: dict,
    points: list[dict],
    metadata: dict,
    profile: str,
    radius: float,
    area_layer: str,
    land_use_layer: str,
) -> dict:
    nearby = nearby_points(anchor, points, radius)
    center_points = [
        (distance((anchor["x"], anchor["y"]), (point["x"], point["y"])), point)
        for point in points
        if point["layer"] == "CEN"
        and distance((anchor["x"], anchor["y"]), (point["x"], point["y"])) <= CEN_SEARCH_RADIUS_DEG
    ]
    area_candidates = []
    land_use_candidates = []
    loaidat_candidates = []
    hamlet_candidates = []
    owner_candidates = []
    note_candidates = []

    for dist, point in nearby:
        name = point["name"]
        desc = point["layer"]
        if not name:
            continue

        if desc == area_layer and AREA_PATTERN.fullmatch(name) and name != "0":
            if area_layer != "Level 49" or grouped_numeric_role(name) == "area":
                area_candidates.append((dist, point))

        if desc == land_use_layer and LAND_USE_PATTERN.fullmatch(name):
            land_use_candidates.append((dist, point))

        if desc == "LOAIDAT":
            loaidat_candidates.append((dist, point))

        if desc in {"Level 56", "HIENTRANG"} and HAMLET_PATTERN.search(name):
            hamlet_candidates.append((dist, point))

        if desc in {"Level 53", "Level 3"}:
            owner_candidates.append((dist, point))

        if desc in {"KIEMTRA", "GHICHU", "TTGCN", "Level 3", "RANHTCTA", "RanhTCTA"} and NOTE_PATTERN.search(name):
            note_candidates.append((dist, point))

    sqm2_raw = None
    sqm2 = None
    area_item, area_dist = choose_item(area_candidates)
    if area_item is not None:
        sqm2_raw = area_item["name"]
        sqm2 = parse_area_value(sqm2_raw)

    land_use_type, land_use_dist = choose_first(land_use_candidates)
    hamlet_name, hamlet_dist = choose_first(hamlet_candidates)
    owner_name, owner_dist = choose_owner(owner_candidates)
    parcel_note, note_dist = choose_note(note_candidates, anchor["name"])

    parsed_loaidat = []
    for dist, item in loaidat_candidates:
        parsed_land_use, parsed_area = parse_land_use_and_area(item["name"])
        if parsed_land_use or parsed_area is not None:
            parsed_loaidat.append((dist, item["name"], parsed_land_use, parsed_area))

    if land_use_type is None and parsed_loaidat:
        dist, _raw_text, parsed_land_use, _parsed_area = min(parsed_loaidat, key=lambda row: row[0])
        land_use_type = parsed_land_use
        land_use_dist = dist

    if sqm2 is None and parsed_loaidat:
        dist, raw_text, _parsed_land_use, parsed_area = min(parsed_loaidat, key=lambda row: row[0])
        if parsed_area is not None:
            sqm2 = parsed_area
            sqm2_raw = raw_text
            area_dist = dist

    center_item, center_dist = choose_item(center_points)
    if center_item is not None:
        latitude = center_item["y"]
        longitude = center_item["x"]
        anchor_rule = "parcel anchor with nearest CEN representative point"
    else:
        latitude = anchor["y"]
        longitude = anchor["x"]
        center_dist = None
        anchor_rule = "parcel anchor fallback to parcel label point"

    nearby_preview = [
        {
            "distance_deg": round(dist, 7),
            "name": item["name"],
            "description": item["description"],
        }
        for dist, item in sorted(nearby, key=lambda pair: pair[0])[:20]
    ]

    return {
        "source_file": source_file,
        "latitude": latitude,
        "longitude": longitude,
        "sheet_number_before": metadata["sheet_number_before"],
        "sheet_number_after": metadata["sheet_number_after"],
        "adjustment_stage": metadata["adjustment_stage"],
        "parcel_total_before": metadata["parcel_total_before"],
        "parcel_total_after": metadata["parcel_total_after"],
        "split_sheet_count_before": metadata["split_sheet_count_before"],
        "split_sheet_count_after": metadata["split_sheet_count_after"],
        "address_label": metadata["address_label"],
        "parcel_number": anchor["name"],
        "land_use_type": land_use_type,
        "sqm2": sqm2,
        "hamlet_name": hamlet_name,
        "owner_name": owner_name,
        "parcel_note": parcel_note,
        "profile": profile,
        "candidate_id": None,
        "sqm2_raw": sqm2_raw,
        "match_distance_land_use": land_use_dist,
        "match_distance_area": area_dist,
        "match_distance_hamlet": hamlet_dist,
        "match_distance_owner": owner_dist,
        "match_distance_note": note_dist,
        "match_distance_center": center_dist,
        "anchor_rule": anchor_rule,
        "nearby_preview": nearby_preview,
    }


def extract_standard_candidates(source_file: str, points: list[dict], metadata: dict) -> list[dict]:
    anchors = [
        point
        for point in points
        if point["layer"] == "Level 13"
        and point["name"]
        and INTEGER_PATTERN.fullmatch(point["name"])
        and point["name"] != "0"
    ]
    candidates = [
        build_candidate(
            source_file=source_file,
            anchor=anchor,
            points=points,
            metadata=metadata,
            profile="standard",
            radius=STANDARD_RADIUS_DEG,
            area_layer="Level 4",
            land_use_layer="Level 2",
        )
        for anchor in anchors
    ]
    return candidates


def extract_grouped_level49_candidates(source_file: str, points: list[dict], metadata: dict) -> list[dict]:
    anchors = [
        point
        for point in points
        if point["layer"] == "Level 49"
        and point["name"]
        and grouped_numeric_role(point["name"]) == "parcel"
    ]
    candidates = []
    for anchor in anchors:
        candidate = build_candidate(
            source_file=source_file,
            anchor=anchor,
            points=points,
            metadata=metadata,
            profile="grouped_level49",
            radius=GROUPED_RADIUS_DEG,
            area_layer="Level 49",
            land_use_layer="Level 49",
        )
        if candidate["land_use_type"] or candidate["sqm2"] is not None:
            candidates.append(candidate)
    return candidates


def input_kmz_paths() -> list[Path]:
    if DATA_DIR.exists():
        data_paths = sorted(path for path in DATA_DIR.glob(BATCH_KMZ_GLOB) if path.is_file())
        if data_paths:
            return data_paths

    paths = []
    if LEGACY_KMZ.exists():
        paths.append(LEGACY_KMZ)
    paths.extend(sorted(path for path in SRC_DIR.glob(BATCH_KMZ_GLOB) if path.is_file()))
    return paths


def extract_all() -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    debug: list[dict] = []

    for kmz_path in input_kmz_paths():
        try:
            points = parse_kmz_points(kmz_path)
        except ET.ParseError as exc:
            debug.append(
                {
                    "source_file": kmz_path.name,
                    "profile": "parse_error",
                    "point_count": 0,
                    "candidate_count": 0,
                    "sheet_number_before": None,
                    "sheet_number_after": None,
                    "adjustment_stage": "unknown",
                    "metadata_labels": [],
                    "error": str(exc),
                    "candidates": [],
                }
            )
            continue
        except Exception as exc:
            debug.append(
                {
                    "source_file": kmz_path.name,
                    "profile": "read_error",
                    "point_count": 0,
                    "candidate_count": 0,
                    "sheet_number_before": None,
                    "sheet_number_after": None,
                    "adjustment_stage": "unknown",
                    "metadata_labels": [],
                    "error": f"{type(exc).__name__}: {exc}",
                    "candidates": [],
                }
            )
            continue

        metadata = extract_metadata(points)
        profile = detect_profile(points)

        if profile == "standard":
            file_candidates = extract_standard_candidates(kmz_path.name, points, metadata)
        elif profile == "grouped_level49":
            file_candidates = extract_grouped_level49_candidates(kmz_path.name, points, metadata)
        else:
            file_candidates = []

        for idx, candidate in enumerate(file_candidates, start=1):
            candidate["candidate_id"] = idx
            clean_row = {
                "source_file": candidate["source_file"],
                "latitude": candidate["latitude"],
                "longitude": candidate["longitude"],
                "sheet_number_before": candidate["sheet_number_before"],
                "sheet_number_after": candidate["sheet_number_after"],
                "adjustment_stage": candidate["adjustment_stage"],
                "parcel_total_before": candidate["parcel_total_before"],
                "parcel_total_after": candidate["parcel_total_after"],
                "split_sheet_count_before": candidate["split_sheet_count_before"],
                "split_sheet_count_after": candidate["split_sheet_count_after"],
                "address_label": candidate["address_label"],
                "parcel_number": candidate["parcel_number"],
                "land_use_type": candidate["land_use_type"],
                "sqm2": candidate["sqm2"],
                "hamlet_name": candidate["hamlet_name"],
                "owner_name": candidate["owner_name"],
                "parcel_note": candidate["parcel_note"],
            }
            candidates.append(clean_row)

        debug.append(
            {
                "source_file": kmz_path.name,
                "profile": profile,
                "point_count": len(points),
                "candidate_count": len(file_candidates),
                "sheet_number_before": metadata["sheet_number_before"],
                "sheet_number_after": metadata["sheet_number_after"],
                "adjustment_stage": metadata["adjustment_stage"],
                "parcel_total_before": metadata["parcel_total_before"],
                "parcel_total_after": metadata["parcel_total_after"],
                "split_sheet_count_before": metadata["split_sheet_count_before"],
                "split_sheet_count_after": metadata["split_sheet_count_after"],
                "parcel_total_labels": metadata["parcel_total_labels"],
                "parcel_total_values": metadata["parcel_total_values"],
                "split_sheet_count_labels": metadata["split_sheet_count_labels"],
                "split_sheet_count_values": metadata["split_sheet_count_values"],
                "address_label": metadata["address_label"],
                "metadata_labels": metadata["metadata_labels"],
                "error": None,
                "candidates": file_candidates,
            }
        )

    return candidates, debug


def write_outputs(candidates: list[dict], debug: list[dict]) -> None:
    fieldnames = [
        "source_file",
        "latitude",
        "longitude",
        "sheet_number_before",
        "sheet_number_after",
        "adjustment_stage",
        "parcel_total_before",
        "parcel_total_after",
        "split_sheet_count_before",
        "split_sheet_count_after",
        "address_label",
        "parcel_number",
        "land_use_type",
        "sqm2",
        "hamlet_name",
        "owner_name",
        "parcel_note",
    ]

    OUTPUT_JSON.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_DEBUG_JSON.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")

    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            writer.writerow(row)


def main() -> None:
    candidates, debug = extract_all()
    write_outputs(candidates, debug)
    print(OUTPUT_CSV)
    print(OUTPUT_JSON)
    print(OUTPUT_DEBUG_JSON)
    print(f"candidate_count={len(candidates)}")


if __name__ == "__main__":
    main()
