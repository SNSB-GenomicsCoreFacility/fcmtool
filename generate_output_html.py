#!/usr/bin/env python3
"""
Generate per-species HTML reports with a two-row (H1, H2) layout using Jinja2.
H1: up to 5 columns that fill available width; each column shows a figure + a per-region metrics table
    including Peak_name, Count, Mean, CV%, Ratio, 2C_size_in_pg, remarks.
H2: contains two tables:
    - Table 1: per-individual metadata.
    - Table 2: species-level genome size stats (per spec).

Validates presence of <base>.png, <base>.json, and <base>.xml for each sample.

Usage:
    python generate_output_html.py <data_folder_path> <cluster_csv_path> <config_yaml_path>

The YAML file must include:
  - Standard_Taxon
  - Standard_genome_size_2C
  - Standard_genome_size_1C
  - Stain
"""

import argparse
import base64
import math
import sys
from pathlib import Path
from datetime import datetime
import re
from statistics import mean, pstdev, stdev

import pandas as pd
from jinja2 import Template
import yaml


HTML_TEMPLATE = Template(r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Species: {{ species }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root { --cols: {{ cols }}; }
    * { box-sizing: border-box; }
    body { font-family: Arial, Helvetica, sans-serif; margin: 24px; background: #fafafa; }
    h1 { margin: 0 0 12px 0; }
    .h-section { margin-bottom: 24px; }
    /* H1 grid: number of columns equals number of samples (max 5) */
    #H1.grid {
      display: grid;
      grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
      gap: 16px;
      align-items: start;
    }
    .col {
      display: flex;
      flex-direction: column;
      gap: 10px;
      border: 1px solid #e1e1e1;
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
      background: #fff;
    }
    figure { margin: 0; }
    figure img {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
    }
    figcaption { margin-top: 6px; font-size: 14px; color: #333; }
    /* Table styling */
    .metrics-table, .h2-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    .metrics-table th, .metrics-table td,
    .h2-table th, .h2-table td {
      border: 1px solid #ddd;
      padding: 6px 8px;
      text-align: right;
      vertical-align: top;
    }
    .metrics-table th:first-child, .metrics-table td:first-child,
    .h2-table th:first-child, .h2-table td:first-child {
      text-align: left;
    }
    .metrics-table thead th {
      background: #f7f7f7;
      position: sticky;
      top: 0;
    }
    .metrics-wrap {
      max-height: 280px;
      overflow: auto;
      border: 1px solid #eee;
      border-radius: 8px;
    }
    tr.emph td { font-weight: 700; }
    /* H2 area */
    #H2 {
      border: 1px solid #e1e1e1;
      border-radius: 12px;
      padding: 16px;
      color: #555;
      min-height: 80px;
      background: #fff;
    }
    #H2 h2 { margin-top: 0; font-size: 18px; }
    .h2-block { margin-bottom: 16px; }
  </style>
</head>
<body>
  <h1>Species: {{ species }}</h1>

  <!-- H1: up to 5 columns; widths expand based on sample count -->
  <div id="H1" class="h-section grid">
    {% for item in items %}
      <div class="col">
        <figure>
          <img src="data:image/png;base64,{{ item.png_b64 }}" alt="{{ item.sample_name|e }}">
          <figcaption>{{ item.sample_name }}</figcaption>
        </figure>
        <div class="metrics-wrap">
          <table class="metrics-table">
            <thead>
              <tr>
                <th>Region No.</th>
                <th>Peak_name</th>
                <th>Count</th>
                <th>Mean</th>
                <th>CV%</th>
                <th>Ratio</th>
                <th>2C_size_in_pg</th>
                <th>remarks</th>
              </tr>
            </thead>
            <tbody>
              {% for r in item.region_rows %}
              <tr class="{{ 'emph' if r.is_standard_g1 else '' }}">
                <td>{{ r.region_no }}</td>
                <td>{{ r.peak_name }}</td>
                <td>{{ r.count }}</td>
                <td>{{ r.mean }}</td>
                <td>{{ r.cv_percent }}</td>
                <td>{{ r.ratio }}</td>
                <td>{{ r.size_2c_pg }}</td>
                <td>{{ r.remarks }}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    {% endfor %}
  </div>

  <!-- H2: two tables -->
  <div id="H2" class="h-section">
    <div class="h2-block">
      <h2>Table 1</h2>
      <table class="h2-table">
        <thead>
          <tr>
            <th>File_name</th>
            <th>Taxon</th>
            <th>Stain</th>
            <th>Individual</th>
            <th>Standard_Taxon</th>
            <th>Standard_region_number</th>
            <th>Standard Genome Size 2C in pg</th>
          </tr>
        </thead>
        <tbody>
          {% for row in h2_table1 %}
          <tr>
            <td>{{ row.file_name }}</td>
            <td>{{ row.taxon }}</td>
            <td>{{ row.stain }}</td>
            <td>{{ row.individual }}</td>
            <td>{{ row.standard_taxon }}</td>
            <td>{{ row.standard_region_number }}</td>
            <td>{{ row.standard_genome_size_2c }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="h2-block">
      <h2>Table 2</h2>
      <table class="h2-table">
        <thead>
          <tr>
            <th>Taxon</th>
            <th>Average_Genome_Size_(2C_in_pg)</th>
            <th>Standard_Deviation</th>
            <th>Average_Genome_Size_(1C_in_pg)</th>
            <th>Average_Genome_Size_(1C_in_Mbps)</th>
            <th>Average_Genome_Size_(2C_in_Mbps)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>{{ h2_table2.taxon }}</td>
            <td>{{ h2_table2.avg_2c_pg }}</td>
            <td>{{ h2_table2.std_2c_pg }}</td>
            <td>{{ h2_table2.avg_1c_pg }}</td>
            <td>{{ h2_table2.avg_1c_mbps }}</td>
            <td>{{ h2_table2.avg_2c_mbps }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
""")


def find_data_csv(folder: Path, cluster_csv_path: Path) -> Path:
    csvs = list(folder.glob("*.csv"))
    csvs = [p for p in csvs if p.resolve() != cluster_csv_path.resolve()]
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {folder} (after excluding the cluster CSV).")
    data_like = [p for p in csvs if "data" in p.name.lower()]
    if len(data_like) == 1:
        return data_like[0]
    if len(csvs) == 1:
        return csvs[0]
    raise RuntimeError(
        "Ambiguous data CSV selection. Found multiple CSVs:\n"
        + "\n".join(f"  - {p}" for p in csvs)
        + "\nPlease ensure there is exactly one data CSV in the folder (ideally named with 'data')."
    )


def parse_and_format_timestamp(ts: str) -> str:
    ts = ts.strip()
    try:
        dt = datetime.strptime(ts, "%d/%m/%Y %H:%M:%S")
    except ValueError as e:
        raise ValueError(
            f"Failed to parse ExportingUserTimestamp '{ts}'. Expected format 'DD/MM/YYYY HH:MM:SS'."
        ) from e
    return dt.strftime("%Y%m%d-%H%M%S")


def build_png_filename(exporting_ts: str, session_id, sample_id) -> str:
    def normalize_id(x):
        try:
            fx = float(x)
            if fx.is_integer():
                return str(int(fx))
            return str(x)
        except Exception:
            return str(x)

    sid = normalize_id(session_id)
    smp = normalize_id(sample_id)
    return f"{exporting_ts}_{sid}_{smp}.png"


def read_cluster_csv(cluster_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cluster_csv)
    if df.shape[1] < 2:
        raise ValueError("Cluster CSV must have at least two columns: SampleName and Species.")
    df.columns = [c.strip() for c in df.columns]
    first_col = df.columns[0]
    if first_col != "SampleName":
        if "SampleName" in df.columns:
            df = df[["SampleName"] + [c for c in df.columns if c != "SampleName"]]
        else:
            df = df.rename(columns={first_col: "SampleName"})
    second_col = [c for c in df.columns if c != "SampleName"][0]
    df = df.rename(columns={second_col: "Species"})
    df = df[["SampleName", "Species"]].copy().dropna(subset=["SampleName", "Species"])
    return df


def read_data_csv(data_csv: Path) -> pd.DataFrame:
    """
    Read the CSV and normalize the four required columns while keeping all others intact.
    """
    df = pd.read_csv(data_csv, sep=";")
    needed = ["SampleName", "SampleID", "SessionID", "ExportingUserTimestamp"]
    norm_map = {c.lower().replace(" ", ""): c for c in df.columns}
    resolved = {}
    for k in needed:
        key = k.lower().replace(" ", "")
        if key not in norm_map:
            raise KeyError(
                f"Required column '{k}' not found in data CSV. Available columns: {list(df.columns)}"
            )
        resolved[k] = norm_map[key]
    df = df.copy().rename(columns={resolved[k]: k for k in needed})
    return df


_region_re = re.compile(r"^Region(\d+)_(Number|NumberOfParticles|Mean|CVPercent|Ratio)$", re.IGNORECASE)

def detect_region_indices(columns) -> list[int]:
    indices = set()
    for c in columns:
        m = _region_re.match(c)
        if m:
            idx, kind = m.groups()
            if kind.lower() == "number":
                try:
                    indices.add(int(idx))
                except ValueError:
                    pass
    return sorted(indices)


def fmt_val(x):
    if x is None:
        return ""
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return ""
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f"{f:.6g}"
    except Exception:
        return str(x)


def parse_float_or_none(x):
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except Exception:
        return None


def assign_peak_names(region_rows, species_name: str, standard_taxon: str):
    """
    Modify region_rows in place to add:
      - peak_name (string)
      - is_standard_g1 (bool) True only for the region with Ratio in [0.99, 1.01]
    """
    by_idx = {}
    for r in region_rows:
        try:
            k = int(r["region_no"])
        except Exception:
            continue
        by_idx[k] = r

    std_idx = None
    for idx in sorted(by_idx.keys()):
        ratio = parse_float_or_none(by_idx[idx].get("ratio"))
        if ratio is not None and 0.99 <= ratio <= 1.01:
            std_idx = idx
            break

    used = set()
    if std_idx is not None:
        r = by_idx.get(std_idx)
        if r is not None:
            r["peak_name"] = f"{standard_taxon}_G1"
            r["is_standard_g1"] = True
            used.add(std_idx)

        alt_candidates = [i for i in [1, 2] if i in by_idx and i != std_idx] or [i for i in sorted(by_idx.keys()) if i != std_idx]
        if alt_candidates:
            alt_idx = alt_candidates[0]
            r2 = by_idx[alt_idx]
            r2["peak_name"] = f"{species_name}_G1"
            r2["is_standard_g1"] = False
            used.add(alt_idx)

    remaining = [i for i in sorted(by_idx.keys()) if i not in used]
    toggle = True
    for idx in remaining:
        r = by_idx[idx]
        if idx <= 4:
            r["peak_name"] = f"{standard_taxon}_G2" if toggle else f"{species_name}_G2"
            toggle = not toggle
        else:
            r["peak_name"] = f"{species_name}_polyploid_?"
        r["is_standard_g1"] = r.get("is_standard_g1", False)

    for r in region_rows:
        r.setdefault("peak_name", "")
        r.setdefault("is_standard_g1", False)


def extract_region_rows(row: pd.Series, region_indices: list[int], species_name: str,
                        standard_taxon: str, standard_genome_size_2c: float) -> list[dict]:
    out = []

    def get_col(idx, prefix):
        for key in (
            f"Region{idx}_{prefix}",
            f"region{idx}_{prefix}",
            f"REGION{idx}_{prefix}",
        ):
            if key in row:
                return row.get(key)
        return None

    for idx in region_indices:
        number = get_col(idx, "Number")
        count  = get_col(idx, "NumberOfParticles")
        mean   = get_col(idx, "Mean")
        cvp    = get_col(idx, "CVPercent")
        ratio  = get_col(idx, "Ratio")

        parsed_no = parse_float_or_none(number)
        region_no = int(round(parsed_no)) if parsed_no is not None else idx

        ratio_val = parse_float_or_none(ratio)
        size_2c_pg = ""
        if ratio_val is not None:
            size_2c_pg = fmt_val(ratio_val * float(standard_genome_size_2c))

        out.append({
            "region_no": region_no,
            "count": fmt_val(count),
            "mean": fmt_val(mean),
            "cv_percent": fmt_val(cvp),
            "ratio": fmt_val(ratio),
            "size_2c_pg": size_2c_pg,
            "remarks": ""
        })

    assign_peak_names(out, species_name=species_name, standard_taxon=standard_taxon)
    return out


def ensure_artifacts_exist(rows: pd.DataFrame) -> None:
    missing_entries = []
    for _, r in rows.iterrows():
        base = r["PNGPath"].with_suffix("")
        expected = {
            "png": base.with_suffix(".png"),
            "json": base.with_suffix(".json"),
            "xml": base.with_suffix(".xml"),
        }
        for kind, path in expected.items():
            if not path.is_file():
                missing_entries.append((r["SampleName"], kind, str(path)))

    if missing_entries:
        lines = ["FOLDER VALIDATION FAILED: Some expected files are missing:"]
        for sample, kind, path in missing_entries:
            lines.append(f"  - Sample '{sample}': missing {kind.upper()} at {path}")
        raise FileNotFoundError("\n".join(lines))


def encode_png_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        b = f.read()
    return base64.b64encode(b).decode("ascii")


def find_standard_region_number(region_rows: list[dict]) -> str:
    for r in region_rows:
        ratio = parse_float_or_none(r.get("ratio"))
        if ratio is not None and 0.99 <= ratio <= 1.01:
            try:
                return str(int(r.get("region_no")))
            except Exception:
                return str(r.get("region_no"))
    return ""


def make_species_html(species: str, species_rows: pd.DataFrame, out_dir: Path, region_indices: list[int],
                      standard_taxon: str, standard_genome_size_2c: float, standard_genome_size_1c: float, stain: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_species = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in species)
    html_path = out_dir / f"{safe_species}.html"

    items = []
    h2_table1 = []
    species_rows_sorted = species_rows.sort_values("SampleName").reset_index(drop=True)

    # Collect per-sample species G1 2C sizes for Table 2 stats
    g1_2c_values = []

    for i, r in species_rows_sorted.iterrows():
        region_rows = extract_region_rows(
            r, region_indices,
            species_name=r["Species"],
            standard_taxon=standard_taxon,
            standard_genome_size_2c=standard_genome_size_2c
        )
        items.append({
            "sample_name": r["SampleName"],
            "png_b64": encode_png_to_base64(r["PNGPath"]),
            "region_rows": region_rows
        })

        # H2 Table 1 row
        fcs_path = r.get("FCSFilePath", "")
        file_name = Path(str(fcs_path)).name if str(fcs_path) not in ("", "nan", "None") else ""
        std_region = find_standard_region_number(region_rows)
        h2_table1.append({
            "file_name": file_name,
            "taxon": r["Species"],
            "stain": stain,
            "individual": str(i + 1),
            "standard_taxon": standard_taxon,
            "standard_region_number": std_region,
            "standard_genome_size_2c": fmt_val(standard_genome_size_2c)
        })

        # collect species G1 2C_size_in_pg for Table 2
        target_peak = f"{r['Species']}_G1"
        for rr in region_rows:
            if rr.get("peak_name") == target_peak:
                val = parse_float_or_none(rr.get("size_2c_pg"))
                if val is not None:
                    g1_2c_values.append(val)

    # Limit to 5 columns in H1
    if len(items) > 5:
        items = items[:5]

    cols = max(1, min(5, len(items)))

    # Compute H2 Table 2 stats
    if g1_2c_values:
        avg_2c_pg = mean(g1_2c_values)
        # Use sample standard deviation if >=2, else 0
        if len(g1_2c_values) >= 2:
            std_2c_pg = stdev(g1_2c_values)
        else:
            std_2c_pg = 0.0
        avg_1c_pg = avg_2c_pg / 2.0
        avg_1c_mbps = avg_1c_pg * float(standard_genome_size_1c)
        avg_2c_mbps = 2.0 * avg_1c_mbps
    else:
        avg_2c_pg = std_2c_pg = avg_1c_pg = avg_1c_mbps = avg_2c_mbps = None

    h2_table2 = {
        "taxon": species,
        "avg_2c_pg": fmt_val(avg_2c_pg),
        "std_2c_pg": fmt_val(std_2c_pg),
        "avg_1c_pg": fmt_val(avg_1c_pg),
        "avg_1c_mbps": fmt_val(avg_1c_mbps),
        "avg_2c_mbps": fmt_val(avg_2c_mbps),
    }

    html = HTML_TEMPLATE.render(
        species=species,
        items=items,
        cols=cols,
        h2_table1=h2_table1,
        h2_table2=h2_table2
    )
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Build per-species HTML with H1/H2 layout (Jinja2) including per-region metrics and H2 Tables; validates PNG/JSON/XML triplets.")
    parser.add_argument("data_folder_path", type=Path, help="Path to folder containing the data CSV and PNG/JSON/XML files")
    parser.add_argument("cluster_csv_path", type=Path, help="Path to cluster CSV with SampleName and Species")
    parser.add_argument("config_yaml_path", type=Path, help="Path to YAML with Standard_Taxon, genome size keys, and Stain")
    args = parser.parse_args()

    folder: Path = args.data_folder_path
    cluster_csv_path: Path = args.cluster_csv_path
    config_yaml_path: Path = args.config_yaml_path

    if not folder.is_dir():
        print(f"ERROR: '{folder}' is not a directory.", file=sys.stderr)
        sys.exit(2)
    if not cluster_csv_path.is_file():
        print(f"ERROR: Cluster CSV '{cluster_csv_path}' not found.", file=sys.stderr)
        sys.exit(2)
    if not config_yaml_path.is_file():
        print(f"ERROR: YAML config '{config_yaml_path}' not found.", file=sys.stderr)
        sys.exit(2)

    # Load YAML
    try:
        with open(config_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        standard_taxon = str(cfg.get("Standard_Taxon", "")).strip()
        if not standard_taxon:
            raise KeyError("Key 'Standard_Taxon' missing or empty in YAML.")
        if "Standard_genome_size_2C" not in cfg:
            raise KeyError("Keys 'Standard_genome_size_2C' must exist in YAML.")
        stain = str(cfg.get("Stain", "")).strip()
        if not stain:
            raise KeyError("Key 'Stain' missing or empty in YAML.")
        standard_genome_size_2c = float(cfg["Standard_genome_size_2C"])
        standard_genome_size_1c = 978
    except Exception as e:
        print(f"ERROR reading YAML: {e}", file=sys.stderr)
        sys.exit(2)

    # Locate data CSV inside the folder
    try:
        data_csv = find_data_csv(folder, cluster_csv_path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    # Read inputs
    try:
        clusters = read_cluster_csv(cluster_csv_path)
        data = read_data_csv(data_csv)
    except Exception as e:
        print(f"ERROR reading inputs: {e}", file=sys.stderr)
        sys.exit(2)

    # Detect region indices from columns present in the data CSV
    region_indices = detect_region_indices(data.columns)
    if not region_indices:
        print("ERROR: No region columns detected (expected headers like 'Region1_Number', 'Region2_NumberOfParticles', etc.).", file=sys.stderr)
        sys.exit(2)

    # Merge on SampleName to get species per row in data CSV
    merged = pd.merge(data, clusters, on="SampleName", how="inner")
    if merged.empty:
        print("ERROR: No overlapping SampleName values between data CSV and cluster CSV.", file=sys.stderr)
        sys.exit(2)

    # Build expected filenames & absolute paths
    def build_row(row):
        ts_fmt = parse_and_format_timestamp(str(row["ExportingUserTimestamp"]))
        fname = build_png_filename(ts_fmt, row["SessionID"], row["SampleID"])
        return fname

    merged["PNGFileName"] = merged.apply(build_row, axis=1)
    merged["PNGPath"] = merged["PNGFileName"].apply(lambda n: (folder / n).resolve())

    # Validate presence of PNG, JSON, XML for each base filename
    try:
        ensure_artifacts_exist(merged)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(3)

    # Create output directory
    out_dir = folder / "species_html"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Generate one HTML per species using Jinja2
    outputs = []
    for species, subdf in merged.groupby("Species"):
        html_path = make_species_html(
            species, subdf, out_dir, region_indices,
            standard_taxon=standard_taxon,
            standard_genome_size_2c=standard_genome_size_2c,
            standard_genome_size_1c=standard_genome_size_1c,
            stain=stain
        )
        outputs.append(html_path)

    # Print summary
    print("Generated HTML files:")
    for p in outputs:
        print(f"  - {p}")

if __name__ == "__main__":
    main()
