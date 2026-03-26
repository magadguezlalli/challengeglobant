"""
IABSA Bra Review - Data Preparation Script
Prepares scraped vendor CSVs for Tableau dashboard
"""

import pandas as pd
import numpy as np
import os
import re
import glob
import json

# load cvs files

def load_all_vendors(folder_path: str) -> pd.DataFrame:
    """Load all CSV files from folder and concatenate."""
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8", on_bad_lines="skip")
        # Infer vendor from filename if 'retailer' column is missing
        if "retailer" not in df.columns:
            df["retailer"] = os.path.basename(f).replace(".csv", "")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined):,} rows from {len(files)} files.")
    return combined


# bra sizes

def classify_single_size(size: str) -> str:
    """
    Returns:
      'bra'     → classic bra size like 34A, 36B, 40D, 36A/B, 36G(4D)
      'bra_xl'  → XL bra sizes like 1X, 2X, 3X
      'not_bra' → S/M/L/XL, numerics like 6/7/8, or single letters like A/B/C
    """
    size = str(size).strip()

    # [1X, 2X, 3X, ...] → bra (XL category)
    if re.fullmatch(r'\d+X', size, re.IGNORECASE):
        return 'bra_xl'

    # [34A, 36B, 40D, 36A/B, 36G(4D), ...] → classic bra
    if re.match(r'^\d{2}[A-Z]', size, re.IGNORECASE):
        return 'bra'

    # [S, M, L, XL, XXL] → NOT bra
    if re.fullmatch(r'(X{0,3}S|X{0,3}M|X{0,3}L|XXL|XXXL)', size, re.IGNORECASE):
        return 'not_bra'

    # [6, 7, 8, 9, ...] → NOT bra
    if re.fullmatch(r'\d+', size):
        return 'not_bra'

    # [A, B, C, D] alone → NOT bra (silicone adhesive)
    if re.fullmatch(r'[A-Z]{1,3}', size, re.IGNORECASE):
        return 'not_bra'

    return 'not_bra'


def extract_underbust(size: str) -> int | None:
    """Extract underbust number from a classic bra size like 34A → 34."""
    m = re.match(r'^(\d{2})', str(size).strip())
    return int(m.group(1)) if m else None


# assing size group

SIZE_GROUP_MAP = {
    30: 'Small', 32: 'Small',
    34: 'Medium', 36: 'Medium',
    38: 'Large', 40: 'Large',
    42: 'Extra Large', 44: 'Extra Large', 46: 'Extra Large',
}

def get_size_group(size_type: str, underbust: int | None) -> str | None:
    if size_type == 'bra_xl':
        return 'Extra Large'
    if size_type == 'bra' and underbust is not None:
        return SIZE_GROUP_MAP.get(underbust, None)
    return None


# parse sizes and explode rows

def parse_size_list(raw: str) -> list:
    """Parse total_sizes or available_size column (handles JSON arrays and comma-separated strings)."""
    if pd.isna(raw) or str(raw).strip() == '':
        return []
    raw = str(raw).strip()
    # Try JSON array first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(s).strip() for s in parsed]
    except Exception:
        pass
    # Fallback: comma-separated
    return [s.strip().strip('"').strip("'") for s in raw.split(',') if s.strip()]


def explode_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode total_sizes so each row = one size offering.
    Adds: size, is_bra, size_group, is_available
    """
    df = df.copy()

    # Parse size lists
    df['total_sizes_list'] = df['total_sizes'].apply(parse_size_list)
    df['available_size_list'] = df['available_size'].apply(parse_size_list)

    # Explode on offered sizes
    df = df.explode('total_sizes_list').rename(columns={'total_sizes_list': 'size'})
    df = df[df['size'].notna() & (df['size'] != '')]

    # Classify
    df['size_type'] = df['size'].apply(classify_single_size)
    df['is_bra'] = df['size_type'].isin(['bra', 'bra_xl'])

    df['underbust'] = df['size'].apply(lambda s: extract_underbust(s) if classify_single_size(s) == 'bra' else None)
    df['size_group'] = df.apply(lambda r: get_size_group(r['size_type'], r['underbust']), axis=1)

    # Availability: is this size in the available_size list?
    df['is_available'] = df.apply(
        lambda r: r['size'] in r['available_size_list'], axis=1
    )

    return df


# general clean and data quality

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Standarize column names
    df.columns = df.columns.str.strip().str.lower()

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Standardize retailer names
    if 'retailer' in df.columns:
        df['retailer'] = df['retailer'].str.strip().str.title()

    # Parse scrapping_datetime
    if 'scrapping_datetime' in df.columns:
        df['scrapping_datetime'] = pd.to_datetime(
            df['scrapping_datetime'], dayfirst=True, errors='coerce'
        )
        df['report_month'] = df['scrapping_datetime'].dt.to_period('M').astype(str)
    else:
        df['report_month'] = 'Unknown'

    # MRP: coerce to numeric
    if 'mrp' in df.columns:
        df['mrp'] = pd.to_numeric(df['mrp'], errors='coerce')

    # Rating
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    return df


# build fact table

def build_fact_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grain: one row per product × color × size × vendor × month
    Includes: is_bra, size_group, is_available
    """
    df = clean_data(df)
    df = explode_sizes(df)

    # Keep only bras
    bras = df[df['is_bra']].copy()

    # Select output columns
    cols = [
        'retailer', 'brand_name', 'product_name', 'color',
        'product_category', 'size', 'size_group',
        'is_available', 'mrp', 'rating', 'review_count',
        'report_month', 'pdp_url'
    ]
    available_cols = [c for c in cols if c in bras.columns]
    bras = bras[available_cols]

    print(f"Fact table: {len(bras):,} rows (bra size offerings)")
    return bras


# 7. BUILD VENDOR × SIZE GROUP SUMMARY
# (for warnings/sanctions visualization)


def build_vendor_summary(fact: pd.DataFrame) -> pd.DataFrame:
    """
    Per vendor per size_group:
      - offered_count: total size offerings
      - available_count: available size offerings
      - availability_pct: ratio
      - status: OK / Warning / Sanction (only for Extra Large)
    """
    summary = fact.groupby(['retailer', 'size_group']).agg(
        offered_count=('is_available', 'count'),
        available_count=('is_available', 'sum')
    ).reset_index()

    summary['availability_pct'] = (
        summary['available_count'] / summary['offered_count'] * 100
    ).round(2)

    # Apply warning/sanction rules only for Extra Large
    def get_status(row):
        if row['size_group'] != 'Extra Large':
            return 'N/A'
        pct = row['availability_pct']
        if pct < 30:
            return 'Sanction'
        elif pct < 50:
            return 'Warning'
        else:
            return 'OK'

    summary['status'] = summary.apply(get_status, axis=1)

    print("\nVendor × Size Group Summary:")
    print(summary[summary['size_group'] == 'Extra Large'][
        ['retailer', 'availability_pct', 'status']
    ].sort_values('availability_pct').to_string(index=False))

    return summary


# main

if __name__ == "__main__":
    # ── Change this path to your dataset folder ──
    DATA_FOLDER = "./dataset"
    OUTPUT_FOLDER = "./tableau_data"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load
    raw = load_all_vendors(DATA_FOLDER)

    # Build fact table
    fact = build_fact_table(raw)
    fact_path = os.path.join(OUTPUT_FOLDER, "fact_bra_items.csv")
    fact.to_csv(fact_path, index=False)
    print(f"\n✅ Fact table saved → {fact_path}")

    # Build summary for warnings/sanctions
    summary = build_vendor_summary(fact)
    summary_path = os.path.join(OUTPUT_FOLDER, "vendor_xl_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"✅ Vendor summary saved → {summary_path}")

    print("\nDone! Import both CSVs into Tableau.")
