#!/usr/bin/env python3
# =============================================================
# utils/postprocess.py  —  Cross-Month Fire Merge  (v5)
#
# CHANGES FROM v4
# ──────────────────────────────────────────────────────────────
# NEW — Stage 1: Within-month spatial merge  (unchanged from v5 spec)
#
# NEW (minor) — igbp_class_name column added to merged_inventory.csv
#   After all merges, a human-readable class name is derived from
#   igbp_lc_mode using the IGBP_CLASS_NAMES lookup table (same as
#   config.IGBP_CLASS_NAMES). This avoids the need to join back to
#   config in every downstream analysis script.
#
# NEW (minor) — IGBP distribution summary logged after merge
#   Logs the count and total burned area per IGBP class so you can
#   see at a glance which vegetation types dominate the inventory.
#
# All v4/v5 spatial merge logic preserved unchanged.
# =============================================================

import os
import sys
import math
import logging
import pandas as pd
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S'
)

# ─────────────────────────────────────────────────────────────
# IGBP CLASS NAMES  (mirrors config.IGBP_CLASS_NAMES)
# ─────────────────────────────────────────────────────────────
# Embedded here so postprocess.py works standalone without importing
# config (useful when running on a machine without GEE installed).

IGBP_CLASS_NAMES = {
    0:  'Unknown',
    1:  'Evergreen Needleleaf Forest',
    2:  'Evergreen Broadleaf Forest',
    3:  'Deciduous Needleleaf Forest',
    4:  'Deciduous Broadleaf Forest',
    5:  'Mixed Forest',
    6:  'Closed Shrubland',
    7:  'Open Shrubland',
    8:  'Woody Savanna',
    9:  'Savanna',
    10: 'Grassland',
    11: 'Permanent Wetland',
    12: 'Cropland',
    13: 'Urban',
    14: 'Cropland-Natural Mosaic',
    15: 'Snow / Ice',
    16: 'Barren',
    17: 'Water',
}

# ─────────────────────────────────────────────────────────────
# CROSS-MONTH MERGE PARAMETERS  (unchanged)
# ─────────────────────────────────────────────────────────────

MERGE_DISTANCE_KM        = 50.0
DOY_CONTINUITY_TOLERANCE = 5

# ─────────────────────────────────────────────────────────────
# WITHIN-MONTH SPATIAL MERGE PARAMETERS  (unchanged)
# ─────────────────────────────────────────────────────────────

SPATIAL_MERGE_BOUNDARY_KM    = 2.0
SPATIAL_MERGE_DOY_TOLERANCE  = 1


# ─────────────────────────────────────────────────────────────
# SHARED HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = (math.sin(dphi/2)**2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2)
    return 2 * R * math.asin(math.sqrt(a))


def circular_radius_km(area_km2):
    return math.sqrt(max(area_km2, 0.0) / math.pi)


def boundary_distance_km(lon1, lat1, area1, lon2, lat2, area2):
    d  = haversine_km(lon1, lat1, lon2, lat2)
    r1 = circular_radius_km(area1)
    r2 = circular_radius_km(area2)
    return max(0.0, d - r1 - r2)


def doy_ranges_adjacent(doy_min_a, doy_max_a, doy_min_b, doy_max_b, tolerance):
    gap = max(doy_min_a, doy_min_b) - min(doy_max_a, doy_max_b)
    return gap <= tolerance


def weighted_mean(val_a, area_a, val_b, area_b):
    if val_a == '' or val_b == '':
        return val_a or val_b or ''
    try:
        total = area_a + area_b
        if total > 0:
            return (float(val_a) * area_a + float(val_b) * area_b) / total
        else:
            return (float(val_a) + float(val_b)) / 2
    except (ValueError, TypeError):
        return ''


def weighted_mean_field(field, a, area_a, b, area_b):
    return weighted_mean(a.get(field, ''), area_a, b.get(field, ''), area_b)


# ─────────────────────────────────────────────────────────────
# WITHIN-MONTH SPATIAL MERGE  (unchanged)
# ─────────────────────────────────────────────────────────────

def is_spatial_merge_candidate(a, b,
                                boundary_km=SPATIAL_MERGE_BOUNDARY_KM,
                                doy_tolerance=SPATIAL_MERGE_DOY_TOLERANCE):
    if a['biome'] != b['biome']:
        return False
    try:
        if (int(a['year'])  != int(b['year']) or
                int(a['month']) != int(b['month'])):
            return False
    except (ValueError, KeyError):
        return False
    try:
        area_a = float(a.get('burn_area_km2', 0) or 0)
        area_b = float(b.get('burn_area_km2', 0) or 0)
        d_bnd  = boundary_distance_km(
            float(a['centroid_lon']), float(a['centroid_lat']), area_a,
            float(b['centroid_lon']), float(b['centroid_lat']), area_b
        )
    except (ValueError, KeyError, TypeError):
        return False
    if d_bnd > boundary_km:
        return False
    try:
        doy_min_a = float(a['burn_doy_min'])
        doy_max_a = float(a['burn_doy_max'])
        doy_min_b = float(b['burn_doy_min'])
        doy_max_b = float(b['burn_doy_max'])
    except (ValueError, KeyError):
        return False
    return doy_ranges_adjacent(doy_min_a, doy_max_a, doy_min_b, doy_max_b, doy_tolerance)


def merge_spatial_group(records):
    records_sorted = sorted(
        records,
        key=lambda r: float(r.get('burn_area_km2', 0) or 0),
        reverse=True
    )
    total_area = sum(float(r.get('burn_area_km2', 0) or 0) for r in records_sorted)

    doy_min_merged = min(float(r['burn_doy_min']) for r in records_sorted
                         if r.get('burn_doy_min') not in ('', None))
    doy_max_merged = max(float(r['burn_doy_max']) for r in records_sorted
                         if r.get('burn_doy_max') not in ('', None))

    ref_year = int(records_sorted[0]['year'])
    try:
        ref_date = datetime(ref_year, 1, 1)
        merged_start = (ref_date.__class__.fromordinal(
            ref_date.toordinal() + int(doy_min_merged) - 1
        ).strftime('%Y-%m-%d'))
        merged_end = (ref_date.__class__.fromordinal(
            ref_date.toordinal() + int(doy_max_merged) - 1
        ).strftime('%Y-%m-%d'))
    except Exception:
        merged_start = records_sorted[0].get('burn_start_date', '')
        merged_end   = records_sorted[-1].get('burn_end_date', '')

    if total_area > 0:
        merged_lon = sum(
            float(r['centroid_lon']) * float(r.get('burn_area_km2', 0) or 0)
            for r in records_sorted
        ) / total_area
        merged_lat = sum(
            float(r['centroid_lat']) * float(r.get('burn_area_km2', 0) or 0)
            for r in records_sorted
        ) / total_area
    else:
        merged_lon = sum(float(r['centroid_lon']) for r in records_sorted) / len(records_sorted)
        merged_lat = sum(float(r['centroid_lat']) for r in records_sorted) / len(records_sorted)

    def aw_mean(field):
        total_w, total_v = 0.0, 0.0
        for r in records_sorted:
            val  = r.get(field, '')
            area = float(r.get('burn_area_km2', 0) or 0)
            if val not in ('', None):
                try:
                    total_v += float(val) * area
                    total_w += area
                except (ValueError, TypeError):
                    pass
        return total_v / total_w if total_w > 0 else ''

    total_t21       = sum(float(r.get('total_t21', 0) or 0) for r in records_sorted)
    merged_mean_t21 = aw_mean('mean_t21')
    igbp_mode       = records_sorted[0].get('igbp_lc_mode', 0)

    try:
        doy_span_flag = max(
            1, max(int(float(r.get('doy_span_flag', 0) or 0)) for r in records_sorted)
        )
    except (ValueError, TypeError):
        doy_span_flag = 1

    return {
        'biome':            records_sorted[0]['biome'],
        'year':             records_sorted[0]['year'],
        'month':            records_sorted[0]['month'],
        'burn_start_date':  merged_start,
        'burn_end_date':    merged_end,
        'burn_doy_min':     doy_min_merged,
        'burn_doy_max':     doy_max_merged,
        'burn_doy_range':   doy_max_merged - doy_min_merged,
        'burn_area_km2':    total_area,
        'total_t21':        total_t21,
        'mean_t21':         merged_mean_t21,
        'centroid_lon':     merged_lon,
        'centroid_lat':     merged_lat,
        'doy_span_flag':    doy_span_flag,
        'igbp_lc_mode':     igbp_mode,
        'non_forest_frac':  aw_mean('non_forest_frac'),
        'mean_slope_deg':   aw_mean('mean_slope_deg'),
        'mean_elev_m':      aw_mean('mean_elev_m'),
        'fire_id':          records_sorted[0].get('fire_id', ''),
        'merged_from':      '+'.join(r.get('fire_id', '') for r in records_sorted),
        'n_fragments':      len(records_sorted),
    }


def run_spatial_merge(df):
    records = df.to_dict('records')
    n = len(records)
    if n == 0:
        return df

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            if is_spatial_merge_candidate(records[i], records[j]):
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(records[i])

    result               = []
    n_merged_groups      = 0
    n_merged_fragments   = 0

    for group_records in groups.values():
        if len(group_records) == 1:
            result.append(group_records[0])
        else:
            result.append(merge_spatial_group(group_records))
            n_merged_groups    += 1
            n_merged_fragments += len(group_records)

    log.info(
        f"Spatial merge: {n_merged_groups} groups of {n_merged_fragments} "
        f"fragments → {n_merged_groups} merged events  "
        f"({n} rows → {len(result)} rows)"
    )
    return pd.DataFrame(result)


# ─────────────────────────────────────────────────────────────
# CROSS-MONTH MERGE  (unchanged)
# ─────────────────────────────────────────────────────────────

def months_apart(row_a, row_b):
    ya, ma = int(row_a['year']), int(row_a['month'])
    yb, mb = int(row_b['year']), int(row_b['month'])
    return abs((ya * 12 + ma) - (yb * 12 + mb))


def is_merge_candidate(a, b):
    if a['biome'] != b['biome']:
        return False
    if months_apart(a, b) != 1:
        return False
    try:
        dist = haversine_km(
            float(a['centroid_lon']), float(a['centroid_lat']),
            float(b['centroid_lon']), float(b['centroid_lat'])
        )
    except (ValueError, KeyError):
        return False
    if dist > MERGE_DISTANCE_KM:
        return False
    try:
        doy_max_a = float(a['burn_doy_max'])
        doy_min_b = float(b['burn_doy_min'])
        doy_max_b = float(b['burn_doy_max'])
        doy_min_a = float(a['burn_doy_min'])
    except (ValueError, KeyError):
        return False

    ya, ma = int(a['year']), int(a['month'])
    yb, mb = int(b['year']), int(b['month'])
    if (ya * 12 + ma) < (yb * 12 + mb):
        earlier_doy_max = doy_max_a
        later_doy_min   = doy_min_b
    else:
        earlier_doy_max = doy_max_b
        later_doy_min   = doy_min_a

    ma_int, mb_int = int(a['month']), int(b['month'])
    year_boundary = ((ma_int == 12 and mb_int == 1) or
                     (ma_int == 1  and mb_int == 12))
    if year_boundary:
        return True
    return (earlier_doy_max + DOY_CONTINUITY_TOLERANCE) >= later_doy_min


def merge_rows(a, b):
    area_a = float(a.get('burn_area_km2', 0) or 0)
    area_b = float(b.get('burn_area_km2', 0) or 0)
    total_area = area_a + area_b

    t21_a = float(a.get('total_t21', 0) or 0)
    t21_b = float(b.get('total_t21', 0) or 0)

    if total_area > 0:
        merged_lon = (float(a['centroid_lon']) * area_a +
                      float(b['centroid_lon']) * area_b) / total_area
        merged_lat = (float(a['centroid_lat']) * area_a +
                      float(b['centroid_lat']) * area_b) / total_area
    else:
        merged_lon = (float(a['centroid_lon']) + float(b['centroid_lon'])) / 2
        merged_lat = (float(a['centroid_lat']) + float(b['centroid_lat'])) / 2

    try:
        start_a = datetime.strptime(str(a['burn_start_date']), '%Y-%m-%d')
        start_b = datetime.strptime(str(b['burn_start_date']), '%Y-%m-%d')
        end_a   = datetime.strptime(str(a['burn_end_date']),   '%Y-%m-%d')
        end_b   = datetime.strptime(str(b['burn_end_date']),   '%Y-%m-%d')
        merged_start = min(start_a, start_b).strftime('%Y-%m-%d')
        merged_end   = max(end_a,   end_b  ).strftime('%Y-%m-%d')
    except (ValueError, KeyError):
        merged_start = a['burn_start_date']
        merged_end   = b['burn_end_date']

    try:
        merged_doy_min = min(float(a['burn_doy_min']), float(b['burn_doy_min']))
        merged_doy_max = max(float(a['burn_doy_max']), float(b['burn_doy_max']))
    except (ValueError, KeyError):
        merged_doy_min = a.get('burn_doy_min', '')
        merged_doy_max = b.get('burn_doy_max', '')

    if total_area > 0:
        mean_t21_a = float(a.get('mean_t21', 0) or 0)
        mean_t21_b = float(b.get('mean_t21', 0) or 0)
        merged_mean_t21 = (mean_t21_a * area_a + mean_t21_b * area_b) / total_area
    else:
        merged_mean_t21 = (float(a.get('mean_t21', 0) or 0) +
                           float(b.get('mean_t21', 0) or 0)) / 2

    igbp_mode = a.get('igbp_lc_mode', 0) if area_a >= area_b else b.get('igbp_lc_mode', 0)

    return {
        'biome':            a['biome'],
        'year':             min(int(a['year']), int(b['year'])),
        'month':            min(int(a['month']), int(b['month'])),
        'burn_start_date':  merged_start,
        'burn_end_date':    merged_end,
        'burn_doy_min':     merged_doy_min,
        'burn_doy_max':     merged_doy_max,
        'burn_doy_range':   (merged_doy_max - merged_doy_min
                             if isinstance(merged_doy_max, float) else ''),
        'burn_area_km2':    total_area,
        'total_t21':        t21_a + t21_b,
        'mean_t21':         merged_mean_t21,
        'centroid_lon':     merged_lon,
        'centroid_lat':     merged_lat,
        'doy_span_flag':    2,
        'igbp_lc_mode':     igbp_mode,
        'non_forest_frac':  weighted_mean_field('non_forest_frac', a, area_a, b, area_b),
        'mean_slope_deg':   weighted_mean_field('mean_slope_deg',  a, area_a, b, area_b),
        'mean_elev_m':      weighted_mean_field('mean_elev_m',     a, area_a, b, area_b),
        'fire_id':          a.get('fire_id', ''),
        'merged_from':      f"{a.get('fire_id','')}+{b.get('fire_id','')}",
    }


def run_merge(df):
    records      = df.to_dict('records')
    merged_flags = [False] * len(records)
    result       = []
    merge_count  = 0

    for i in range(len(records)):
        if merged_flags[i]:
            continue
        row_a       = records[i]
        found_merge = False

        for j in range(i + 1, len(records)):
            if merged_flags[j]:
                continue
            if is_merge_candidate(row_a, records[j]):
                merged = merge_rows(row_a, records[j])
                result.append(merged)
                merged_flags[i] = True
                merged_flags[j] = True
                merge_count += 1
                found_merge = True
                break

        if not found_merge:
            result.append(row_a)

    log.info(f"Cross-month merge: {merge_count} pairs merged. {len(result)} events remain.")
    return pd.DataFrame(result)


# ─────────────────────────────────────────────────────────────
# IGBP CLASS NAME ANNOTATION  (NEW v5)
# ─────────────────────────────────────────────────────────────

def add_igbp_class_name(df):
    """
    Add igbp_class_name column derived from igbp_lc_mode.

    Avoids the need to import config in every downstream analysis script.
    The column is human-readable (e.g. 'Evergreen Broadleaf Forest')
    and safe to use directly as a plot label.
    """
    if 'igbp_lc_mode' not in df.columns:
        log.warning("igbp_lc_mode column not found — igbp_class_name not added.")
        log.warning("  This column is produced by pipeline_1_inventory.py v5.")
        log.warning("  If running with an older inventory, re-run pipeline_1.")
        return df

    def lookup(val):
        try:
            return IGBP_CLASS_NAMES.get(int(float(val)), 'Unknown')
        except (ValueError, TypeError):
            return 'Unknown'

    df['igbp_class_name'] = df['igbp_lc_mode'].apply(lookup)
    return df


def log_igbp_distribution(df):
    """
    Log a summary table of fire count and total burned area per IGBP class.
    Helps identify which vegetation types dominate the global inventory.
    """
    if 'igbp_lc_mode' not in df.columns:
        return

    log.info("\n  IGBP class distribution in merged inventory:")
    log.info(f"  {'Class':>4}  {'Name':<32}  {'Fires':>6}  {'Area (km²)':>12}")
    log.info(f"  {'-'*4}  {'-'*32}  {'-'*6}  {'-'*12}")

    try:
        area_col = df['burn_area_km2'].apply(
            lambda x: float(x) if x not in ('', None) else 0.0
        )
        df2 = df.copy()
        df2['_area'] = area_col
        df2['_igbp'] = df2['igbp_lc_mode'].apply(
            lambda x: int(float(x)) if x not in ('', None) else 0
        )

        summary = (df2.groupby('_igbp')
                      .agg(fires=('fire_id', 'count'),
                           total_area=('_area', 'sum'))
                      .sort_values('total_area', ascending=False))

        for igbp_cls, row in summary.iterrows():
            name = IGBP_CLASS_NAMES.get(igbp_cls, 'Unknown')
            log.info(f"  {igbp_cls:>4}  {name:<32}  {int(row['fires']):>6}  "
                     f"{row['total_area']:>12,.1f}")
    except Exception as e:
        log.warning(f"  Could not compute IGBP distribution: {e}")


# ─────────────────────────────────────────────────────────────
# CSV I/O  (unchanged)
# ─────────────────────────────────────────────────────────────

def load_all_csvs(inventory_dir):
    csv_files = [f for f in os.listdir(inventory_dir) if f.endswith('.csv')]
    if not csv_files:
        log.error(f"No CSVs found in {inventory_dir}")
        return pd.DataFrame()

    frames = []
    for fname in sorted(csv_files):
        fpath = os.path.join(inventory_dir, fname)
        try:
            df = pd.read_csv(fpath, dtype=str)
            if 'biome' not in df.columns:
                parts = fname.replace('.csv', '').split('_')
                df['biome'] = parts[1] if len(parts) > 1 else 'unknown'
            frames.append(df)
        except Exception as e:
            log.warning(f"  Could not read {fname}: {e}")

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded {len(all_df)} rows from {len(csv_files)} CSV files.")
    return all_df


def assign_fire_ids(df):
    if 'fire_id' not in df.columns or df['fire_id'].isna().all():
        df['fire_id'] = [
            f"{row.get('biome','unk')}_{row.get('year','0')}_{i:05d}"
            for i, row in df.iterrows()
        ]
    return df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    inventory_dir = config.INVENTORY_DIR
    output_path   = os.path.join(inventory_dir, 'merged_inventory.csv')

    log.info(f"Loading CSVs from {inventory_dir} ...")
    df = load_all_csvs(inventory_dir)
    if df.empty:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    df = assign_fire_ids(df)
    log.info(f"Starting with {len(df)} raw fire polygons.")

    # ── Stage 1: Within-month spatial merge ────────────────────
    log.info(
        f"\nStage 1 — Spatial merge  "
        f"(boundary ≤ {SPATIAL_MERGE_BOUNDARY_KM} km, "
        f"DOY tolerance = {SPATIAL_MERGE_DOY_TOLERANCE} day)"
    )
    df = run_spatial_merge(df)
    log.info(f"After spatial merge : {len(df)} events.")

    # ── Stage 2: Cross-month merge ──────────────────────────────
    log.info(
        f"\nStage 2 — Cross-month merge  "
        f"(centroid ≤ {MERGE_DISTANCE_KM} km, "
        f"DOY gap ≤ {DOY_CONTINUITY_TOLERANCE} days)"
    )
    merged_df = run_merge(df)
    log.info(f"After cross-month merge : {len(merged_df)} events.")

    # ── Stage 3: Add igbp_class_name  (NEW v5) ─────────────────
    merged_df = add_igbp_class_name(merged_df)

    # ── Summary ─────────────────────────────────────────────────
    flag_col = merged_df.get('doy_span_flag',
                              pd.Series(dtype=str)).astype(str)
    log.info(f"\n  doy_span_flag=0 (clean)          : {(flag_col == '0').sum()}")
    log.info(f"  doy_span_flag=1 (spatial merge)  : {(flag_col == '1').sum()}")
    log.info(f"  doy_span_flag=2 (cross-month)    : {(flag_col == '2').sum()}")

    # ── IGBP distribution summary  (NEW v5) ────────────────────
    log_igbp_distribution(merged_df)

    merged_df.to_csv(output_path, index=False)
    log.info(f"\nMerged inventory written to: {output_path}")
    log.info("Columns added in v5: igbp_class_name")
    log.info("Next step: pipeline_2_analysis.py")


if __name__ == '__main__':
    main()