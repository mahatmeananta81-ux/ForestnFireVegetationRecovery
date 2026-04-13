#!/usr/bin/env python3
# =============================================================
# screen_cloudy_fires.py  (v2 — scene-count approach)
# Run AFTER postprocess.py, BEFORE pipeline_2_analysis.py.
#
# WHY v1 WAS WRONG
# ────────────────
# v1 used col.count().reduceRegion(sum) to count valid pixels.
# This sums the number of unmasked observations per pixel across
# the AOI. After aggressive SCL cloud masking over the Amazon wet
# season, a collection with 8 scenes may have only 8 pixels that
# survived masking — far below MIN_VALID_PIXELS=50.
# Result: 35788/35788 fires rejected (100%) even though GEE and
# MODIS are working perfectly.
#
# WHY build_composite() ALMOST NEVER FAILS NOW
# ─────────────────────────────────────────────
# With the three-tier fallback (S2 → L8 → MODIS), the only way
# build_composite() produces a 0-band image is:
#   1. Invalid / ocean centroid (AOI has no land pixels at all)
#   2. MODIS has zero scenes for the date window (practically
#      impossible for 2019-2023 over land — MOD09A1 is global
#      and continuous since 2000)
#   3. Date window is malformed (start > end, or year < 2000)
#
# WHAT v2 CHECKS
# ───────────────
# 1. Coordinate validity — not null, not 0/0, in WGS84 range
# 2. AOI is on land — MODIS only covers land surfaces
# 3. MODIS has >= 1 scene for the PRE window
# 4. MODIS has >= 1 scene for the POST window
# 5. Date windows are internally consistent
#
# Optical coverage (S2/L8) is NOT checked for pass/fail.
# MODIS handles optical gaps. S2/L8 scene counts are logged
# as diagnostics so you know which fires use MODIS fallback.
#
# OUTPUT FILES
# ─────────────
# merged_inventory_screened.csv  — feed into pipeline_2
# merged_inventory_rejected.csv  — inspect rejection reasons
# screen_cloudy_fires.log        — full per-fire log
#
# USAGE
# ─────
# python screen_cloudy_fires.py
# =============================================================

import ee
import os
import sys
import csv
import time
import logging
import concurrent.futures
from datetime import datetime, timedelta

# ── CSV field size limit (handles long merged_from strings) ──
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ── Logging ──────────────────────────────────────────────────
os.makedirs(os.path.join(config.BASE_DIR, 'outputs'), exist_ok=True)
LOG_PATH = os.path.join(config.BASE_DIR, 'outputs', 'screen_cloudy_fires.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── Parameters ───────────────────────────────────────────────
# Minimum MOD09A1 8-day composites needed in each date window.
# A 90-day window contains ~11 composites. Set to 1 — we only
# need MODIS to have any data at all.
MIN_MODIS_SCENES = 1

# Concurrent GEE threads. Scene-count calls are metadata-only
# (no pixel computation), so 8 workers is safe.
N_WORKERS = 8


# ─────────────────────────────────────────────────────────────
# DATE WINDOW BUILDER
# ─────────────────────────────────────────────────────────────

def make_pre_window(fire_start_str):
    """±45 days around same calendar period one year before fire."""
    fs    = datetime.strptime(fire_start_str, '%Y-%m-%d')
    pivot = fs.replace(year=fs.year - 1)
    start = (pivot - timedelta(days=45)).strftime('%Y-%m-%d')
    end   = (pivot + timedelta(days=45)).strftime('%Y-%m-%d')
    return start, end


def make_post_window(fire_end_str):
    """POST_FIRE_DAYS_START .. POST_FIRE_DAYS_END after fire end."""
    fe    = datetime.strptime(fire_end_str, '%Y-%m-%d')
    start = (fe + timedelta(days=config.POST_FIRE_DAYS_START)).strftime('%Y-%m-%d')
    end   = (fe + timedelta(days=config.POST_FIRE_DAYS_END)).strftime('%Y-%m-%d')
    return start, end


# ─────────────────────────────────────────────────────────────
# SCENE COUNT HELPERS
# ─────────────────────────────────────────────────────────────

def count_modis_scenes(aoi, start, end):
    """
    Count MOD09A1 8-day composites covering this AOI/window.
    Metadata-only call — no pixel computation whatsoever.
    """
    return (ee.ImageCollection('MODIS/061/MOD09A1')
              .filterBounds(aoi)
              .filterDate(start, end)
              .size()
              .getInfo())


def count_s2_scenes(aoi, start, end):
    """Count S2 scenes below cloud threshold. Diagnostic only."""
    try:
        return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',
                                       config.MAX_CLOUD_PCT))
                  .size()
                  .getInfo())
    except Exception:
        return -1


def count_l8_scenes(aoi, start, end):
    """Count L8 scenes below cloud threshold. Diagnostic only."""
    try:
        return (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                  .filterBounds(aoi)
                  .filterDate(start, end)
                  .filter(ee.Filter.lt('CLOUD_COVER', config.MAX_CLOUD_PCT))
                  .size()
                  .getInfo())
    except Exception:
        return -1


def check_aoi_on_land(aoi):
    """
    Confirm AOI contains at least one MODIS land pixel.
    Uses MOD44W water mask (250 m). Returns True if land found.
    Fails open (returns True) so a failing check never wrongly
    rejects a real fire.
    """
    try:
        land_water = (ee.ImageCollection('MODIS/006/MOD44W')
                        .filterDate('2015-01-01', '2016-01-01')
                        .first()
                        .select('water_mask'))
        # water_mask: 0=land, 1=water. Count land (0) pixels.
        result = (land_water.eq(0)
                    .reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=aoi,
                        scale=250,
                        bestEffort=True,
                        maxPixels=int(1e6)
                    ).getInfo())
        land_pixels = int(result.get('water_mask', 0) or 0)
        return land_pixels >= 1
    except Exception:
        return True  # fail open


# ─────────────────────────────────────────────────────────────
# SINGLE-FIRE SCREENER
# ─────────────────────────────────────────────────────────────

def screen_fire(row):
    """
    Returns (fire_id, passes: bool, reason: str, detail: dict).
    Thread-safe. Re-initializes GEE per thread (Windows requirement).
    """
    # Re-initialize per thread — required on Windows
    try:
        ee.Initialize(project=config.GEE_PROJECT)
    except Exception:
        pass

    fire_id = row.get('fire_id', 'unknown')

    # ── 1. Validate coordinates ────────────────────────────────
    try:
        lon_raw = row.get('centroid_lon', '')
        lat_raw = row.get('centroid_lat', '')
        if lon_raw in ('', None) or lat_raw in ('', None):
            return fire_id, False, 'missing_centroid', {}
        lon = float(lon_raw)
        lat = float(lat_raw)
    except (ValueError, TypeError):
        return fire_id, False, 'invalid_centroid_format', {}

    if lon == 0.0 and lat == 0.0:
        return fire_id, False, 'null_centroid_(0,0)', {}

    if not (-180.0 <= lon <= 180.0) or not (-90.0 <= lat <= 90.0):
        return fire_id, False, f'coords_out_of_range', {}

    # ── 2. Validate dates ──────────────────────────────────────
    fire_start = (row.get('burn_start_date', '') or '').strip()
    fire_end   = (row.get('burn_end_date',   '') or '').strip() or fire_start

    if not fire_start:
        return fire_id, False, 'missing_burn_start_date', {}

    try:
        datetime.strptime(fire_start, '%Y-%m-%d')
        datetime.strptime(fire_end,   '%Y-%m-%d')
    except ValueError:
        return fire_id, False, f'bad_date_format', {}

    try:
        pre_start,  pre_end  = make_pre_window(fire_start)
        post_start, post_end = make_post_window(fire_end)
    except Exception as e:
        return fire_id, False, f'date_window_error', {}

    # ── 3. Build AOI ───────────────────────────────────────────
    try:
        aoi = (ee.Geometry.Point([lon, lat])
                 .buffer(distance=config.AOI_BUFFER_M, maxError=1))
    except Exception as e:
        return fire_id, False, f'buffer_error', {}

    # ── 4. Land check ──────────────────────────────────────────
    if not check_aoi_on_land(aoi):
        return fire_id, False, 'ocean_centroid_no_land_pixels', {}

    # ── 5. MODIS scene count — pre window (pass/fail) ─────────
    try:
        pre_modis = count_modis_scenes(aoi, pre_start, pre_end)
    except Exception as e:
        return fire_id, False, f'modis_pre_error', {}

    if pre_modis < MIN_MODIS_SCENES:
        return fire_id, False, (
            f'pre_modis_no_scenes_[{pre_start}..{pre_end}]'
        ), {'pre_modis_scenes': pre_modis}

    # ── 6. MODIS scene count — post window (pass/fail) ────────
    try:
        post_modis = count_modis_scenes(aoi, post_start, post_end)
    except Exception as e:
        return fire_id, False, f'modis_post_error', {}

    if post_modis < MIN_MODIS_SCENES:
        return fire_id, False, (
            f'post_modis_no_scenes_[{post_start}..{post_end}]'
        ), {'post_modis_scenes': post_modis}

    # ── 7. Optical counts — diagnostic only ───────────────────
    pre_s2  = count_s2_scenes(aoi, pre_start,  pre_end)
    pre_l8  = count_l8_scenes(aoi, pre_start,  pre_end)
    post_s2 = count_s2_scenes(aoi, post_start, post_end)
    post_l8 = count_l8_scenes(aoi, post_start, post_end)

    pre_tier  = 'S2' if pre_s2  > 0 else ('L8' if pre_l8  > 0 else 'MODIS')
    post_tier = 'S2' if post_s2 > 0 else ('L8' if post_l8 > 0 else 'MODIS')

    return fire_id, True, 'ok', {
        'pre_modis_scenes':  pre_modis,
        'post_modis_scenes': post_modis,
        'pre_s2_scenes':     pre_s2,
        'pre_l8_scenes':     pre_l8,
        'post_s2_scenes':    post_s2,
        'post_l8_scenes':    post_l8,
        'pre_tier':          pre_tier,
        'post_tier':         post_tier,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    try:
        ee.Initialize(project=config.GEE_PROJECT)
        log.info("GEE initialised.")
    except Exception as e:
        log.error(f"GEE init failed: {e}")
        sys.exit(1)

    merged_path = os.path.join(config.INVENTORY_DIR, 'merged_inventory.csv')
    if not os.path.exists(merged_path):
        log.error(f"merged_inventory.csv not found: {merged_path}")
        log.error("Run postprocess.py first.")
        sys.exit(1)

    with open(merged_path, newline='', encoding='utf-8') as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames
        fires      = list(reader)

    log.info(f"Loaded {len(fires)} fires from merged_inventory.csv")
    log.info(f"MIN_MODIS_SCENES={MIN_MODIS_SCENES}  "
             f"AOI_BUFFER_M={config.AOI_BUFFER_M}  "
             f"N_WORKERS={N_WORKERS}")

    # ── Resume support ─────────────────────────────────────────
    screened_path = os.path.join(config.INVENTORY_DIR,
                                  'merged_inventory_screened.csv')
    rejected_path = os.path.join(config.INVENTORY_DIR,
                                  'merged_inventory_rejected.csv')
    already_done  = set()
    for path in [screened_path, rejected_path]:
        if os.path.exists(path):
            with open(path, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    already_done.add(row.get('fire_id', ''))

    pending = [r for r in fires if r.get('fire_id') not in already_done]
    log.info(f"Already screened: {len(already_done)}  Pending: {len(pending)}")

    if not pending:
        log.info("All fires already screened.")
        return

    # ── Output setup ───────────────────────────────────────────
    extra = ['screen_reason', 'pre_tier', 'post_tier',
             'pre_s2_scenes', 'pre_l8_scenes', 'pre_modis_scenes',
             'post_s2_scenes', 'post_l8_scenes', 'post_modis_scenes']
    out_fields = list(fieldnames or []) + [
        f for f in extra if f not in (fieldnames or [])]

    mode = 'a' if already_done else 'w'
    f_pass = open(screened_path, mode, newline='', encoding='utf-8')
    f_fail = open(rejected_path, mode, newline='', encoding='utf-8')
    pw = csv.DictWriter(f_pass, fieldnames=out_fields, extrasaction='ignore')
    fw = csv.DictWriter(f_fail, fieldnames=out_fields, extrasaction='ignore')
    if mode == 'w':
        pw.writeheader()
        fw.writeheader()

    # ── Screen ─────────────────────────────────────────────────
    n_pass = n_fail = 0
    tier_counts = {'S2': 0, 'L8': 0, 'MODIS': 0}
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        future_map = {ex.submit(screen_fire, row): row for row in pending}

        for idx, future in enumerate(
                concurrent.futures.as_completed(future_map)):
            row = future_map[future]
            try:
                fire_id, passes, reason, detail = future.result(timeout=180)
            except concurrent.futures.TimeoutError:
                fire_id, passes, reason, detail = (
                    row.get('fire_id', '?'), False, 'timeout_180s', {})
            except Exception as e:
                fire_id, passes, reason, detail = (
                    row.get('fire_id', '?'), False, f'exception_{e}', {})

            row['screen_reason']     = reason
            row['pre_tier']          = detail.get('pre_tier', '')
            row['post_tier']         = detail.get('post_tier', '')
            row['pre_s2_scenes']     = detail.get('pre_s2_scenes', '')
            row['pre_l8_scenes']     = detail.get('pre_l8_scenes', '')
            row['pre_modis_scenes']  = detail.get('pre_modis_scenes', '')
            row['post_s2_scenes']    = detail.get('post_s2_scenes', '')
            row['post_l8_scenes']    = detail.get('post_l8_scenes', '')
            row['post_modis_scenes'] = detail.get('post_modis_scenes', '')

            if passes:
                pw.writerow(row)
                f_pass.flush()
                n_pass += 1
                t = detail.get('pre_tier', '')
                if t in tier_counts:
                    tier_counts[t] += 1
                icon = '✓'
            else:
                fw.writerow(row)
                f_fail.flush()
                n_fail += 1
                icon = '✗'

            done    = idx + 1
            elapsed = time.time() - t0
            rate    = done / max(elapsed, 1)
            eta     = (len(pending) - done) / max(rate, 1e-6)

            log.info(
                f"[{done:>5}/{len(pending)}] {icon} {fire_id:<40} "
                f"reason={reason:<40} "
                f"pre={row['pre_tier']:<5} post={row['post_tier']:<5} "
                f"ETA {eta/60:.1f}m"
            )

    f_pass.close()
    f_fail.close()

    total = n_pass + n_fail
    log.info(f"\n{'='*60}")
    log.info(f"Screening complete.  {total} fires checked.")
    log.info(f"  Passed  : {n_pass} ({100*n_pass/max(total,1):.1f}%)")
    log.info(f"  Rejected: {n_fail} ({100*n_fail/max(total,1):.1f}%)")
    log.info(f"\n  Pre-fire tier breakdown (passed fires):")
    for t, c in tier_counts.items():
        log.info(f"    {t:<6}: {c}")

    # Rejection reason summary
    reasons = {}
    with open(rejected_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            key = r.get('screen_reason', 'unknown').split('[')[0].rstrip('_')
            reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        log.info(f"\n  Rejection breakdown:")
        for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
            log.info(f"    {c:>6}  {r}")

    log.info(f"\n  Screened CSV: {screened_path}")
    log.info(f"  Rejected CSV: {rejected_path}")


if __name__ == '__main__':
    main()