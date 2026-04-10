#!/usr/bin/env python3
# =============================================================
# pipeline_2_analysis.py  —  Per-Fire Deep Analysis  (v5)
#
# CHANGES FROM v4
# ──────────────────────────────────────────────────────────────
# NEW — igbp_lc_mode wired into all exported features
#   Pipeline 1 v5 produced igbp_lc_mode per fire polygon. This
#   field is now forwarded onto every timeseries and area-table
#   feature so downstream analysis can stratify by forest type
#   without re-joining the inventory.
#
# NEW — EVI added to annual time-series
#   calc_evi() was already in gee_functions. NDVI saturates above
#   LAI ≈ 3, making it nearly useless for distinguishing recovery
#   stages in dense tropical canopy (IGBP classes 1–2: Amazon,
#   Congo, SE Asia). EVI reduces canopy background and atmospheric
#   effects. Both are now exported so the divergence between them
#   is available as a canopy-density change signal.
#
# NEW — Pre/post NDVI and EVI baseline means stored in timeseries
#   pre_ndvi_mean, post_ndvi_mean, pre_evi_mean, post_evi_mean are
#   computed from the pre/post composites and stored as fixed
#   properties on every year's timeseries feature. This lets the
#   visualize_recovery.py script compute recovery completeness:
#     (yr_NDVI − post_NDVI) / (pre_NDVI − post_NDVI)
#   without needing a separate GEE call.
#
# NEW — fire_year stored in timeseries features
#   Allows downstream scripts to compute years_since_fire = year − fire_year
#   without parsing burn_start_date.
#
# NEW — build_severity_igbp_table()
#   Computes burned area (km²) for every combination of USGS burn
#   severity class (1–6) × IGBP land cover class (1–17) within
#   the burn scar. Exported as a CSV per fire. Answers: do tropical
#   evergreen forests burn at higher severity than savannas?
#
# NEW — NDVI and EVI pre/post GeoTIFF exports
#   ndvi_pre, ndvi_post, evi_pre, evi_post added to raster exports.
#   Together with the existing nbr_pre/post these give a full
#   spectral picture of vegetation state before and after the fire.
#
# All v4 changes preserved unchanged.
# =============================================================

import ee
import os
import sys
import csv
import time
import logging

# ── FIX FOR CSV "field larger than field limit" ERROR ──────────
max_int = sys.maxsize
while True:
    # Decrease the max_int value by a factor of 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)
# ─────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.gee_functions import (
    build_composite, calc_dnbr, calc_nbr, calc_ndvi, calc_evi,
    calc_burn_severity, build_burn_mask, build_feature_stack
)
from utils.export_helpers import (
    submit_fire_exports, wait_for_capacity, get_running_task_count
)

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(config.BASE_DIR, 'outputs',
                                          'pipeline2.log')),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────

def load_checkpoint():
    """Return set of fire_ids already processed."""
    path = config.CHECKPOINT_FILE
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def save_checkpoint(fire_id):
    """Append a fire_id to the checkpoint file."""
    with open(config.CHECKPOINT_FILE, 'a') as f:
        f.write(fire_id + '\n')


# ─────────────────────────────────────────────────────────────
# INVENTORY LOADER  (unchanged from v4)
# ─────────────────────────────────────────────────────────────

def load_all_inventories(inventory_dir):
    """
    Load fire inventory for Pipeline 2.

    Preferred source: merged_inventory.csv (postprocess.py output).
    Fallback: individual inv_*.csv files.

    v5 note: merged_inventory.csv now carries igbp_lc_mode and
    igbp_class_name added by postprocess.py v5.
    """
    if not os.path.exists(inventory_dir):
        log.error(f"Inventory directory not found: {inventory_dir}")
        log.error("Run pipeline_1, download CSVs, then run postprocess.py")
        return []

    merged_path = os.path.join(inventory_dir, 'merged_inventory.csv')

    if os.path.exists(merged_path):
        log.info(f"Using merged inventory: {merged_path}")
        with open(merged_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        for i, row in enumerate(rows):
            if not row.get('fire_id'):
                row['fire_id'] = (f"{row.get('biome','unk')}_"
                                  f"{row.get('year','0')}_{i:05d}")
        log.info(f"Loaded {len(rows)} fire events from merged inventory.")
        return rows

    log.warning("merged_inventory.csv not found. Reading raw per-tile CSVs.")
    csv_files = [f for f in os.listdir(inventory_dir)
                 if f.endswith('.csv') and not f.startswith('merged')]
    if not csv_files:
        log.error(f"No CSV files found in {inventory_dir}")
        return []

    all_fires = []
    for fname in sorted(csv_files):
        fpath = os.path.join(inventory_dir, fname)
        with open(fpath, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
        for i, row in enumerate(rows):
            biome = row.get('biome', fname.replace('.csv', ''))
            year  = row.get('year', 'unknown')
            row['fire_id'] = f"{biome}_{year}_{i:05d}"
            all_fires.append(row)

    log.info(f"Loaded {len(all_fires)} fires from {len(csv_files)} raw CSV files.")
    return all_fires


# ─────────────────────────────────────────────────────────────
# DATE WINDOW BUILDER  (unchanged)
# ─────────────────────────────────────────────────────────────

def build_date_windows(fire_start_str, fire_end_str):
    """
    Build temporal analysis windows from MCD64A1 DOY-derived dates.
    See v4 docstring for full documentation.
    """
    fire_start = ee.Date(fire_start_str)
    if not fire_end_str or fire_end_str == fire_start_str:
        fire_end_str = fire_start_str
    fire_end = ee.Date(fire_end_str)

    pre_start  = fire_start.advance(-1, 'year').advance(-45, 'day')
    pre_end    = fire_start.advance(-1, 'year').advance( 45, 'day')

    post_start = fire_end.advance(config.POST_FIRE_DAYS_START, 'day')
    post_end   = fire_end.advance(config.POST_FIRE_DAYS_END,   'day')

    rec_start  = fire_start.advance(config.RECOVERY_YEARS_AFTER, 'year').advance(-3, 'month')
    rec_end    = fire_start.advance(config.RECOVERY_YEARS_AFTER, 'year').advance( 3, 'month')

    fire_year  = int(fire_start_str[:4])
    ts_start   = fire_year - 1
    ts_end     = min(fire_year + config.RECOVERY_YEARS_AFTER + 1, 2024)

    return {
        'pre_start':     pre_start,
        'pre_end':       pre_end,
        'post_start':    post_start,
        'post_end':      post_end,
        'rec_start':     rec_start,
        'rec_end':       rec_end,
        'ts_years':      list(range(ts_start, ts_end + 1)),
        'fire_start_ee': fire_start,
        'fire_year':     fire_year,
    }


# ─────────────────────────────────────────────────────────────
# RANDOM FOREST LAND COVER CLASSIFICATION  (unchanged from v4)
# ─────────────────────────────────────────────────────────────

def run_rf_classification(rec_img, burn_mask, aoi):
    """
    Train + apply Random Forest on 13-band feature stack.
    Labels: ESA WorldCover v200. See v4 docstring for rationale.
    """
    world_cover = (ee.ImageCollection('ESA/WorldCover/v200')
                     .first().clip(aoi))
    land_cover_ref = (world_cover.select('Map')
                                 .remap(config.WORLDCOVER_FROM,
                                        config.WORLDCOVER_TO)
                                 .rename('landcover').toInt())

    feature_stack = build_feature_stack(rec_img)

    training_data = (feature_stack
                     .addBands(land_cover_ref)
                     .stratifiedSample(
                         numPoints=config.RF_NUM_POINTS,
                         classBand='landcover',
                         region=aoi, scale=20, seed=42,
                         tileScale=16, geometries=True
                     ))

    classifier = (ee.Classifier.smileRandomForest(
                      numberOfTrees=300, variablesPerSplit=4,
                      minLeafPopulation=2, seed=42
                  ).train(
                      features=training_data,
                      classProperty='landcover',
                      inputProperties=feature_stack.bandNames()
                  ))

    classified = (feature_stack.updateMask(burn_mask)
                               .classify(classifier)
                               .rename('LandCover').clip(aoi))
    return classified, training_data


# ─────────────────────────────────────────────────────────────
# RECOVERY AREA TABLE  (unchanged — igbp_lc_mode added to features)
# ─────────────────────────────────────────────────────────────

def build_area_table(classified, aoi, fire_row, windows):
    """
    Compute area (km²) per RF land-cover class within the burn scar.
    v5: igbp_lc_mode forwarded onto each feature for later stratification.
    """
    class_ids   = ee.List([1, 2, 3, 4, 5])
    class_names = ee.List(list(config.CLASS_NAMES.values()))

    igbp_mode = int(float(fire_row.get('igbp_lc_mode', 0) or 0))

    def area_for_class(class_val):
        class_val = ee.Number(class_val)
        sq_m = (classified.eq(class_val)
                          .multiply(ee.Image.pixelArea())
                          .reduceRegion(
                              reducer=ee.Reducer.sum(),
                              geometry=aoi, scale=20,
                              bestEffort=True, tileScale=16
                          ))
        idx = class_ids.indexOf(class_val)
        return ee.Feature(None, {
            'ClassID':        class_val,
            'ClassName':      class_names.get(idx),
            'Area_km2':       ee.Number(sq_m.get('LandCover')).divide(1e6),
            'fire_id':        fire_row['fire_id'],
            'biome':          fire_row.get('biome', ''),
            'igbp_lc_mode':   igbp_mode,
            'fire_start':     fire_row.get('burn_start_date', ''),
            'rec_year':       windows['rec_start'].format('YYYY'),
        })

    return ee.FeatureCollection(class_ids.map(area_for_class))


# ─────────────────────────────────────────────────────────────
# BURN SEVERITY × IGBP CROSS-TABLE  (NEW v5)
# ─────────────────────────────────────────────────────────────

def build_severity_igbp_table(burn_severity, aoi, fire_row):
    """
    Compute burned area (km²) per USGS severity class × IGBP land cover class.

    Why this cross-table?
    The central question in fire ecology is whether specific forest types
    are more vulnerable to high-severity fire. This table gives a direct
    answer: for each fire, how much of each vegetation type burned at
    each severity level?

    Severity classes (USGS, from calc_burn_severity):
      1=Enhanced Regrowth, 2=Unburned, 3=Low, 4=Moderate-Low,
      5=Moderate-High, 6=High

    IGBP classes: 1–17 (see pipeline_1 comments for full mapping).
    Only classes present in the AOI will appear in the output.

    Output features: one per (severity_class, igbp_class) pair with
    area_km2 > 0. Zero-area combinations are dropped to keep CSV lean.

    Design note — why not a grouped reducer?
    ee.Reducer.group() in GEE cannot group by two dimensions simultaneously.
    The nested ee.List.map() approach is equivalent and produces a flat
    FeatureCollection compatible with CSV export.
    """
    # Use (fire_year - 1) for pre-fire land cover — same logic as pipeline_1
    lc_year = int(float(fire_row.get('year', 2019) or 2019)) - 1
    igbp_img = (ee.ImageCollection('MODIS/061/MCD12Q1')
                  .filterDate(f'{lc_year}-01-01', f'{lc_year}-12-31')
                  .first()
                  .select('LC_Type1'))

    fire_id   = fire_row['fire_id']
    biome     = fire_row.get('biome', '')
    igbp_mode = int(float(fire_row.get('igbp_lc_mode', 0) or 0))

    severity_classes = ee.List([1, 2, 3, 4, 5, 6])
    igbp_classes     = ee.List(list(range(1, 18)))  # 1–17

    def per_severity(sev):
        sev = ee.Number(sev)

        def per_igbp(igbp):
            igbp = ee.Number(igbp)
            combined_mask = burn_severity.eq(sev).And(igbp_img.eq(igbp))
            area_m2 = (combined_mask
                       .multiply(ee.Image.pixelArea())
                       .reduceRegion(
                           reducer=ee.Reducer.sum(),
                           geometry=aoi, scale=20,
                           bestEffort=True, tileScale=16
                       ))
            area_km2 = ee.Number(area_m2.get('BurnSeverity')).divide(1e6)
            return ee.Feature(None, {
                'fire_id':            fire_id,
                'biome':              biome,
                'igbp_lc_mode':       igbp_mode,
                'severity_class':     sev,
                'igbp_class':         igbp,
                'area_km2':           area_km2,
            })

        return igbp_classes.map(per_igbp)

    nested = severity_classes.map(per_severity)
    flat   = nested.flatten()
    fc     = ee.FeatureCollection(flat)

    # Drop zero-area combinations to keep CSV manageable.
    # A fire that didn't burn any savanna at high severity doesn't need
    # that row. Filter threshold: > 0.01 km² (1 ha) to exclude noise pixels.
    return fc.filter(ee.Filter.gt('area_km2', 0.01))


# ─────────────────────────────────────────────────────────────
# ANNUAL TIME-SERIES  (v5 — EVI, baselines, IGBP, fire_year)
# ─────────────────────────────────────────────────────────────

def build_annual_timeseries(aoi, burn_mask, windows, fire_row,
                             pre_composite, post_composite):
    """
    Annual NDVI + EVI + NBR + CHIRPS rainfall time-series per fire.

    v5 ADDITIONS vs v4:
    ─────────────────────────────────────────────────────────────

    1. EVI added alongside NDVI each year.
       NDVI saturates in dense tropical canopy (LAI > 3). EVI does not.
       The NDVI–EVI gap in recovery curves is a signal of canopy density
       change — a forest recovering structurally shows EVI rising faster
       than NDVI.

    2. Pre/post NDVI and EVI baseline means stored per feature.
       pre_ndvi_mean, post_ndvi_mean, pre_evi_mean, post_evi_mean are
       computed from the SAME composites used for dNBR. These are constant
       across all years of the time-series for a given fire but stored on
       every row so that visualize_recovery.py can compute recovery
       completeness without a separate join:

         completeness = (yr_NDVI − post_NDVI) / (pre_NDVI − post_NDVI)

       Values at t=fire_year: completeness ≈ 0 (just burned)
       Values at t=fire_year+N: completeness → 1 when fully recovered

    3. igbp_lc_mode forwarded onto each feature.
       Enables direct stratification in visualize_recovery.py by IGBP
       class without joining back to merged_inventory.csv.

    4. fire_year stored on each feature.
       Avoids parsing burn_start_date to compute years_since_fire in Python.

    Parameters
    ──────────
    pre_composite  : ee.Image — same pre-fire composite used for dNBR
    post_composite : ee.Image — same post-fire composite used for dNBR
    """
    igbp_mode = int(float(fire_row.get('igbp_lc_mode', 0) or 0))
    fire_year = windows['fire_year']

    # ── Compute NDVI and EVI baselines from pre/post composites ────
    # These are server-side computations that become fixed properties
    # on every row — no .getInfo() call required here.
    pre_ndvi_img  = calc_ndvi(pre_composite).updateMask(burn_mask)
    post_ndvi_img = calc_ndvi(post_composite).updateMask(burn_mask)
    pre_evi_img   = calc_evi(pre_composite).updateMask(burn_mask)
    post_evi_img  = calc_evi(post_composite).updateMask(burn_mask)

    pre_ndvi_mean  = ee.Number(
        pre_ndvi_img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi,
            scale=20, bestEffort=True, tileScale=16
        ).get('NDVI'))
    post_ndvi_mean = ee.Number(
        post_ndvi_img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi,
            scale=20, bestEffort=True, tileScale=16
        ).get('NDVI'))
    pre_evi_mean   = ee.Number(
        pre_evi_img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi,
            scale=20, bestEffort=True, tileScale=16
        ).get('EVI'))
    post_evi_mean  = ee.Number(
        post_evi_img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi,
            scale=20, bestEffort=True, tileScale=16
        ).get('EVI'))

    ts_years = ee.List([float(y) for y in windows['ts_years']])

    def year_stats(yr):
        yr      = ee.Number(yr)
        start   = ee.Date.fromYMD(yr, 1, 1)
        end     = start.advance(1, 'year')
        yr_img  = build_composite(aoi, start, end, config.MAX_CLOUD_PCT)

        ndvi = calc_ndvi(yr_img).updateMask(burn_mask)
        nbr  = calc_nbr(yr_img).rename('NBR').updateMask(burn_mask)
        evi  = calc_evi(yr_img).updateMask(burn_mask)

        # Single reduceRegion call for all three indices — more efficient
        # than three separate calls because GEE fuses them server-side.
        stats = (ndvi.rename('NDVI')
                     .addBands(nbr)
                     .addBands(evi.rename('EVI'))
                     .reduceRegion(
                         reducer=ee.Reducer.mean(),
                         geometry=aoi, scale=20,
                         bestEffort=True, tileScale=16
                     ))

        # CHIRPS annual precipitation total → mean over AOI
        rain = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterBounds(aoi).filterDate(start, end)
                  .select('precipitation').sum().clip(aoi))
        rain_stats = rain.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi,
            scale=5566, bestEffort=True
        )

        return ee.Feature(None, {
            'fire_id':        fire_row['fire_id'],
            'biome':          fire_row.get('biome', ''),
            'igbp_lc_mode':   igbp_mode,                # NEW v5
            'fire_start':     fire_row.get('burn_start_date', ''),
            'fire_year':      fire_year,                  # NEW v5
            'year':           yr,
            'mean_NDVI':      stats.get('NDVI'),
            'mean_NBR':       stats.get('NBR'),
            'mean_EVI':       stats.get('EVI'),           # NEW v5
            'precip_mm':      rain_stats.get('precipitation'),
            # Baselines for recovery completeness (NEW v5)
            'pre_ndvi_mean':  pre_ndvi_mean,
            'post_ndvi_mean': post_ndvi_mean,
            'pre_evi_mean':   pre_evi_mean,
            'post_evi_mean':  post_evi_mean,
        })

    return ee.FeatureCollection(ts_years.map(year_stats))


# ─────────────────────────────────────────────────────────────
# SINGLE FIRE ANALYSIS  (v5)
# ─────────────────────────────────────────────────────────────

def analyse_fire(fire_row):
    """
    Full analysis pipeline for one fire event.

    v5 additions vs v4:
      - NDVI and EVI pre/post composites computed and exported as GeoTIFFs
      - pre/post composites passed into build_annual_timeseries()
      - build_severity_igbp_table() called → new CSV per fire
      - igbp_lc_mode and igbp_class_name logged and forwarded to all exports
    """
    fire_id    = fire_row['fire_id']
    fire_start = fire_row.get('burn_start_date', fire_row.get('fire_start', ''))
    fire_end   = fire_row.get('burn_end_date',   fire_row.get('fire_end', fire_start))

    if not fire_start:
        log.warning(f"  {fire_id}: missing burn_start_date — skipping.")
        return []

    # ── Log inventory-derived fields ───────────────────────────
    non_forest_frac  = fire_row.get('non_forest_frac', 'N/A')
    mean_slope_deg   = fire_row.get('mean_slope_deg',  'N/A')
    mean_elev_m      = fire_row.get('mean_elev_m',     'N/A')
    igbp_mode        = int(float(fire_row.get('igbp_lc_mode', 0) or 0))
    igbp_class_name  = fire_row.get('igbp_class_name',
                                    config.IGBP_CLASS_NAMES.get(igbp_mode, 'Unknown'))
    log.info(f"    IGBP={igbp_mode} ({igbp_class_name})  "
             f"non_forest={non_forest_frac}  "
             f"slope={mean_slope_deg}°  elev={mean_elev_m}m")

    # ── AOI ────────────────────────────────────────────────────
    try:
        lon = float(fire_row.get('centroid_lon', 0))
        lat = float(fire_row.get('centroid_lat', 0))
    except ValueError:
        log.warning(f"  {fire_id}: invalid centroid coordinates — skipping.")
        return []

    centroid = ee.Geometry.Point([lon, lat])
    aoi      = centroid.buffer(config.AOI_BUFFER_M)

    # ── Temporal windows ───────────────────────────────────────
    windows = build_date_windows(fire_start, fire_end)

    # ── Composites ─────────────────────────────────────────────
    pre_composite  = build_composite(aoi, windows['pre_start'],  windows['pre_end'],
                                     config.MAX_CLOUD_PCT)
    post_composite = build_composite(aoi, windows['post_start'], windows['post_end'],
                                     config.MAX_CLOUD_POST)
    rec_composite  = build_composite(aoi, windows['rec_start'],  windows['rec_end'],
                                     config.MAX_CLOUD_PCT)

    # ── dNBR + burn mask ───────────────────────────────────────
    dnbr              = calc_dnbr(pre_composite, post_composite)
    burn_mask, otsu_t = build_burn_mask(
        dnbr, aoi, min_patch_pixels=config.MIN_BURN_PATCH_PIXELS
    )
    burn_severity     = calc_burn_severity(dnbr).clip(aoi)

    # ── Spectral indices for export ────────────────────────────
    nbr_pre   = calc_nbr(pre_composite)
    nbr_post  = calc_nbr(post_composite)
    ndvi_pre  = calc_ndvi(pre_composite)   # NEW v5
    ndvi_post = calc_ndvi(post_composite)  # NEW v5
    evi_pre   = calc_evi(pre_composite)    # NEW v5
    evi_post  = calc_evi(post_composite)   # NEW v5
    ndvi_rec  = calc_ndvi(rec_composite)

    # ── RF classification on recovery composite ────────────────
    classified, _ = run_rf_classification(rec_composite, burn_mask, aoi)

    # ── Area table + severity-IGBP cross-table + time-series ──
    area_fc         = build_area_table(classified, aoi, fire_row, windows)
    severity_igbp_fc = build_severity_igbp_table(burn_severity, aoi, fire_row)  # NEW v5
    ts_fc           = build_annual_timeseries(
        aoi, burn_mask, windows, fire_row,
        pre_composite, post_composite   # NEW v5: pass composites for baselines
    )

    # ── Submit all exports ─────────────────────────────────────
    tasks = submit_fire_exports(
        fire_id=fire_id,
        folder=config.EXPORT_FOLDER,
        aoi=aoi,
        dnbr=dnbr,
        burn_severity=burn_severity,
        burn_mask=burn_mask,
        nbr_pre=nbr_pre,
        nbr_post=nbr_post,
        ndvi_pre=ndvi_pre,    # NEW v5
        ndvi_post=ndvi_post,  # NEW v5
        evi_pre=evi_pre,      # NEW v5
        evi_post=evi_post,    # NEW v5
        ndvi_rec=ndvi_rec,
        classified=classified,
        area_fc=area_fc,
        severity_igbp_fc=severity_igbp_fc,  # NEW v5
        timeseries_fc=ts_fc,
        max_concurrent=config.MAX_CONCURRENT_TASKS,
        poll_seconds=config.TASK_POLL_SECONDS,
        sleep_between=config.SLEEP_BETWEEN_FIRES,
    )

    return tasks


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    try:
        ee.Initialize(project=config.GEE_PROJECT)
        log.info("GEE initialised.")
    except Exception as e:
        log.error(f"GEE init failed: {e}")
        log.error("Run  ee.Authenticate()  once in a Python shell first.")
        sys.exit(1)

    os.makedirs(config.PER_FIRE_DIR, exist_ok=True)

    fires    = load_all_inventories(config.INVENTORY_DIR)
    if not fires:
        sys.exit(1)

    done_ids = load_checkpoint()
    pending  = [f for f in fires if f['fire_id'] not in done_ids]
    log.info(f"Total fires: {len(fires)}  |  Done: {len(done_ids)}  "
             f"|  Pending: {len(pending)}")
    log.info(f"RF feature stack     : 13 bands (spectral + topographic)")
    log.info(f"RF training points   : {config.RF_NUM_POINTS} per class")
    log.info(f"Min burn patch pixels: {config.MIN_BURN_PATCH_PIXELS}")
    log.info(f"EVI timeseries       : enabled (v5)")
    log.info(f"Severity×IGBP table  : enabled (v5)")

    all_tasks = []
    for idx, fire_row in enumerate(pending):
        fire_id = fire_row['fire_id']
        igbp_mode = int(float(fire_row.get('igbp_lc_mode', 0) or 0))
        igbp_name = config.IGBP_CLASS_NAMES.get(igbp_mode, 'Unknown')
        log.info(f"\n[{idx+1}/{len(pending)}]  {fire_id}  "
                 f"({fire_row.get('biome','')}  "
                 f"IGBP={igbp_mode}/{igbp_name}  "
                 f"{fire_row.get('burn_start_date','')}  "
                 f"area={fire_row.get('burn_area_km2','?')} km²)")
        try:
            tasks = analyse_fire(fire_row)
            if tasks:
                all_tasks.extend(tasks)
                save_checkpoint(fire_id)
                log.info(f"  Submitted {len(tasks)} export tasks.")
            else:
                log.warning(f"  No tasks submitted for {fire_id}.")
        except Exception as e:
            log.error(f"  FAILED: {fire_id}: {e}", exc_info=True)
            continue

        time.sleep(config.SLEEP_BETWEEN_FIRES)

    log.info(f"\n{'='*60}")
    log.info(f"Pipeline 2 (v5) complete.")
    log.info(f"Total fires processed : {len(pending)}")
    log.info(f"GEE export tasks      : {len(all_tasks)}")
    log.info(f"\nNEW exports in v5 (per fire):")
    log.info(f"  *_ndvi_pre_x10000.tif   — NDVI before fire")
    log.info(f"  *_ndvi_post_x10000.tif  — NDVI after fire")
    log.info(f"  *_evi_pre_x10000.tif    — EVI before fire")
    log.info(f"  *_evi_post_x10000.tif   — EVI after fire")
    log.info(f"  *_severity_igbp.csv     — area by severity×IGBP class")
    log.info(f"  *_annual_timeseries.csv — now includes mean_EVI, "
             f"pre/post baselines, fire_year, igbp_lc_mode")
    log.info(f"\nNEXT: run visualize_recovery.py")


if __name__ == '__main__':
    main()