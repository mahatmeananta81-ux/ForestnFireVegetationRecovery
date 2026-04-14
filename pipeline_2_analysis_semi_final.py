#!/usr/bin/env python3
# =============================================================
# pipeline_2_analysis.py  —  Per-Fire Deep Analysis  (v5.2)
#
# CHANGES FROM v5.1
# ──────────────────────────────────────────────────────────────
# ROBUST RESUME  (replaces simple checkpoint.txt)
#   v5 called save_checkpoint(fire_id) immediately after task
#   submission, before knowing whether GEE tasks would succeed.
#   Result: a fire whose tasks all failed was permanently skipped
#   on the next run because its ID was already in checkpoint.txt.
#
#   v5.2 uses a two-file system:
#     checkpoint.txt        — fire_ids whose tasks ALL completed
#     submitted_tasks.json  — {fire_id: [{desc, task_id}, ...]}
#                             for fires currently in GEE queue
#   At startup, reconcile_submitted_tasks() fetches live GEE
#   status for every tracked task ID:
#     All COMPLETED  → move fire_id to checkpoint, remove from JSON
#     Any FAILED     → remove from JSON (fire will be re-queued)
#     Any RUNNING    → keep in JSON (skip this fire this run)
#   This means a fire is never permanently skipped until every
#   one of its export tasks has reached COMPLETED state.
#
# FIX D  —  build_date_windows() caps future dates
#   Recovery windows for 2023 fires land in 2026 (future). All
#   three satellite tiers return 0 scenes → 0-band composite →
#   "Image.divide: Got 0 and 1" in calc_evi(). All end-dates
#   are now capped to DATA_CUTOFF (today minus 90 days).
#
# FIX E  —  burn mask empty check before export submission
#   When compute_otsu_threshold() falls back to 0.10 (degenerate
#   histogram), the resulting burn mask may be empty (no pixels
#   above threshold). Submitting 7 tasks against an empty mask
#   wastes EECU budget and fills the task queue with instant
#   failures. A cheap .count() reduceRegion check now gates all
#   export submissions. Empty-mask fires are logged to
#   failed_fires.csv and not retried automatically.
#
# TIMEOUT FIX  —  build_annual_timeseries() at 60 m scale
#   The 20 m reduceRegion scale over a 20 km buffer AOI
#   (3+ billion pixels per year) was the primary cause of the
#   21-minute timeout. Scale raised to TS_SCALE_M (default 60 m)
#   and CHIRPS uses its native 5566 m resolution.
#
# All v5 analytical features preserved unchanged.
# =============================================================

import ee
import os
import sys
import csv
import json
import time
import logging
from datetime import datetime, timedelta

# ── CSV field size limit ───────────────────────────────────────
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.gee_functions import (
    build_composite, calc_dnbr, calc_nbr, calc_ndvi, calc_evi,
    calc_burn_severity, build_burn_mask, build_feature_stack
)
from utils.export_helpers import (
    submit_fire_exports, wait_for_capacity, get_running_task_count
)

# ── Logging ───────────────────────────────────────────────────
os.makedirs(os.path.join(config.BASE_DIR, 'outputs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(
            os.path.join(config.BASE_DIR, 'outputs', 'pipeline2.log')),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────
# DATA_CUTOFF: no satellite data exists beyond today - 90 days.
# Prevents requesting future composites that return 0-band images.
DATA_CUTOFF     = (datetime.today() - timedelta(days=90)).strftime('%Y-%m-%d')
DATA_CUTOFF_EE  = None   # initialised lazily after ee.Initialize()

# Timeseries scale — 60 m prevents timeout on large Amazon AOIs.
# NDVI/EVI/NBR mean values converge at 60 m for areas ≥ 50 km².
TS_SCALE_M   = getattr(config, 'TS_SCALE_M',   60)
MAX_TS_YEARS = getattr(config, 'MAX_TS_YEARS',   5)

# Paths
CHECKPOINT_FILE      = config.CHECKPOINT_FILE
SUBMITTED_TASKS_FILE = os.path.join(config.BASE_DIR, 'outputs', 'submitted_tasks.json')
FAILED_FIRES_FILE    = os.path.join(config.BASE_DIR, 'outputs','failed_fires.csv')


# ─────────────────────────────────────────────────────────────
# ROBUST CHECKPOINT SYSTEM
# ─────────────────────────────────────────────────────────────

def load_checkpoint():
    """Return set of fire_ids whose tasks ALL completed successfully."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE) as f:
        return {line.strip() for line in f if line.strip()}


def save_checkpoint(fire_id):
    """Mark a fire as fully complete."""
    with open(CHECKPOINT_FILE, 'a') as f:
        f.write(fire_id + '\n')


def load_submitted_tasks():
    """
    Load the submitted-tasks registry.
    Returns dict: {fire_id: [{'desc': str, 'task_id': str}, ...]}
    """
    if not os.path.exists(SUBMITTED_TASKS_FILE):
        return {}
    try:
        with open(SUBMITTED_TASKS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        log.warning("submitted_tasks.json unreadable — starting fresh.")
        return {}


def save_submitted_tasks(tasks_dict):
    """Persist the submitted-tasks registry atomically."""
    tmp = SUBMITTED_TASKS_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(tasks_dict, f, indent=2)
    os.replace(tmp, SUBMITTED_TASKS_FILE)


def log_failed_fire(fire_id, reason):
    """Append a fire to failed_fires.csv for human review."""
    write_header = not os.path.exists(FAILED_FIRES_FILE)
    with open(FAILED_FIRES_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['fire_id', 'reason', 'timestamp'])
        w.writerow([fire_id, reason,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')])


def reconcile_submitted_tasks(submitted):
    """
    Check GEE live status for every task in submitted_tasks.json.

    Calls ee.data.getTaskList() once and builds a lookup by task ID.
    This is O(1) per fire regardless of how many tasks are tracked.

    Returns three sets:
      still_running   — fire_ids with at least one task RUNNING/READY
      newly_completed — fire_ids where ALL tasks reached COMPLETED
      newly_failed    — fire_ids where at least one task FAILED
                        (or CANCELLED, or disappeared from GEE history)

    Fires in newly_failed will be re-queued on the next loop iteration.
    Fires in still_running are skipped this run to avoid double-submit.
    """
    if not submitted:
        return set(), set(), set()

    # Collect all known task IDs
    all_ids = [t['task_id']
               for tasks in submitted.values()
               for t in tasks
               if t.get('task_id')]

    if not all_ids:
        return set(), set(), set()

    # Fetch live GEE status — one network call for all IDs
    log.info(f"Reconciling {len(all_ids)} GEE tasks from previous runs...")
    try:
        all_tasks = ee.data.getTaskList()
        status_map = {t['id']: t['state'] for t in all_tasks}
    except Exception as e:
        log.warning(f"Could not fetch GEE task list: {e}. "
                    f"Treating all submitted fires as still-running.")
        return set(submitted.keys()), set(), set()

    still_running   = set()
    newly_completed = set()
    newly_failed    = set()

    for fire_id, task_list in submitted.items():
        states = []
        for t in task_list:
            tid   = t.get('task_id', '')
            state = status_map.get(tid, 'UNKNOWN')
            states.append(state)
            if state == 'FAILED':
                desc = t.get('desc', tid)
                log.info(f"  FAILED task: {desc}")
            elif state == 'UNKNOWN':
                log.debug(f"  Task not in GEE history: {tid} (may be >30 days old)")

        if all(s == 'COMPLETED' for s in states):
            newly_completed.add(fire_id)
        elif any(s in ('RUNNING', 'READY', 'SUBMITTED') for s in states):
            still_running.add(fire_id)
        else:
            # Any FAILED, CANCELLED, UNKNOWN → re-queue
            newly_failed.add(fire_id)

    log.info(f"  Reconciliation: "
             f"{len(newly_completed)} newly completed, "
             f"{len(still_running)} still running, "
             f"{len(newly_failed)} failed/unknown → will rerun")
    return still_running, newly_completed, newly_failed


def register_submitted_tasks(submitted_dict, fire_id, tasks):
    """
    Record GEE task IDs for a just-submitted fire.
    Reads task ID from task.status() with a short retry for timing.
    """
    task_records = []
    for task in tasks:
        task_id = None
        for attempt in range(5):
            try:
                status  = task.status()
                task_id = status.get('id') or status.get('task_id')
                if task_id:
                    break
            except Exception:
                pass
            time.sleep(1)
        task_records.append({
            'desc':    task.status().get('description', 'unknown'),
            'task_id': task_id or 'unknown',
        })
    submitted_dict[fire_id] = task_records
    save_submitted_tasks(submitted_dict)


# ─────────────────────────────────────────────────────────────
# INVENTORY LOADER  (unchanged from v5)
# ─────────────────────────────────────────────────────────────

def load_all_inventories(inventory_dir):
    """
    Load fire inventory for Pipeline 2.
    Reads merged_inventory_screened.csv if it exists (output of
    screen_cloudy_fires.py), falling back to merged_inventory.csv,
    then individual inv_*.csv files.
    """
    if not os.path.exists(inventory_dir):
        log.error(f"Inventory directory not found: {inventory_dir}")
        return []

    # Prefer screened inventory
    for fname in ['merged_inventory_screened.csv', 'merged_inventory.csv']:
        path = os.path.join(inventory_dir, fname)
        if os.path.exists(path):
            log.info(f"Loading inventory: {path}")
            with open(path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows   = list(reader)
            for i, row in enumerate(rows):
                if not row.get('fire_id'):
                    row['fire_id'] = (f"{row.get('biome','unk')}_"
                                      f"{row.get('year','0')}_{i:05d}")
            log.info(f"Loaded {len(rows)} fire events.")
            return rows

    log.warning("No merged inventory found. Reading raw per-tile CSVs.")
    csv_files = [f for f in os.listdir(inventory_dir)
                 if f.endswith('.csv') and not f.startswith('merged')]
    if not csv_files:
        log.error(f"No CSV files found in {inventory_dir}")
        return []

    all_fires = []
    for fname in sorted(csv_files):
        fpath = os.path.join(inventory_dir, fname)
        with open(fpath, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        for i, row in enumerate(rows):
            biome = row.get('biome', fname.replace('.csv', ''))
            year  = row.get('year', 'unknown')
            row['fire_id'] = f"{biome}_{year}_{i:05d}"
            all_fires.append(row)

    log.info(f"Loaded {len(all_fires)} fires from {len(csv_files)} raw CSVs.")
    return all_fires


# ─────────────────────────────────────────────────────────────
# DATE CUTOFF HELPER
# ─────────────────────────────────────────────────────────────

def _get_cutoff_ee():
    """Return DATA_CUTOFF as ee.Date (lazy init after ee.Initialize)."""
    global DATA_CUTOFF_EE
    if DATA_CUTOFF_EE is None:
        DATA_CUTOFF_EE = ee.Date(DATA_CUTOFF)
    return DATA_CUTOFF_EE


def _cap(ee_date):
    """
    Clamp an ee.Date to DATA_CUTOFF (today - 90 days).
    Prevents requesting satellite data that doesn't exist yet.
    GEE's Algorithms.If is lazy — only the taken branch is evaluated.
    """
    cutoff = _get_cutoff_ee()
    return ee.Date(
        ee.Algorithms.If(
            ee_date.difference(cutoff, 'day').gt(0),
            cutoff,
            ee_date
        )
    )


# ─────────────────────────────────────────────────────────────
# DATE WINDOW BUILDER  (FIX D — caps future dates)
# ─────────────────────────────────────────────────────────────

def build_date_windows(fire_start_str, fire_end_str):
    """
    Build temporal analysis windows from MCD64A1 DOY-derived dates.

    FIX D: All window end-dates are capped to DATA_CUTOFF.
    For 2023 fires with RECOVERY_YEARS_AFTER=3, the recovery
    window would fall in mid-2026 — no satellite has data yet.
    Previously this caused 0-band composites and "Image.divide:
    Got 0 and 1" errors in calc_evi(). The cap ensures every
    window only requests dates with actual satellite coverage.

    ts_years is also capped to yesterday's year so the annual
    timeseries never requests a year with incomplete data.
    """
    fire_start = ee.Date(fire_start_str)
    if not fire_end_str or fire_end_str == fire_start_str:
        fire_end_str = fire_start_str
    fire_end = ee.Date(fire_end_str)

    pre_start = fire_start.advance(-1, 'year').advance(-45, 'day')
    pre_end   = _cap(fire_start.advance(-1, 'year').advance(45, 'day'))

    post_start = fire_end.advance(config.POST_FIRE_DAYS_START, 'day')
    post_end   = _cap(fire_end.advance(config.POST_FIRE_DAYS_END, 'day'))

    rec_start = fire_start.advance(config.RECOVERY_YEARS_AFTER,
                                    'year').advance(-3, 'month')
    rec_end   = _cap(fire_start.advance(config.RECOVERY_YEARS_AFTER,
                                         'year').advance(3, 'month'))

    fire_year = int(fire_start_str[:4])
    ts_start  = fire_year - 1
    ts_end    = min(
        fire_year + config.RECOVERY_YEARS_AFTER + 1,
        datetime.today().year - 1   # cap to last complete year
    )

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
# BURN MASK EMPTY CHECK  (FIX E)
# ─────────────────────────────────────────────────────────────

def burn_mask_has_pixels(burn_mask, aoi):
    """
    Check whether the burn mask contains any valid pixels.
    Returns True if ≥ 1 pixel is marked as burned, else False.

    FIX E: Called after build_burn_mask() and before submitting
    exports. An empty mask means:
      - Otsu fell back to 0.10 but dNBR is uniformly below that
        (very low-severity fire or zero-fill composite)
      - OR the recovery composite was zero-filled (future date)
    In both cases all 7 downstream products are meaningless.
    Skipping saves EECU budget and keeps the task queue clean.

    Cost: one .getInfo() call (~1-2 s). Worth it to avoid
    submitting tasks that all fail in <2 s each.
    """
    try:
        result = (burn_mask
                    .reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=aoi,
                        scale=60,          # coarse — need count > 0 only
                        bestEffort=True,
                        maxPixels=int(1e6)
                    ).getInfo())
        count = int(result.get('BurnMask', 0) or 0)
        return count > 0
    except Exception as e:
        # If the check itself fails, proceed optimistically.
        # The export will fail fast (<2 s) if mask is truly empty.
        log.warning(f"    burn_mask_has_pixels check failed ({e}) — proceeding.")
        return True


# ─────────────────────────────────────────────────────────────
# RF LAND COVER CLASSIFICATION  (unchanged from v5)
# ─────────────────────────────────────────────────────────────

def run_rf_classification(rec_img, burn_mask, aoi):
    """Train + apply Random Forest on 13-band feature stack."""
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
# RECOVERY AREA TABLE  (unchanged from v5)
# ─────────────────────────────────────────────────────────────

def build_area_table(classified, aoi, fire_row, windows):
    """Compute area (km²) per RF land-cover class within the burn scar."""
    class_ids   = ee.List([1, 2, 3, 4, 5])
    class_names = ee.List(list(config.CLASS_NAMES.values()))
    igbp_mode   = int(float(fire_row.get('igbp_lc_mode', 0) or 0))

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
            'ClassID':      class_val,
            'ClassName':    class_names.get(idx),
            'Area_km2':     ee.Number(sq_m.get('LandCover')).divide(1e6),
            'fire_id':      fire_row['fire_id'],
            'biome':        fire_row.get('biome', ''),
            'igbp_lc_mode': igbp_mode,
            'fire_start':   fire_row.get('burn_start_date', ''),
            'rec_year':     windows['rec_start'].format('YYYY'),
        })

    return ee.FeatureCollection(class_ids.map(area_for_class))


# ─────────────────────────────────────────────────────────────
# BURN SEVERITY × IGBP CROSS-TABLE  (unchanged from v5)
# ─────────────────────────────────────────────────────────────

def build_severity_igbp_table(burn_severity, aoi, fire_row):
    """Compute burned area (km²) per severity class × IGBP class."""
    lc_year   = int(float(fire_row.get('year', 2019) or 2019)) - 1
    igbp_img  = (ee.ImageCollection('MODIS/061/MCD12Q1')
                   .filterDate(f'{lc_year}-01-01', f'{lc_year}-12-31')
                   .first().select('LC_Type1'))
    fire_id   = fire_row['fire_id']
    biome     = fire_row.get('biome', '')
    igbp_mode = int(float(fire_row.get('igbp_lc_mode', 0) or 0))

    severity_classes = ee.List([1, 2, 3, 4, 5, 6])
    igbp_classes     = ee.List(list(range(1, 18)))

    def per_severity(sev):
        sev = ee.Number(sev)
        def per_igbp(igbp):
            igbp     = ee.Number(igbp)
            combined = burn_severity.eq(sev).And(igbp_img.eq(igbp))
            area_m2  = (combined.multiply(ee.Image.pixelArea())
                                .reduceRegion(
                                    reducer=ee.Reducer.sum(),
                                    geometry=aoi, scale=20,
                                    bestEffort=True, tileScale=16
                                ))
            return ee.Feature(None, {
                'fire_id':        fire_id,
                'biome':          biome,
                'igbp_lc_mode':   igbp_mode,
                'severity_class': sev,
                'igbp_class':     igbp,
                'area_km2':       ee.Number(
                    area_m2.get('BurnSeverity')).divide(1e6),
            })
        return igbp_classes.map(per_igbp)

    flat = severity_classes.map(per_severity).flatten()
    return ee.FeatureCollection(flat).filter(ee.Filter.gt('area_km2', 0.01))


# ─────────────────────────────────────────────────────────────
# ANNUAL TIME-SERIES  (TIMEOUT FIX — 60 m scale, CHIRPS native)
# ─────────────────────────────────────────────────────────────

def build_annual_timeseries(aoi, burn_mask, windows, fire_row,
                             pre_composite, post_composite):
    """
    Annual NDVI + EVI + NBR + CHIRPS time-series.

    TIMEOUT FIX vs v5:
      - TS_SCALE_M = 60 m (was 20 m) → 9× fewer pixels per reduceRegion.
        At 20 km buffer: 20 m = 3.1B pixels/year → 60 m = 347M pixels/year.
      - CHIRPS at native 5566 m (was output scale=20 m which forced GEE
        to generate billions of synthetic resampled pixels per year).
      - ts_years capped at MAX_TS_YEARS = 5 to bound total compute.
      - Single fused reduceRegion for NDVI + NBR + EVI per year.
      - build_composite() now has MODIS fallback + zero-band safety
        (FIX C in gee_functions.py) so empty images no longer occur.
    """
    igbp_mode = int(float(fire_row.get('igbp_lc_mode', 0) or 0))
    fire_year = windows['fire_year']

    # Baseline means from pre/post composites (same as v5)
    pre_ndvi_img   = calc_ndvi(pre_composite).updateMask(burn_mask)
    post_ndvi_img  = calc_ndvi(post_composite).updateMask(burn_mask)
    pre_evi_img    = calc_evi(pre_composite).updateMask(burn_mask)
    post_evi_img   = calc_evi(post_composite).updateMask(burn_mask)

    def _mean(img, band):
        return ee.Number(img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi,
            scale=TS_SCALE_M, bestEffort=True, tileScale=4
        ).get(band))

    pre_ndvi_mean  = _mean(pre_ndvi_img,  'NDVI')
    post_ndvi_mean = _mean(post_ndvi_img, 'NDVI')
    pre_evi_mean   = _mean(pre_evi_img,   'EVI')
    post_evi_mean  = _mean(post_evi_img,  'EVI')

    # Cap ts_years to MAX_TS_YEARS, keeping fire_year-1 and post-fire years
    ts_years_raw = windows['ts_years']
    if len(ts_years_raw) > MAX_TS_YEARS:
        try:
            fi = ts_years_raw.index(fire_year)
        except ValueError:
            fi = 1
        si = max(0, fi - 1)
        ts_years_raw = ts_years_raw[si: si + MAX_TS_YEARS]

    ts_years = ee.List([float(y) for y in ts_years_raw])

    def year_stats(yr):
        yr     = ee.Number(yr)
        yr_int = yr.int()
        start  = ee.Date.fromYMD(yr_int, 1, 1)
        end    = _cap(start.advance(1, 'year'))   # FIX D applied here too

        # build_composite has MODIS fallback + zero-band safety (FIX C)
        yr_img = build_composite(aoi, start, end, config.MAX_CLOUD_PCT)

        ndvi = calc_ndvi(yr_img).updateMask(burn_mask)
        nbr  = calc_nbr(yr_img).rename('NBR').updateMask(burn_mask)
        evi  = calc_evi(yr_img).updateMask(burn_mask)

        # Single fused reduceRegion for all three indices
        stats = (ndvi.rename('NDVI')
                     .addBands(nbr)
                     .addBands(evi.rename('EVI'))
                     .reduceRegion(
                         reducer=ee.Reducer.mean(),
                         geometry=aoi,
                         scale=TS_SCALE_M,
                         bestEffort=True,
                         tileScale=4
                     ))

        # CHIRPS at native 5566 m — fast, correct
        rain = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterBounds(aoi).filterDate(start, end)
                  .select('precipitation').sum().clip(aoi))
        rain_stats = rain.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=5566,
            bestEffort=True
        )

        return ee.Feature(None, {
            'fire_id':        fire_row['fire_id'],
            'biome':          fire_row.get('biome', ''),
            'igbp_lc_mode':   igbp_mode,
            'fire_start':     fire_row.get('burn_start_date', ''),
            'fire_year':      fire_year,
            'year':           yr,
            'mean_NDVI':      stats.get('NDVI'),
            'mean_NBR':       stats.get('NBR'),
            'mean_EVI':       stats.get('EVI'),
            'precip_mm':      rain_stats.get('precipitation'),
            'pre_ndvi_mean':  pre_ndvi_mean,
            'post_ndvi_mean': post_ndvi_mean,
            'pre_evi_mean':   pre_evi_mean,
            'post_evi_mean':  post_evi_mean,
        })

    return ee.FeatureCollection(ts_years.map(year_stats))


# ─────────────────────────────────────────────────────────────
# SINGLE FIRE ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyse_fire(fire_row):
    """
    Full analysis pipeline for one fire event.
    Returns list of submitted GEE task objects, or [] on skip.
    """
    fire_id    = fire_row['fire_id']
    fire_start = fire_row.get('burn_start_date',
                               fire_row.get('fire_start', ''))
    fire_end   = fire_row.get('burn_end_date',
                               fire_row.get('fire_end', fire_start))

    if not fire_start:
        log.warning(f"  {fire_id}: missing burn_start_date — skipping.")
        log_failed_fire(fire_id, 'missing_burn_start_date')
        return []

    igbp_mode       = int(float(fire_row.get('igbp_lc_mode', 0) or 0))
    igbp_class_name = fire_row.get(
        'igbp_class_name',
        config.IGBP_CLASS_NAMES.get(igbp_mode, 'Unknown'))
    log.info(f"    IGBP={igbp_mode} ({igbp_class_name})  "
             f"non_forest={fire_row.get('non_forest_frac','N/A')}  "
             f"slope={fire_row.get('mean_slope_deg','N/A')}°  "
             f"elev={fire_row.get('mean_elev_m','N/A')}m")

    # ── AOI ───────────────────────────────────────────────────
    try:
        lon = float(fire_row.get('centroid_lon', 0))
        lat = float(fire_row.get('centroid_lat', 0))
    except ValueError:
        log.warning(f"  {fire_id}: invalid centroid — skipping.")
        log_failed_fire(fire_id, 'invalid_centroid')
        return []

    if lon == 0.0 and lat == 0.0:
        log.warning(f"  {fire_id}: null centroid (0,0) — skipping.")
        log_failed_fire(fire_id, 'null_centroid')
        return []

    centroid = ee.Geometry.Point([lon, lat])
    aoi      = centroid.buffer(distance=config.AOI_BUFFER_M, maxError=1)

    # ── Temporal windows (FIX D) ──────────────────────────────
    windows = build_date_windows(fire_start, fire_end)

    # ── Composites ────────────────────────────────────────────
    # build_composite now has three-tier fallback + zero-band safety (FIX C)
    pre_composite  = build_composite(aoi, windows['pre_start'],
                                     windows['pre_end'],
                                     config.MAX_CLOUD_PCT)
    post_composite = build_composite(aoi, windows['post_start'],
                                     windows['post_end'],
                                     config.MAX_CLOUD_POST)
    rec_composite  = build_composite(aoi, windows['rec_start'],
                                     windows['rec_end'],
                                     config.MAX_CLOUD_PCT)

    # ── dNBR + burn mask ──────────────────────────────────────
    dnbr              = calc_dnbr(pre_composite, post_composite)
    burn_mask, otsu_t = build_burn_mask(
        dnbr, aoi, min_patch_pixels=config.MIN_BURN_PATCH_PIXELS
    )
    burn_severity     = calc_burn_severity(dnbr).clip(aoi)

    # ── FIX E: gate all exports on burn mask having pixels ────
    # compute_otsu_threshold() now falls back to 0.10 instead of
    # crashing, but the resulting mask may still be empty.
    # One cheap .getInfo() here saves 7 doomed export tasks.
    # ── FIX E: gate all exports on burn mask having pixels ────
    if not burn_mask_has_pixels(burn_mask, aoi):
        log.warning(f"  {fire_id}: burn mask empty — skipping exports.")
        log_failed_fire(fire_id, 'empty_burn_mask')
        return []

    # ── Spectral indices ──────────────────────────────────────
    nbr_pre   = calc_nbr(pre_composite)
    nbr_post  = calc_nbr(post_composite)
    ndvi_pre  = calc_ndvi(pre_composite)
    ndvi_post = calc_ndvi(post_composite)
    evi_pre   = calc_evi(pre_composite)
    evi_post  = calc_evi(post_composite)
    ndvi_rec  = calc_ndvi(rec_composite)

    # ── RF classification ─────────────────────────────────────
    classified, _ = run_rf_classification(rec_composite, burn_mask, aoi)

    # ── Feature collections ───────────────────────────────────
    area_fc          = build_area_table(classified, aoi, fire_row, windows)
    severity_igbp_fc = build_severity_igbp_table(burn_severity, aoi, fire_row)
    ts_fc            = build_annual_timeseries(
        aoi, burn_mask, windows, fire_row,
        pre_composite, post_composite
    )

    # ── Submit exports ────────────────────────────────────────
    tasks = submit_fire_exports(
        fire_id=fire_id,
        folder=config.EXPORT_FOLDER,
        aoi=aoi,
        dnbr=dnbr,
        burn_severity=burn_severity,
        burn_mask=burn_mask,
        nbr_pre=nbr_pre,
        nbr_post=nbr_post,
        ndvi_pre=ndvi_pre,
        ndvi_post=ndvi_post,
        evi_pre=evi_pre,
        evi_post=evi_post,
        ndvi_rec=ndvi_rec,
        classified=classified,
        area_fc=area_fc,
        severity_igbp_fc=severity_igbp_fc,
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
        log.error("Run ee.Authenticate() once first.")
        sys.exit(1)

    os.makedirs(config.PER_FIRE_DIR, exist_ok=True)

    fires = load_all_inventories(config.INVENTORY_DIR)
    if not fires:
        sys.exit(1)

    # ── Robust resume ─────────────────────────────────────────
    done_ids  = load_checkpoint()
    submitted = load_submitted_tasks()

    still_running, newly_completed, newly_failed = \
        reconcile_submitted_tasks(submitted)

    # Promote newly completed → checkpoint
    for fire_id in newly_completed:
        if fire_id not in done_ids:
            save_checkpoint(fire_id)
            done_ids.add(fire_id)
        submitted.pop(fire_id, None)

    # Remove failed from submitted so they are re-queued below
    for fire_id in newly_failed:
        submitted.pop(fire_id, None)
        log.info(f"  Re-queuing failed fire: {fire_id}")

    save_submitted_tasks(submitted)

    # Skip: fully done OR currently running in GEE
    skip_ids = done_ids | still_running
    pending  = [f for f in fires if f['fire_id'] not in skip_ids]

    log.info(f"Total fires         : {len(fires)}")
    log.info(f"Fully completed     : {len(done_ids)}")
    log.info(f"Still running in GEE: {len(still_running)}")
    log.info(f"Re-queued (failed)  : {len(newly_failed)}")
    log.info(f"Pending this run    : {len(pending)}")
    log.info(f"DATA_CUTOFF (no future dates beyond): {DATA_CUTOFF}")
    log.info(f"TS_SCALE_M          : {TS_SCALE_M} m")
    log.info(f"MAX_TS_YEARS        : {MAX_TS_YEARS}")

    for idx, fire_row in enumerate(pending):
        fire_id   = fire_row['fire_id']
        igbp_mode = int(float(fire_row.get('igbp_lc_mode', 0) or 0))
        igbp_name = config.IGBP_CLASS_NAMES.get(igbp_mode, 'Unknown')

        log.info(
            f"\n[{idx+1}/{len(pending)}]  {fire_id}  "
            f"({fire_row.get('biome','')}  "
            f"IGBP={igbp_mode}/{igbp_name}  "
            f"{fire_row.get('burn_start_date','')}  "
            f"area={fire_row.get('burn_area_km2','?')} km²)"
        )

        try:
            tasks = analyse_fire(fire_row)
        except Exception as e:
            log.error(f"  FAILED: {fire_id}: {e}", exc_info=True)
            log_failed_fire(fire_id, f'analyse_fire_exception: {e}')
            continue

        if tasks:
            # Track submitted task IDs — enables GEE status reconciliation
            # on the next run so we know whether to checkpoint or re-queue.
            register_submitted_tasks(submitted, fire_id, tasks)
            log.info(f"  Submitted {len(tasks)} export tasks. "
                     f"Tracked in submitted_tasks.json.")
        else:
            # analyse_fire returned [] — fire was skipped (empty burn mask,
            # missing date, etc.). Already logged to failed_fires.csv.
            log.info(f"  Skipped (no tasks submitted).")

        time.sleep(config.SLEEP_BETWEEN_FIRES)

    # ── Final reconciliation ───────────────────────────────────
    # Do one more pass so tasks submitted in THIS run that complete
    # quickly (small fires) are checkpointed before we exit.
    log.info("\nFinal reconciliation of this run's tasks...")
    time.sleep(30)   # brief wait for GEE to register new tasks
    submitted = load_submitted_tasks()
    still_running, newly_completed, newly_failed = \
        reconcile_submitted_tasks(submitted)

    for fire_id in newly_completed:
        if fire_id not in done_ids:
            save_checkpoint(fire_id)
            done_ids.add(fire_id)
        submitted.pop(fire_id, None)
    for fire_id in newly_failed:
        submitted.pop(fire_id, None)
    save_submitted_tasks(submitted)

    log.info(f"\n{'='*60}")
    log.info(f"Pipeline 2 (v5.2) run complete.")
    log.info(f"  Checkpoint (all tasks done): {len(done_ids)} fires")
    log.info(f"  Still in GEE queue        : {len(still_running)} fires")
    log.info(f"  Failed (see failed_fires.csv): "
             f"{sum(1 for _ in open(FAILED_FIRES_FILE)) - 1 if os.path.exists(FAILED_FIRES_FILE) else 0}")
    log.info(f"\nRe-run pipeline_2.py at any time to:")
    log.info(f"  1. Checkpoint newly-completed GEE tasks")
    log.info(f"  2. Re-queue any that failed")
    log.info(f"  3. Submit the next batch of pending fires")


if __name__ == '__main__':
    main()
