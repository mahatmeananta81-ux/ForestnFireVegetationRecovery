#!/usr/bin/env python3
# =============================================================
# pipeline_1_inventory.py  —  Fire Inventory  (v5)
#
# CHANGES FROM v4 (bug-fix + forest type addition)
# ──────────────────────────────────────────────────────────────
#
# BUG FIX 1 — ee.Algorithms.If null check was wrong for forest_frac
#   The v4 fix used ee.Algorithms.If(forest_frac_raw, ...) which treats
#   0.0 as falsy. A polygon that is entirely non-forested has
#   forest_frac = 0.0 — a valid measurement being wrongly overwritten
#   with 0.5. Fixed with null_safe_number() using IsEqual(None).
#   Same fix applied to slope and elev null guards.
#
# BUG FIX 2 — doy_span_flag dropped from compute_derived
#   Silently removed during the v4 null-safety rewrite. Restored.
#   postprocess.py uses this to distinguish merged events (flag=2)
#   from within-month possible merges (flag=1).
#
# BUG FIX 3 — Empty T21 fields in exported CSVs
#   firms_col.sum() on a fully masked image returns null, not 0.
#   Fix: .unmask(0) on both T21_sum and T21_mean before reduceRegions.
#   Now: 0 = no confident FIRMS detections; empty cell = never.
#
# BUG FIX 4 — min().combine(max()) produces wrong property names
#   v5 initially replaced minMax() with min().combine(max(), sharedInputs=True).
#   GEE's combine() resolves name collisions by appending '_1', producing
#   'BurnDate' (min) and 'BurnDate_1' (max) — not 'BurnDate_min'/'BurnDate_max'.
#   The notNull filter on those names then dropped ALL features → count=0.
#   Fix: use ee.Reducer.minMax() directly — the documented way to get _min/_max.
#
# BUG FIX 5 — Duplicate notNull filter removed
#   The notNull filter on BurnDate_min/max appeared twice in v4.
#   Kept only the single instance after DOY reduceRegions.
#
# NEW FEATURE — Pre-fire forest type from MODIS MCD12Q1 IGBP
#   Each polygon now carries igbp_lc_mode: the dominant IGBP land
#   cover class within the burn polygon for the year BEFORE the fire.
#   Why MCD12Q1 not WorldCover?
#     WorldCover v200 is a single 2021 epoch. For fires in 2019-2020,
#     pixels deforested before 2021 are classified by post-deforestation
#     type — wrong for pre-fire context. MCD12Q1 is annual (2001–present)
#     at 500m, temporally matched to each fire year.
#   Stored as integer 1–17. Decoded with config.IGBP_CLASS_NAMES.
#
# All v3 fixes and v4 NESAC improvements are preserved unchanged.
# =============================================================

import ee
import time
import logging
import os
import sys
import calendar
import concurrent.futures
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.export_helpers import wait_for_capacity, export_table_to_drive

# ── Logging ────────────────────────────────────────────────────
os.makedirs(os.path.join(config.BASE_DIR, 'outputs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(
            os.path.join(config.BASE_DIR, 'outputs', 'pipeline1.log')),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# GRID CHUNKING  (v3 Fix 4 — unchanged)
# ─────────────────────────────────────────────────────────────

def grid_bbox(bbox, step_deg=10.0):
    """
    Split [lon_min, lat_min, lon_max, lat_max] into ≤step_deg sub-tiles.
    Prevents per-task memory overflow on large biomes during reduceToVectors.
    10°×10° at equator ≈ 1.2M km² ≈ 4.8M MODIS pixels — within GEE limits.
    """
    lon_min, lat_min, lon_max, lat_max = bbox

    lon_steps = []
    cur = lon_min
    while cur < lon_max:
        lon_steps.append((cur, min(cur + step_deg, lon_max)))
        cur += step_deg

    lat_steps = []
    cur = lat_min
    while cur < lat_max:
        lat_steps.append((cur, min(cur + step_deg, lat_max)))
        cur += step_deg

    tiles = []
    for (lo0, lo1), (la0, la1) in product(lon_steps, lat_steps):
        tiles.append([lo0, la0, lo1, la1])
    return tiles


# ─────────────────────────────────────────────────────────────
# FIRE SEASON MONTHS PER BIOME
# ─────────────────────────────────────────────────────────────

BIOME_FIRE_MONTHS = {
    # southeast_asia covers BOTH:
    #   Mainland (Myanmar/Thailand/Vietnam): Jan-May dry season
    #   Maritime (Borneo/Sumatra/Papua): Jul-Oct dry season
    "south_asia":      list(range(1, 6)),
    "southeast_asia":  list(range(1, 6)) + list(range(7, 11)),
    "amazon":          list(range(6, 12)),
    "cerrado":         list(range(6, 11)),
    "west_africa":     list(range(11, 13)) + list(range(1, 4)),
    "east_africa":     list(range(1, 4))   + list(range(6, 10)),
    "southern_africa": list(range(6, 11)),
    "boreal_canada":   list(range(5, 10)),
    "boreal_russia":   list(range(5, 10)),
    "scandinavia":     list(range(4, 9)),
    "australia_east":  list(range(9, 13)) + list(range(1, 3)),
    "australia_sw":    list(range(11, 13)) + list(range(1, 4)),
    "mediterranean":   list(range(6, 10)),
    "western_usa":     list(range(5, 11)),
    "mexico":          list(range(2, 7)),
}


# ─────────────────────────────────────────────────────────────
# NULL-SAFE NUMERIC HELPER
# ─────────────────────────────────────────────────────────────

def null_safe_number(value, default):
    """
    Return value as ee.Number if not null, otherwise return default.

    BUG FIX 1: Uses ee.Algorithms.IsEqual(value, None) as the null test.
    This is the ONLY correct server-side null check in GEE.

    DO NOT use ee.Algorithms.If(value, ...) — it treats 0.0, 0, and
    False as falsy, silently replacing valid zero measurements with
    the default. For example:
      - forest_frac=0.0 (fully non-forested burn) would become 0.5
      - slope=0.0 (perfectly flat terrain) would become 0.0 only by
        coincidence, but the semantics are wrong
    """
    return ee.Number(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(value, None),
            default,
            value
        )
    )


# ─────────────────────────────────────────────────────────────
# METHOD 1: MCD64A1 CLUSTERING
# ─────────────────────────────────────────────────────────────

def discover_fires_mcd64(biome_name, tile_bbox, year, month,
                          min_pixels, max_doy_gap, max_clusters):
    """
    Cluster MCD64A1 burned-area pixels into fire polygons for one
    (biome tile, year, month). Returns a FeatureCollection.

    All reduceRegions calls are server-side single-pass operations
    (v3 Fix 1). tileScale=16 on all reduces (v3 Fix 3). Polygons
    are simplified immediately after vectorisation (v3 Fix 2).

    Output CSV fields per polygon:
      burn_pixel_count  — number of 500m MODIS burned pixels
      burn_doy_min      — first burned DOY (fire front arrival)
      burn_doy_max      — last burned DOY (last pixel extinguished)
      burn_doy_range    — doy_max - doy_min (scar duration in days)
      burn_start_date   — YYYY-MM-DD of burn_doy_min
      burn_end_date     — YYYY-MM-DD of burn_doy_max (pipeline_2 uses this)
      doy_span_flag     — 0=clean single event, 1=possible merged events
      burn_area_km2     — polygon area in km²
      centroid_lon/lat  — polygon centroid coordinates
      biome/year/month  — provenance
      non_forest_frac   — fraction of burned pixels outside ESA forest class
      mean_slope_deg    — mean SRTM terrain slope (degrees)
      mean_elev_m       — mean SRTM elevation (metres)
      igbp_lc_mode      — dominant IGBP land cover class, year BEFORE fire
    """
    region = ee.Geometry.Rectangle(tile_bbox)

    days_in_month = calendar.monthrange(year, month)[1]
    month_start   = f"{year}-{month:02d}-01"
    month_end     = f"{year}-{month:02d}-{days_in_month}"

    # ── Step A: Load MCD64A1 ────────────────────────────────────
    # mcd64    = (ee.ImageCollection('MODIS/061/MCD64A1')
    #               .filterBounds(region)
    #               .filterDate(month_start, month_end)
    #               .first())
    # burn_doy = mcd64.select('BurnDate')

    # AFTER (correct — composites all MODIS tiles):
    burn_doy = (ee.ImageCollection('MODIS/061/MCD64A1')
              .filterBounds(region)
              .filterDate(month_start, month_end)
              .select('BurnDate')
              .max())           # pixel-wise max across all tiles
    burned   = burn_doy.gt(0).selfMask().clip(region).rename('burned')

    # connectedPixelCount cap=1024 (~256 km²). Fires larger than 1024
    # connected pixels still qualify — the counter saturates at 1024.
    connected   = burned.connectedPixelCount(1024, True)
    significant = burned.updateMask(connected.gte(min_pixels))

    # ── Step B: Vectorise (tileScale=16) ───────────────────────
    clusters = significant.reduceToVectors(
        geometry=region,
        scale=500,
        geometryType='polygon',
        maxPixels=int(1e9),
        bestEffort=True,
        tileScale=16
    )

    # ── Step C: Simplify polygon boundaries ────────────────────
    # Reduces MODIS staircase vertices ~80%. Invisible at 500m resolution.
    # Cuts cost of all subsequent spatial operations significantly.
    clusters = clusters.map(
        lambda f: f.setGeometry(f.geometry().simplify(maxError=100))
    )

    # ── Step C2: Area filter — drop small polygons BEFORE reduceRegions ──
    # min_pixels * 0.25 converts pixel count to km² (each MODIS pixel = 0.25 km²).
    # This is the same threshold already enforced at the pixel level by
    # connectedPixelCount, but applied again on the actual polygon geometry
    # post-vectorization. In fragmented landscapes (Sahel, savanna mosaics),
    # reduceToVectors can produce thousands of qualifying pixel clusters that
    # are geometrically tiny after simplification. Filtering here cuts polygon
    # count before the 6 reduceRegions calls, which scale linearly with count.
    min_area_km2 = min_pixels * 0.25
    clusters = clusters.map(
        lambda f: f.set('area_km2', f.geometry().area(10).divide(1e6))
    )
    clusters = clusters.filter(ee.Filter.gte('area_km2', min_area_km2))

    # ── Step D: Pixel count per polygon ────────────────────────
    clusters = significant.reduceRegions(
        collection=clusters,
        reducer=ee.Reducer.count().setOutputs(['burn_pixel_count']),
        scale=500,
        tileScale=16
    )

    # ── Step E: DOY min/max per polygon ────────────────────────
    # ee.Reducer.minMax() is the documented GEE way to get per-band
    # _min and _max outputs. Applied to band 'BurnDate' it produces
    # properties named exactly 'BurnDate_min' and 'BurnDate_max'.
    #
    # NOTE: Do NOT use min().combine(max(), sharedInputs=True).
    # GEE's combine() resolves name collisions by appending '_1' to
    # the second reducer's output, producing 'BurnDate' and 'BurnDate_1'
    # — not the '_min'/'_max' names the notNull filter below expects.
    # That mismatch drops all features, causing count=0 for every tile.
    clusters = burn_doy.reduceRegions(
        collection=clusters,
        reducer=ee.Reducer.minMax().setOutputs(['BurnDate_min', 'BurnDate_max']),
        scale=500,
        tileScale=16
    )

    # ── Step F: Drop degenerate polygons (BUG FIX 5: single filter)
    # Polygons collapsed by .simplify() or at tile edges may have null
    # BurnDate_min/max. Drop before compute_derived to prevent
    # ee.Number(null).subtract(...) server-side errors.
    clusters = clusters.filter(ee.Filter.notNull(['BurnDate_min', 'BurnDate_max']))

    # ── Step G: WorldCover forest fraction (IMPROVEMENT 2, v4) ─
    # ESA WorldCover v200 class 10 = Tree cover.
    # Mean of binary (0/1) = fraction of polygon pixels that are forest.
    # Sampled at 500m to match MCD64A1 burn pixel definition.
    world_cover   = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
    forest_binary = world_cover.eq(10).rename('is_forest')

    clusters = forest_binary.reduceRegions(
        collection=clusters,
        reducer=ee.Reducer.mean().setOutputs(['forest_frac']),
        scale=500,
        tileScale=16
    )

    # ── Step H: SRTM topographic enrichment (IMPROVEMENT 3, v4) ─
    # SRTM 30m DEM coverage: 56°S to 60°N.
    # Above 60°N (boreal Canada/Russia, Scandinavia), both fields
    # will be null, which compute_derived replaces with 0.0.
    srtm      = ee.Image('USGS/SRTMGL1_003')
    terrain   = ee.Terrain.products(srtm)
    slope_img = terrain.select('slope')     # degrees 0–90
    elev_img  = srtm.select('elevation')    # metres above sea level

    clusters = slope_img.reduceRegions(
        collection=clusters,
        reducer=ee.Reducer.mean().setOutputs(['mean_slope_deg']),
        scale=90,       # 3× SRTM native; fine enough for polygon mean
        tileScale=16
    )
    clusters = elev_img.reduceRegions(
        collection=clusters,
        reducer=ee.Reducer.mean().setOutputs(['mean_elev_m']),
        scale=90,
        tileScale=16
    )

    # ── Step I: Pre-fire forest type from MCD12Q1 IGBP (NEW v5) ─
    # Why MCD12Q1 and not ESA WorldCover for forest type?
    #
    # WorldCover v200 is a single 2021 snapshot. For fires in 2019–2020,
    # pixels deforested between the fire year and 2021 would be classified
    # by their post-deforestation type (e.g. cropland), not their pre-fire
    # forest type. This introduces temporal mismatch errors in exactly the
    # regions most important for forest fire research (deforestation fronts).
    #
    # MCD12Q1 (MODIS Land Cover Type) is produced annually at 500m,
    # covering 2001–present. Using (year - 1) gives the land cover state
    # one full growing season before the fire — the correct pre-fire baseline
    # for every fire year without any region-specific adjustments.
    #
    # IGBP LC_Type1 classes relevant to global fire ecology:
    #   Forest (1–5):
    #     1 = Evergreen Needleleaf Forest  (boreal pine/fir, Pacific coast)
    #     2 = Evergreen Broadleaf Forest   (tropical rainforest: Amazon,
    #                                       Congo, SE Asia)
    #     3 = Deciduous Needleleaf Forest  (boreal larch: Siberia, Canada)
    #     4 = Deciduous Broadleaf Forest   (temperate: oak, beech, maple)
    #     5 = Mixed Forest                 (transition zones)
    #   Woody non-forest (6–9):
    #     6 = Closed Shrublands
    #     7 = Open Shrublands              (Mediterranean maquis, fynbos)
    #     8 = Woody Savannas               (Africa, Cerrado — fire-adapted)
    #     9 = Savannas
    #   Other (10–17):
    #     10=Grasslands, 11=Permanent Wetlands (peat), 12=Croplands,
    #     13=Urban, 14=Cropland-Natural Mosaic, 15=Snow/Ice,
    #     16=Barren, 17=Water
    #
    # ee.Reducer.mode() = most frequent integer class within polygon.
    # Appropriate for categorical data. Null-safe default = 0 (Unknown).
    lc_year = year - 1    # always >= 2018 given FIRE_YEARS starts at 2019
    lc_img  = (ee.ImageCollection('MODIS/061/MCD12Q1')
                 .filterDate(f'{lc_year}-01-01', f'{lc_year}-12-31')
                 .first()
                 .select('LC_Type1'))

    clusters = lc_img.reduceRegions(
        collection=clusters,
        reducer=ee.Reducer.mode().setOutputs(['igbp_lc_mode']),
        scale=500,
        tileScale=16
    )

    # ── Step J: compute_derived — pure property math ────────────
    # No spatial reducers here. All values come from properties attached
    # in Steps D–I. Only arithmetic and date conversions.
    year_ee = ee.Number(year)

    def compute_derived(f):
        # BurnDate_min/max guaranteed non-null by the filter in Step F.
        doy_min   = ee.Number(f.get('BurnDate_min'))
        doy_max   = ee.Number(f.get('BurnDate_max'))
        doy_range = doy_max.subtract(doy_min)

        burn_start_date = (ee.Date.fromYMD(year_ee, 1, 1)
                             .advance(doy_min.subtract(1), 'day')
                             .format('YYYY-MM-dd'))
        burn_end_date   = (ee.Date.fromYMD(year_ee, 1, 1)
                             .advance(doy_max.subtract(1), 'day')
                             .format('YYYY-MM-dd'))

        area_km2 = f.geometry().area(10).divide(1e6)
        centroid = f.geometry().centroid(10)

        # BUG FIX 2: doy_span_flag was dropped in v4. Restored.
        # 0 = clean single fire event (doy_range <= max_doy_gap)
        # 1 = possible merged events (doy_range > max_doy_gap)
        # 2 = confirmed cross-month merge (set by postprocess.py)
        doy_span_flag = doy_range.gt(max_doy_gap).toInt()

        # BUG FIX 1: null_safe_number uses IsEqual(None), not If(value).
        # Default choices:
        #   forest_frac → 0.5  (unknown; neither fully forest nor non-forest)
        #   slope       → 0.0  (flat terrain OR outside SRTM coverage)
        #   elev        → 0.0  (sea level OR outside SRTM coverage)
        #   igbp_lc     → 0    (sentinel for "no MCD12Q1 data")
        forest_frac     = null_safe_number(f.get('forest_frac'), 0.5).max(0).min(1)
        non_forest_frac = ee.Number(1).subtract(forest_frac)
        slope           = null_safe_number(f.get('mean_slope_deg'), 0.0)
        elev            = null_safe_number(f.get('mean_elev_m'),     0.0)
        igbp_mode       = null_safe_number(f.get('igbp_lc_mode'),    0.0).toInt()

        return f.set({
            'biome':             biome_name,
            'year':              year,
            'month':             month,
            'burn_area_km2':     area_km2,
            'burn_doy_min':      doy_min,
            'burn_doy_max':      doy_max,
            'burn_doy_range':    doy_range,
            'burn_start_date':   burn_start_date,
            'burn_end_date':     burn_end_date,
            'doy_span_flag':     doy_span_flag,      # BUG FIX 2: restored
            'centroid_lon':      centroid.coordinates().get(0),
            'centroid_lat':      centroid.coordinates().get(1),
            'non_forest_frac':   non_forest_frac,    # IMPROVEMENT 2
            'mean_slope_deg':    slope,               # IMPROVEMENT 3
            'mean_elev_m':       elev,                # IMPROVEMENT 3
            'igbp_lc_mode':      igbp_mode,           # NEW v5
        })

    clusters = clusters.map(compute_derived)
    return clusters.sort('burn_area_km2', False).limit(max_clusters)


# ─────────────────────────────────────────────────────────────
# METHOD 2: FIRMS FRP ENRICHMENT
# ─────────────────────────────────────────────────────────────

def enrich_with_frp(clusters, year, month):
    """
    Attach FIRMS VIIRS T21 brightness temperature statistics to each
    MCD64A1 polygon.

    BUG FIX 3 — Empty T21 cells in exported CSV:
      firms_col.sum() on a fully masked image (no FIRMS detections in
      this polygon for this month) returns GEE null, which becomes an
      empty cell in the exported CSV — not 0.

      Root cause: FIRMS pixels are masked where confidence < threshold.
      If ALL FIRMS pixels over a polygon are below the confidence gate,
      the mosaic is fully masked. .sum() on a fully masked image = null.

      Fix: .unmask(0) converts null → 0 in both sum and mean composites
      BEFORE reduceRegions. The reducer then aggregates over explicit 0s,
      producing 0 in the output rather than null.

      Interpretation of output values:
        total_t21 = 0  → no confident FIRMS detection this month.
                         MCD64A1 mapped the scar from post-fire reflectance
                         change (cloud-robust). Likely causes: slow
                         smouldering below thermal detection limit, complete
                         cloud cover during active phase, or sub-pixel fires
                         in fragmented landscapes.
        total_t21 > 0  → active fire thermal signal confirmed by VIIRS.
                         Higher value = larger or hotter fire.
                         Reference: Giglio et al. (2016) RSE 178:31–41.

    IMPROVEMENT 1 (v4 — unchanged):
      FIRMS filtered to confidence >= MIN_FIRMS_CONFIDENCE (default 60%)
      before T21 statistics. Suppresses sunglint, agricultural smoke
      false positives, and industrial thermal anomalies globally.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    month_start   = f"{year}-{month:02d}-01"
    month_end     = f"{year}-{month:02d}-{days_in_month}"

    # Apply confidence filter (IMPROVEMENT 1, v4)
    firms_col = (ee.ImageCollection('FIRMS')
                   .filterDate(month_start, month_end)
                   .map(lambda img: img.updateMask(
                       img.select('confidence').gte(config.MIN_FIRMS_CONFIDENCE)
                   ))
                   .select('T21'))

    # BUG FIX 3: .unmask(0) before reduceRegions.
    # Without this, fully masked polygons produce null in the output CSV.
    t21_sum  = firms_col.sum().unmask(0).rename('T21_sum')
    t21_mean = firms_col.mean().unmask(0).rename('T21_mean')

    clusters = t21_sum.reduceRegions(
        collection=clusters,
        reducer=ee.Reducer.sum().setOutputs(['total_t21']),
        scale=375,          # FIRMS VIIRS native resolution
        tileScale=16
    )
    clusters = t21_mean.reduceRegions(
        collection=clusters,
        reducer=ee.Reducer.mean().setOutputs(['mean_t21']),
        scale=375,
        tileScale=16
    )

    return clusters


# ─────────────────────────────────────────────────────────────
# COMBINED PIPELINE FOR ONE TILE
# ─────────────────────────────────────────────────────────────

def process_tile(biome_name, tile_bbox, year, month):
    """Full inventory pipeline for one (biome, tile, year, month)."""
    clusters = discover_fires_mcd64(
        biome_name=biome_name,
        tile_bbox=tile_bbox,
        year=year,
        month=month,
        min_pixels=config.MIN_BURN_PIXELS,
        max_doy_gap=config.MAX_DOY_GAP,
        max_clusters=config.MAX_CLUSTERS_PER_RUN
    )

    clusters = enrich_with_frp(clusters, year, month)

    # Non-forest fraction filter (config-gated, default off: 1.0 = keep all)
    if config.MAX_NON_FOREST_FRAC < 1.0:
        clusters = clusters.filter(
            ee.Filter.lte('non_forest_frac', config.MAX_NON_FOREST_FRAC)
        )

    # T21 intensity filter (config-gated, default off: 0 = keep all)
    if config.MIN_FRP_MW > 0:
        clusters = clusters.filter(
            ee.Filter.gte('total_t21', config.MIN_FRP_MW)
        )

    return clusters


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

    os.makedirs(config.INVENTORY_DIR, exist_ok=True)

   # ── Build job list ──────────────────────────────────────────
    jobs = []
    for biome_name, bbox in config.BIOMES.items():
        tiles  = grid_bbox(bbox, step_deg=10.0)
        months = BIOME_FIRE_MONTHS.get(biome_name, list(range(1, 13)))
        for tile_idx, tile_bbox in enumerate(tiles):
            for year in config.FIRE_YEARS:
                for month in months:
                    jobs.append({
                        'biome':     biome_name,
                        'tile_bbox': tile_bbox,
                        'tile_idx':  tile_idx,
                        'year':      year,
                        'month':     month,
                    })

    log.info(f"Total export tasks : {len(jobs)}")
    log.info(f"  Min burn area    : {config.MIN_BURN_PIXELS} pixels "
             f"= {config.MIN_BURN_PIXELS * 0.25:.0f} km²")
    log.info(f"  Max DOY gap      : {config.MAX_DOY_GAP} days")
    log.info(f"  FIRMS confidence : >= {config.MIN_FIRMS_CONFIDENCE}%")
    log.info(f"  Forest type src  : MCD12Q1 IGBP LC_Type1 (year-1)")

    # ── Pre-filter: drop jobs whose CSVs already exist locally ──
    # Pure filesystem check — zero GEE calls. Do this before fetching
    # the GEE task list so we don't pay any network cost for done jobs.
    pending_jobs = [
        job for job in jobs
        if not os.path.exists(
            os.path.join(
                config.INVENTORY_DIR,
                f"inv_{job['biome']}_t{job['tile_idx']:02d}_{job['year']}_{job['month']:02d}.csv"
            )
        )
    ]
    log.info(f"Already completed  : {len(jobs) - len(pending_jobs)} jobs (CSVs found locally)")
    log.info(f"Remaining to submit: {len(pending_jobs)} jobs")

    # ── Cache GEE active tasks once — avoids per-job getTaskList() ──
    # Calling getTaskList() inside the loop fetches the full task history
    # on every iteration. After thousands of tasks it gets progressively
    # slower and eventually hangs. One upfront call + a set lookup is O(1).
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(ee.data.getTaskList)
            all_tasks = future.result(timeout=60)
        active_descs = {
            t['description'] for t in all_tasks
            if t.get('state') in ('RUNNING', 'READY', 'SUBMITTED')
        }
        log.info(f"GEE active tasks cached: {len(active_descs)}")
    except concurrent.futures.TimeoutError:
        log.warning("getTaskList timed out. Proceeding without dedup check.")
        active_descs = set()
    except Exception as e:
        log.warning(f"Could not fetch GEE task list ({e}). Proceeding without dedup check.")
        active_descs = set()

    submitted = []

    for idx, job in enumerate(pending_jobs):
        biome_name = job['biome']
        tile_bbox  = job['tile_bbox']
        tile_idx   = job['tile_idx']
        year       = job['year']
        month      = job['month']

        desc   = f"inv_{biome_name}_t{tile_idx:02d}_{year}_{month:02d}"
        prefix = desc

        log.info(f"[{idx+1}/{len(pending_jobs)}]  {biome_name}  tile={tile_idx}  "
                 f"{year}-{month:02d}  bbox={[round(x,1) for x in tile_bbox]}")

        # GEE dedup check — O(1) set lookup, no network call
        if desc in active_descs:
            log.info(f"  ⚙️ Task {desc} already running in GEE. Skipping.")
            continue

        try:
            fire_fc = process_tile(biome_name, tile_bbox, year, month)

            # # Count check with hard timeout — prevents silent hangs on
            # # high-density tiles (e.g. peak Sahel dry season).
            # try:
            #     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            #         future = executor.submit(fire_fc.size().getInfo)
            #         count = future.result(timeout=120)

            #     if count == 0:
            #         log.info(f"  No fires found for {desc}. Skipping export.")
            #         continue
            #     log.info(f"  Found {count} fire polygons. Submitting.")

            # except concurrent.futures.TimeoutError:
            #     log.warning(f"  Count timed out (>120s). Submitting {desc} anyway.")
            # except Exception as count_err:
            #     log.warning(f"  Count check failed ({count_err}). Submitting {desc} anyway.")

            wait_for_capacity(config.MAX_CONCURRENT_TASKS,
                              config.TASK_POLL_SECONDS)
            task = export_table_to_drive(
                collection=fire_fc,
                description=desc,
                folder=config.EXPORT_FOLDER,
                file_prefix=prefix,
                wait=False,
                poll_seconds=config.TASK_POLL_SECONDS
            )
            submitted.append((desc, task))
            time.sleep(config.SLEEP_BETWEEN_FIRES)

        except Exception as e:
            log.error(f"  Error on {desc}: {e}")
            continue

    log.info(f"\nAll {len(submitted)} tasks submitted. Polling...")
    for desc, task in submitted:
        while task.status()['state'] in ('RUNNING', 'READY', 'SUBMITTED'):
            log.info(f"  Polling: {desc} — {task.status()['state']}")
            time.sleep(config.TASK_POLL_SECONDS)
        log.info(f"  {desc}: {task.status()['state']}")

    log.info("\n--- Pipeline 1 (v5) complete ---")
    log.info("CSV output fields (* = new or changed in v5):")
    log.info("  burn_start_date   : fire front arrival (DOY_min → date)")
    log.info("  burn_end_date     : last pixel burned  (DOY_max → date)")
    log.info("  burn_area_km2     : polygon area in km²")
    log.info("  doy_span_flag     : 0=clean, 1=possible multi-event  [BUG FIX 2]")
    log.info("  total_t21         : VIIRS T21 sum (0=no detections)  [BUG FIX 3]")
    log.info("  mean_t21          : VIIRS T21 mean (0=no detections) [BUG FIX 3]")
    log.info("  non_forest_frac   : 0=forested, 1=no forest")
    log.info("  mean_slope_deg    : mean SRTM slope (degrees)")
    log.info("  mean_elev_m       : mean SRTM elevation (metres)")
    log.info("* igbp_lc_mode      : dominant pre-fire IGBP class (MCD12Q1, year-1)")
    log.info("  IGBP class lookup : config.IGBP_CLASS_NAMES[id]")
    log.info(f"\nNEXT: download CSVs → utils/postprocess.py → pipeline_2_analysis.py")


if __name__ == '__main__':
    main()