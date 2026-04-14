#!/usr/bin/env python3
# =============================================================
# utils/export_helpers.py  (v5.2 — optional TIFF export switch)
#
# CHANGES FROM v5.1
# ─────────────────
# 1. GeoTIFF exports are now fully optional, controlled by three
#    keys in config.py (added in config v7):
#
#      EXPORT_TIFFS        = False     ← master switch
#      EXPORT_FULL_RASTERS = "minimal" ← "minimal" | "full"
#      FULL_RASTER_FIRE_IDS = []       ← always-full fire ID list
#
#    EXPORT_TIFFS = False  →  only CSVs submitted (3 tasks/fire)
#    EXPORT_TIFFS = True,  EXPORT_FULL_RASTERS = "minimal"
#                          →  burn_map only + 3 CSVs (4 tasks/fire)
#    EXPORT_TIFFS = True,  EXPORT_FULL_RASTERS = "full"
#                          →  all 3 raster stacks + 3 CSVs (6 tasks/fire)
#
# 2. burn_severity=None is handled safely:
#    When mask_type='mcd64_fallback', burn_severity is None.
#    The burn_map stack is skipped (no severity band to stack).
#    BurnMask alone is exported as a 1-band minimal raster.
#
# 3. severity_igbp_fc=None is handled safely:
#    The SeverityIGBP CSV export is silently skipped.
#
# TASK COUNT SUMMARY
# ──────────────────
#  EXPORT_TIFFS=False                    → 3 tasks/fire  (CSV only)
#  EXPORT_TIFFS=True, minimal, otsu      → 4 tasks/fire  (burn_map + 3 CSV)
#  EXPORT_TIFFS=True, minimal, mcd64fb  → 4 tasks/fire  (burn_mask 1-band + 3 CSV)
#  EXPORT_TIFFS=True, full, otsu         → 6 tasks/fire  (3 rasters + 3 CSV)
#  EXPORT_TIFFS=True, full, mcd64fb     → 5 tasks/fire  (spectral+recovery + 3 CSV)
#
# v5.1 DESIGN NOTES (unchanged)
# ───────────────────────────────
# Multi-band batching reduces task count vs original 14-task design:
#   spectral_indices (6 bands): NBR/NDVI/EVI pre+post
#   burn_map         (3 bands): dNBR, BurnSeverity, BurnMask
#   recovery         (2 bands): NDVI_rec, LandCover
#
# Decode float bands:  pixel_value / 10000  → float
# Integer bands (BurnSeverity, LandCover, BurnMask): no decode needed.
# Band order within each file: see BAND_ORDER dict below.
#
# READING MULTI-BAND GeoTIFFs IN PYTHON
# ───────────────────────────────────────
# import rasterio
# with rasterio.open('amazon_fire01_spectral_indices_x10000.tif') as src:
#     nbr_pre  = src.read(1) / 10000   # band 1 = NBR_pre
#     nbr_post = src.read(2) / 10000   # band 2 = NBR_post
#     ndvi_pre = src.read(3) / 10000   # band 3 = NDVI_pre
#     ...
# =============================================================

import time
import logging
import ee

from config import (
    EXPORT_TIFFS,
    EXPORT_FULL_RASTERS,
    FULL_RASTER_FIRE_IDS,
)

log = logging.getLogger(__name__)

# Band order within each multi-band GeoTIFF (for documentation + readers)
BAND_ORDER = {
    'spectral_indices': ['NBR_pre_x10000', 'NBR_post_x10000',
                         'NDVI_pre_x10000', 'NDVI_post_x10000',
                         'EVI_pre_x10000',  'EVI_post_x10000'],
    'burn_map':         ['dNBR_x10000', 'BurnSeverity', 'BurnMask'],
    'burn_map_fallback': ['BurnMask'],          # mcd64_fallback: no severity/dNBR
    'recovery':         ['NDVI_rec_x10000', 'LandCover'],
}


# ─────────────────────────────────────────────────────────────
# INTERNAL: TIFF EXPORT DECISION
# ─────────────────────────────────────────────────────────────

def _tiff_flags(fire_id: str) -> tuple[bool, bool]:
    """
    Determine raster export scope for a given fire_id.

    Returns:
        (do_minimal, do_full)
        do_minimal : export burn-map raster (BurnMask ± BurnSeverity/dNBR)
        do_full    : additionally export spectral_indices and recovery rasters

    Logic:
        EXPORT_TIFFS = False            → (False, False)
        fire_id in FULL_RASTER_FIRE_IDS → (True,  True)   always-full override
        EXPORT_FULL_RASTERS = "full"    → (True,  True)
        EXPORT_FULL_RASTERS = "minimal" → (True,  False)   default for full run
    """
    if not EXPORT_TIFFS:
        return False, False
    if fire_id in FULL_RASTER_FIRE_IDS:
        return True, True
    if EXPORT_FULL_RASTERS == "full":
        return True, True
    return True, False   # "minimal"


# ─────────────────────────────────────────────────────────────
# TASK STATUS HELPERS
# ─────────────────────────────────────────────────────────────

def get_running_task_count() -> int:
    statuses = ee.data.getTaskList()
    return len([t for t in statuses if t['state'] in ('RUNNING', 'READY')])


def wait_for_capacity(max_concurrent: int, poll_seconds: int) -> None:
    while True:
        n = get_running_task_count()
        if n < max_concurrent:
            return
        log.info(f"  {n} tasks active — waiting {poll_seconds}s...")
        time.sleep(poll_seconds)


def wait_for_task(task, poll_seconds: int = 30, description: str = "task") -> bool:
    while True:
        status = task.status()
        state  = status['state']
        if state == 'COMPLETED':
            log.info(f"  ✓ {description} completed.")
            return True
        elif state == 'FAILED':
            log.error(f"  ✗ {description} FAILED: {status.get('error_message', '')}")
            return False
        log.info(f"  … {description}: {state} — polling in {poll_seconds}s")
        time.sleep(poll_seconds)


# ─────────────────────────────────────────────────────────────
# LOW-LEVEL EXPORT WRAPPERS
# ─────────────────────────────────────────────────────────────

def export_table_to_drive(collection, description, folder,
                           file_prefix, wait=False, poll_seconds=30):
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description[:100],
        folder=folder,
        fileNamePrefix=file_prefix,
        fileFormat='CSV'
    )
    task.start()
    log.info(f"  → CSV:    {file_prefix}")
    if wait:
        wait_for_task(task, poll_seconds, description=file_prefix)
    return task


def export_image_to_drive(image, description, folder, file_prefix,
                           region, scale=20, crs='EPSG:4326',
                           wait=False, poll_seconds=30):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description[:100],
        folder=folder,
        fileNamePrefix=file_prefix,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=int(1e9),
        formatOptions={'cloudOptimized': True}
    )
    task.start()
    log.info(f"  → TIFF:   {file_prefix}  (scale={scale}m)")
    if wait:
        wait_for_task(task, poll_seconds, description=file_prefix)
    return task


# ─────────────────────────────────────────────────────────────
# COMBINED FIRE EXPORT — BATCHED (v5.2)
# ─────────────────────────────────────────────────────────────

def submit_fire_exports(
    fire_id,
    folder,
    aoi,
    # burn products
    dnbr,
    burn_severity,        # ee.Image or None  (None when mask_type='mcd64_fallback')
    burn_mask,
    # spectral indices
    nbr_pre,  nbr_post,
    ndvi_pre, ndvi_post,
    evi_pre,  evi_post,
    # recovery
    ndvi_rec,
    classified,           # LandCover raster
    # CSV feature collections
    area_fc,
    severity_igbp_fc,     # ee.FeatureCollection or None (None when mcd64_fallback)
    timeseries_fc,
    # rate-limiting
    max_concurrent,
    poll_seconds,
    sleep_between,
):
    """
    Submit all exports for one fire.

    GeoTIFF scope is determined by config.EXPORT_TIFFS,
    config.EXPORT_FULL_RASTERS, and config.FULL_RASTER_FIRE_IDS.

    burn_severity=None  →  mcd64_fallback path:
        minimal TIFF = BurnMask only (1-band), no dNBR/severity stack.
        SeverityIGBP CSV skipped.

    Returns list of submitted ee.batch.Task objects.
    """
    tasks = []
    do_minimal, do_full = _tiff_flags(fire_id)

    # ── RASTER EXPORTS ────────────────────────────────────────────────────

    if do_minimal:
        if burn_severity is not None:
            # Full burn-map stack: dNBR + BurnSeverity + BurnMask (3 bands)
            burn_stack = (
                dnbr.multiply(10000).toInt16().rename('dNBR_x10000')
                    .addBands(burn_severity.toInt16().rename('BurnSeverity'))
                    .addBands(burn_mask.toInt16().rename('BurnMask'))
            )
            wait_for_capacity(max_concurrent, poll_seconds)
            t = export_image_to_drive(
                burn_stack,
                f"{fire_id}_BurnMap",
                folder,
                f"{fire_id}_burn_map_x10000",
                aoi,
                scale=20,
                poll_seconds=poll_seconds,
            )
            tasks.append(t)
            time.sleep(sleep_between)
        else:
            # mcd64_fallback: only BurnMask available (1 band)
            burn_mask_img = burn_mask.toInt16().rename('BurnMask')
            wait_for_capacity(max_concurrent, poll_seconds)
            t = export_image_to_drive(
                burn_mask_img,
                f"{fire_id}_BurnMask",
                folder,
                f"{fire_id}_burn_mask",
                aoi,
                scale=500,          # MCD64A1 native resolution
                poll_seconds=poll_seconds,
            )
            tasks.append(t)
            time.sleep(sleep_between)

    if do_full:
        # Spectral indices stack: NBR/NDVI/EVI pre+post (6 bands)
        spectral_stack = (
            nbr_pre  .multiply(10000).toInt16().rename('NBR_pre_x10000')
                     .addBands(nbr_post .multiply(10000).toInt16().rename('NBR_post_x10000'))
                     .addBands(ndvi_pre .multiply(10000).toInt16().rename('NDVI_pre_x10000'))
                     .addBands(ndvi_post.multiply(10000).toInt16().rename('NDVI_post_x10000'))
                     .addBands(evi_pre  .multiply(10000).toInt16().rename('EVI_pre_x10000'))
                     .addBands(evi_post .multiply(10000).toInt16().rename('EVI_post_x10000'))
        )
        wait_for_capacity(max_concurrent, poll_seconds)
        t = export_image_to_drive(
            spectral_stack,
            f"{fire_id}_SpectralIndices",
            folder,
            f"{fire_id}_spectral_indices_x10000",
            aoi,
            scale=20,
            poll_seconds=poll_seconds,
        )
        tasks.append(t)
        time.sleep(sleep_between)

        # Recovery stack: NDVI_rec + LandCover (2 bands)
        recovery_stack = (
            ndvi_rec  .multiply(10000).toInt16().rename('NDVI_rec_x10000')
                      .addBands(classified.toInt16().rename('LandCover'))
        )
        wait_for_capacity(max_concurrent, poll_seconds)
        t = export_image_to_drive(
            recovery_stack,
            f"{fire_id}_Recovery",
            folder,
            f"{fire_id}_recovery_x10000",
            aoi,
            scale=30,              # 30 m acceptable for recovery products
            poll_seconds=poll_seconds,
        )
        tasks.append(t)
        time.sleep(sleep_between)

    # ── CSV EXPORTS  (always, regardless of TIFF switch) ──────────────────

    csv_exports = [
        # (feature_collection, description, file_prefix)
        (area_fc,
         f"{fire_id}_AreaByClass",
         f"{fire_id}_area_by_class"),
        (timeseries_fc,
         f"{fire_id}_TimeSeries",
         f"{fire_id}_annual_timeseries"),
    ]

    # SeverityIGBP only available from Otsu path (None on mcd64_fallback)
    if severity_igbp_fc is not None:
        csv_exports.insert(
            1,
            (severity_igbp_fc,
             f"{fire_id}_SeverityIGBP",
             f"{fire_id}_severity_igbp"),
        )

    for fc, desc, prefix in csv_exports:
        wait_for_capacity(max_concurrent, poll_seconds)
        t = export_table_to_drive(
            fc, desc, folder, prefix,
            poll_seconds=poll_seconds,
        )
        tasks.append(t)
        time.sleep(sleep_between)

    n_tiff = sum(1 for _ in tasks) - len(csv_exports)
    log.info(
        f"  [{fire_id}]  submitted {len(tasks)} tasks  "
        f"({n_tiff} TIFFs, {len(csv_exports)} CSVs)  "
        f"EXPORT_TIFFS={EXPORT_TIFFS}  scope={EXPORT_FULL_RASTERS}"
    )
    return tasks
