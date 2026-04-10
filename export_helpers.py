#!/usr/bin/env python3
# =============================================================
# utils/export_helpers.py  (v5)
# GEE task submission, status polling, and rate-limiting.
#
# CHANGES FROM v4
# ──────────────────────────────────────────────────────────────
# submit_fire_exports() updated signature:
#   + ndvi_pre, ndvi_post   — NDVI before and after fire (Int16 ×10000)
#   + evi_pre, evi_post     — EVI before and after fire  (Int16 ×10000)
#   + severity_igbp_fc      — FeatureCollection: area by severity×IGBP
#
# All other helpers (wait_for_capacity, export_table_to_drive,
# export_image_to_drive, wait_for_task) are unchanged from v4.
# =============================================================

import time
import logging
import ee

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# TASK STATUS HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────

def get_running_task_count():
    """Return how many GEE tasks are currently RUNNING or READY."""
    statuses = ee.data.getTaskList()
    active   = [t for t in statuses if t['state'] in ('RUNNING', 'READY')]
    return len(active)


def wait_for_capacity(max_concurrent, poll_seconds):
    """Block until fewer than max_concurrent tasks are active."""
    while True:
        n = get_running_task_count()
        if n < max_concurrent:
            return
        log.info(f"  {n} tasks active — waiting {poll_seconds}s...")
        time.sleep(poll_seconds)


def wait_for_task(task, poll_seconds=30, description="task"):
    """Block until a single task completes. Returns True if successful."""
    while True:
        status = task.status()
        state  = status['state']
        if state == 'COMPLETED':
            log.info(f"  ✓ {description} completed.")
            return True
        elif state == 'FAILED':
            log.error(f"  ✗ {description} FAILED: {status.get('error_message','')}")
            return False
        log.info(f"  … {description}: {state} — polling in {poll_seconds}s")
        time.sleep(poll_seconds)


# ─────────────────────────────────────────────────────────────
# EXPORT WRAPPERS  (unchanged)
# ─────────────────────────────────────────────────────────────

def export_table_to_drive(collection, description, folder,
                           file_prefix, wait=False, poll_seconds=30):
    """Export a FeatureCollection as CSV to Google Drive."""
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description[:100],
        folder=folder,
        fileNamePrefix=file_prefix,
        fileFormat='CSV'
    )
    task.start()
    log.info(f"  → Submitted CSV export: {file_prefix}")
    if wait:
        wait_for_task(task, poll_seconds, description=file_prefix)
    return task


def export_image_to_drive(image, description, folder, file_prefix,
                           region, scale=20, crs='EPSG:4326',
                           wait=False, poll_seconds=30):
    """
    Export an image as Cloud-Optimised GeoTIFF to Google Drive.
    Float rasters should be pre-multiplied by 10000 and cast to Int16
    before calling this function (halves file size vs Float32).
    """
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
    log.info(f"  → Submitted GeoTIFF export: {file_prefix}")
    if wait:
        wait_for_task(task, poll_seconds, description=file_prefix)
    return task


# ─────────────────────────────────────────────────────────────
# COMBINED FIRE EXPORT  (v5)
# ─────────────────────────────────────────────────────────────

def submit_fire_exports(fire_id, folder, aoi,
                         dnbr, burn_severity, burn_mask,
                         nbr_pre, nbr_post,
                         ndvi_pre, ndvi_post,       # NEW v5
                         evi_pre, evi_post,          # NEW v5
                         ndvi_rec,
                         classified,
                         area_fc,
                         severity_igbp_fc,           # NEW v5
                         timeseries_fc,
                         max_concurrent, poll_seconds, sleep_between):
    """
    Submit all GeoTIFF + CSV exports for one fire event.

    v5 raster exports (all Int16 ×10000, divide by 10000 in Python):
      dNBR                — differenced NBR (fire severity signal)
      BurnSeverity        — USGS 6-class severity map
      BurnMask            — binary burn scar mask
      NBR_pre / NBR_post  — NBR before / after fire
      NDVI_pre / NDVI_post — NDVI before / after fire  (NEW v5)
      EVI_pre / EVI_post  — EVI before / after fire    (NEW v5)
      NDVI_rec            — NDVI at recovery window
      LandCover           — RF land cover classification

    v5 CSV exports:
      area_by_class       — area per RF land cover class
      severity_igbp       — area per severity × IGBP class (NEW v5)
      annual_timeseries   — NDVI/EVI/NBR/CHIRPS per year + baselines

    Decode all Int16 rasters: value_float = pixel_value / 10000
    Exception: BurnSeverity and LandCover are class integers (no decode).
    """
    tasks  = []
    region = aoi

    # ── Int16-encode float rasters ─────────────────────────────
    dnbr_int      = dnbr.multiply(10000).toInt16().rename('dNBR_x10000')
    nbr_pre_int   = nbr_pre.multiply(10000).toInt16().rename('NBR_pre_x10000')
    nbr_post_int  = nbr_post.multiply(10000).toInt16().rename('NBR_post_x10000')
    ndvi_pre_int  = ndvi_pre.multiply(10000).toInt16().rename('NDVI_pre_x10000')
    ndvi_post_int = ndvi_post.multiply(10000).toInt16().rename('NDVI_post_x10000')
    evi_pre_int   = evi_pre.multiply(10000).toInt16().rename('EVI_pre_x10000')
    evi_post_int  = evi_post.multiply(10000).toInt16().rename('EVI_post_x10000')
    ndvi_rec_int  = ndvi_rec.multiply(10000).toInt16().rename('NDVI_rec_x10000')

    raster_exports = [
        # (image, description, file_prefix)
        (dnbr_int,              f"{fire_id}_dNBR",         f"{fire_id}_dnbr_x10000"),
        (burn_severity.toByte(),f"{fire_id}_BurnSeverity", f"{fire_id}_burn_severity"),
        (burn_mask.toByte(),    f"{fire_id}_BurnMask",     f"{fire_id}_burn_mask"),
        # NBR pair
        (nbr_pre_int,           f"{fire_id}_NBR_pre",      f"{fire_id}_nbr_pre_x10000"),
        (nbr_post_int,          f"{fire_id}_NBR_post",     f"{fire_id}_nbr_post_x10000"),
        # NDVI pair (NEW v5)
        (ndvi_pre_int,          f"{fire_id}_NDVI_pre",     f"{fire_id}_ndvi_pre_x10000"),
        (ndvi_post_int,         f"{fire_id}_NDVI_post",    f"{fire_id}_ndvi_post_x10000"),
        # EVI pair (NEW v5)
        (evi_pre_int,           f"{fire_id}_EVI_pre",      f"{fire_id}_evi_pre_x10000"),
        (evi_post_int,          f"{fire_id}_EVI_post",     f"{fire_id}_evi_post_x10000"),
        # Recovery window
        (ndvi_rec_int,          f"{fire_id}_NDVI_rec",     f"{fire_id}_ndvi_recovery_x10000"),
        (classified.toByte(),   f"{fire_id}_LandCover",    f"{fire_id}_recovery_landcover"),
    ]

    for img, desc, prefix in raster_exports:
        wait_for_capacity(max_concurrent, poll_seconds)
        t = export_image_to_drive(img, desc, folder, prefix, region,
                                   scale=20, poll_seconds=poll_seconds)
        tasks.append(t)
        time.sleep(sleep_between)

    # ── CSV exports ────────────────────────────────────────────
    csv_exports = [
        (area_fc,          f"{fire_id}_AreaByClass",    f"{fire_id}_area_by_class"),
        (severity_igbp_fc, f"{fire_id}_SeverityIGBP",  f"{fire_id}_severity_igbp"),   # NEW v5
        (timeseries_fc,    f"{fire_id}_TimeSeries",     f"{fire_id}_annual_timeseries"),
    ]

    for fc, desc, prefix in csv_exports:
        wait_for_capacity(max_concurrent, poll_seconds)
        t = export_table_to_drive(fc, desc, folder, prefix,
                                   poll_seconds=poll_seconds)
        tasks.append(t)
        time.sleep(sleep_between)

    return tasks