# =============================================================================
# utils/gee_functions.py  —  v7
# All server-side GEE image processing for the Global Fire Recovery Pipeline.
#
# Bug-fix history (cumulative — all fixes present):
#   v1-v5  argmax() returns ee.List not ee.Array  → ee.Number(ee.List(...).get(0))
#          null check  ee.Algorithms.If(value)    → ee.Algorithms.IsEqual(value, None)
#          MCD64A1     .first()                   → .max()
#          DOY reducer min().combine(max())        → ee.Reducer.minMax()
#          FIRMS null  masked sum                  → .unmask(0) before reduceRegions
#   v6     empty Amazon composites (cloud>20%)    → three-tier server-side composite
#          Otsu upper bound 0.7                   → 0.9  (chaparral/mallee)
#          pre-fire window ±90 days               → ±60 days  (phenology boundary)
#          post-fire window +90 days              → +180 days (Amazon smoke)
#          MCD64A1 fallback ±30 days              → ±3 days   (no scar merging)
#          argmax() fix silently dropped in rewrite → restored
#   v7     zero-band image guard in build_composite() (S2 archive edge, 2019)
#          annual TS composite guard before .gt() call
#          large-fire time-series scale degradation (30 000 km² threshold)
# =============================================================================

import ee
from config import (
    PRE_FIRE_DAYS, POST_FIRE_DAYS_START, POST_FIRE_DAYS_END,
    MAX_CLOUD_PCT, MAX_CLOUD_POST,
    OTSU_LOWER_BOUND, OTSU_UPPER_BOUND,
    MCD64A1_WINDOW_DAYS,
    RECOVERY_YEARS_AFTER,
    AOI_BUFFER_M,
    BAND_NAMES,
    SCALE_M,
    NODATA,
    LARGE_FIRE_KM2, TS_SCALE_NORMAL, TS_SCALE_LARGE,
)


# =============================================================================
# 1.  SENTINEL-2 / LANDSAT COMPOSITE  (fully server-side, no .getInfo())
# =============================================================================

def _s2_cloud_mask(image):
    """Pixel-level SCL-based cloud mask for Sentinel-2 SR."""
    scl = image.select("SCL")
    # Keep: vegetation(4), bare soil(5), water(6), snow/ice(11)
    # Reject: saturated(1), dark(2), shadow(3), cloud_medium(8), cloud_high(9), cirrus(10)
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(11))
    return image.updateMask(good)


def _l8_cloud_mask(image):
    """Pixel-level QA_PIXEL cloud mask for Landsat-8 C2 T1 L2."""
    qa = image.select("QA_PIXEL")
    cloud      = qa.bitwiseAnd(1 << 3).neq(0)
    cloud_shad = qa.bitwiseAnd(1 << 4).neq(0)
    return image.updateMask(cloud.Or(cloud_shad).Not())


def _harmonise_l8_to_s2(image):
    """
    HLS harmonisation: Landsat-8 SR → Sentinel-2 SR spectral space.
    Coefficients: Claverie et al. (2018), Table 2.
    Renames L8 bands to S2 band names for downstream consistency.
    """
    # L8 C2 scale factor: multiply by 0.0000275 + (-0.2) per USGS
    b2  = image.select("SR_B2").multiply(0.0000275).add(-0.2)
    b3  = image.select("SR_B3").multiply(0.0000275).add(-0.2)
    b4  = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    b8  = image.select("SR_B5").multiply(0.0000275).add(-0.2)   # NIR ~ B8
    b8a = image.select("SR_B5").multiply(0.0000275).add(-0.2)   # NIR ~ B8A (approx)
    b11 = image.select("SR_B6").multiply(0.0000275).add(-0.2)
    b12 = image.select("SR_B7").multiply(0.0000275).add(-0.2)
    return (ee.Image.cat([b2, b3, b4, b8, b8a, b11, b12])
              .rename(["B2", "B3", "B4", "B8", "B8A", "B11", "B12"])
              .copyProperties(image, ["system:time_start"]))


def _s2_scale(image):
    """
    Scale S2 SR DN integers → surface reflectance floats.
    S2 SR scale factor: divide by 10000 (ESA convention).
    This makes S2 dtype homogeneous with HLS-harmonised L8 floats
    so that .merge() + .median() across both collections succeeds.
    Without this cast GEE raises:
      "Expected a homogeneous image collection ... Integer<0,65535>
       vs Float<-0.2, 1.6022125>"
    """
    scaled = image.select(["B2","B3","B4","B8","B8A","B11","B12"]).toFloat().divide(10000)
    return scaled.copyProperties(image, ["system:time_start"])


def _s2_collection(aoi, start_date, end_date, max_cloud):
    """Returns a cloud-filtered, pixel-masked, float-scaled S2 SR collection."""
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(aoi)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
              .map(_s2_cloud_mask)
              .map(_s2_scale))


def _l8_collection(aoi, start_date, end_date):
    """Returns a cloud-masked, HLS-harmonised Landsat-8 collection."""
    return (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
              .filterBounds(aoi)
              .filterDate(start_date, end_date)
              .map(_l8_cloud_mask)
              .map(_harmonise_l8_to_s2))


def _safe_zero_band_image():
    """
    Returns a fully-masked dummy image with correct band structure.
    Used as a guaranteed fallback so build_composite() NEVER returns
    a zero-band image — composite_is_empty() will catch it instead.

    Must be Float to match _s2_scale() output — mixing Integer constant
    with Float collection causes the same homogeneity error.
    """
    dummy_bands = [ee.Image.constant(0.0).toFloat().rename(b)
                   for b in ["B2", "B3", "B4", "B8", "B8A", "B11", "B12"]]
    dummy = (ee.Image.cat(dummy_bands)
               .updateMask(ee.Image(0)))   # fully masked
    return dummy


def build_composite(aoi, start_date, end_date, max_cloud=MAX_CLOUD_PCT):
    """
    Three-tier server-side composite (fully lazy — safe inside ee.List.map()).

    Tier 1: strict cloud filter  (max_cloud)
    Tier 2: relaxed cloud filter (min(max_cloud × 1.5, 75))
    Tier 3: no scene-level filter, pixel-level SCL/QA mask only

    Falls back through tiers with .unmask().  If all tiers are empty the
    function returns a fully-masked image with the correct band structure
    so that composite_is_empty() can detect it without crashing downstream
    .gt() or band-math operations.

    NO .getInfo() calls — safe for server-side map().
    """
    relaxed_cloud = min(max_cloud * 1.5, 75)

    s2_t1 = _s2_collection(aoi, start_date, end_date, max_cloud)
    s2_t2 = _s2_collection(aoi, start_date, end_date, relaxed_cloud)
    s2_t3 = _s2_collection(aoi, start_date, end_date, 100)   # all scenes
    l8_t3 = _l8_collection(aoi, start_date, end_date)

    comp_t1 = s2_t1.median()
    comp_t2 = s2_t2.median().unmask(comp_t1)
    # Tier 3: merge S2 (pixel-masked) + L8 (HLS-harmonised)
    comp_t3 = (s2_t3.merge(l8_t3)
                    .sort("system:time_start")
                    .median()
                    .unmask(comp_t2))

    # Hard zero-band guard — ensures the return value always has the
    # correct band structure even if all collections were empty.
    # composite_is_empty() will detect the fully-masked result downstream.
    safe = _safe_zero_band_image()
    composite = ee.Image(
        ee.Algorithms.If(
            comp_t3.bandNames().size().eq(0),
            safe,
            comp_t3
        )
    )
    return composite


def composite_is_empty(composite, aoi):
    """
    Python-side emptiness check.  Calls .getInfo() ONCE — call this
    only in analyse_fire(), never inside a server-side map().

    Returns True if the composite has no valid pixels over the AOI.

    Diagnostic logging included — remove once confirmed working.

    Root-cause history:
      scale=500  → misses pixels in small AOIs (false empty)
      scale=100  → still false-empty if aoi geometry is malformed
      bestEffort=True → prevents maxPixels errors on large AOIs
    """
    import logging
    _log = logging.getLogger(__name__)
    try:
        result_dict = (composite.select(0)
                                .reduceRegion(
                                    reducer=ee.Reducer.count(),
                                    geometry=aoi,
                                    scale=100,
                                    maxPixels=1e9,
                                    bestEffort=True,
                                ).getInfo())
        _log.debug(f"composite_is_empty raw result: {result_dict}")
        if not result_dict:
            _log.warning("composite_is_empty: reduceRegion returned empty dict")
            return True
        count = list(result_dict.values())[0]
        _log.debug(f"composite_is_empty pixel count: {count}")
        return (count is None) or (count == 0)
    except Exception as exc:
        _log.warning(f"composite_is_empty exception: {exc}")
        return True


# =============================================================================
# 2.  SPECTRAL INDICES
# =============================================================================

def compute_nbr(image):
    """Normalised Burn Ratio: (NIR - SWIR2) / (NIR + SWIR2)."""
    nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    return nbr


def compute_ndvi(image):
    """Normalised Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
    return image.normalizedDifference(["B8", "B4"]).rename("NDVI")


def compute_evi(image):
    """
    Enhanced Vegetation Index — less saturated than NDVI at LAI > 3.
    Critical for tropical canopy (Amazon, Congo, SEA).
    EVI = 2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)
    """
    nir  = image.select("B8")
    red  = image.select("B4")
    blue = image.select("B2")
    evi = (nir.subtract(red)
              .multiply(2.5)
              .divide(
                  nir.add(red.multiply(6))
                     .subtract(blue.multiply(7.5))
                     .add(1)
              )
              .rename("EVI"))
    return evi


def compute_dnbr(pre_nbr, post_nbr):
    """dNBR = pre-fire NBR − post-fire NBR  (positive = burned)."""
    return pre_nbr.subtract(post_nbr).rename("dNBR")


# =============================================================================
# 2b.  COMBINED BASELINE MEANS  (single reduceRegion — replaces 4 separate calls)
# =============================================================================

def compute_baseline_means(pre_ndvi, post_ndvi, pre_evi, post_evi,
                            burn_mask, aoi, scale):
    """
    Compute pre/post NDVI and EVI means in a SINGLE reduceRegion call.

    Previously this was 4 separate _masked_mean() calls, each costing
    ~90 seconds of GEE round-trip time = ~6 minutes per fire.
    One stacked reduceRegion call costs ~90-120 seconds total.
    Saves ~270 seconds per fire.

    Returns dict with keys:
        pre_ndvi_mean, post_ndvi_mean, pre_evi_mean, post_evi_mean
    All values are Python floats. NODATA=-9999.0 on null.
    """
    stacked = (pre_ndvi .rename("pre_ndvi")
                        .addBands(post_ndvi.rename("post_ndvi"))
                        .addBands(pre_evi  .rename("pre_evi"))
                        .addBands(post_evi .rename("post_evi"))
                        .updateMask(burn_mask))

    result = stacked.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()

    def _safe(key):
        val = result.get(key)
        # IsEqual(None) pattern: treat missing and None as NODATA
        return float(val) if val is not None else -9999.0

    return {
        "pre_ndvi_mean":  _safe("pre_ndvi"),
        "post_ndvi_mean": _safe("post_ndvi"),
        "pre_evi_mean":   _safe("pre_evi"),
        "post_evi_mean":  _safe("post_evi"),
    }


# =============================================================================
# 3.  BURN SEVERITY
# =============================================================================

def _severity_from_dnbr(dnbr):
    """
    USGS dNBR severity thresholds → integer class raster.
    1 = Enhanced Regrowth Low   (dNBR < -0.25)
    2 = Enhanced Regrowth High  (-0.25 ≤ dNBR < -0.10)
    3 = Unburned                (-0.10 ≤ dNBR < +0.10)
    4 = Low Severity            (+0.10 ≤ dNBR < +0.27)
    5 = Moderate-Low            (+0.27 ≤ dNBR < +0.44)
    6 = Moderate-High           (+0.44 ≤ dNBR < +0.66)
    7 = High Severity           (dNBR ≥ +0.66)
    """
    sev = (dnbr.lt(-0.25).multiply(1)
               .add(dnbr.gte(-0.25).And(dnbr.lt(-0.10)).multiply(2))
               .add(dnbr.gte(-0.10).And(dnbr.lt( 0.10)).multiply(3))
               .add(dnbr.gte( 0.10).And(dnbr.lt( 0.27)).multiply(4))
               .add(dnbr.gte( 0.27).And(dnbr.lt( 0.44)).multiply(5))
               .add(dnbr.gte( 0.44).And(dnbr.lt( 0.66)).multiply(6))
               .add(dnbr.gte( 0.66).multiply(7))
               .rename("BurnSeverity"))
    return sev


def compute_otsu_threshold(dnbr, aoi):
    """
    Compute Otsu threshold on dNBR histogram.
    Returns Python float or None if computation fails / histogram empty.
    This is the only .getInfo() call in the burn-mask path.
    """
    try:
        histogram = dnbr.reduceRegion(
            reducer=ee.Reducer.autoHistogram(maxBuckets=256, cumulative=False),
            geometry=aoi,
            scale=SCALE_M,
            maxPixels=1e9
        ).get("dNBR")

        # histogram may be null if all pixels are masked
        is_null = ee.Algorithms.IsEqual(histogram, None).getInfo()
        if is_null:
            return None

        hist_array = ee.Array(histogram)
        counts = hist_array.slice(1, 1, 2).project([0])
        values = hist_array.slice(1, 0, 1).project([0])

        total     = counts.reduce(ee.Reducer.sum(), [0]).get([0])
        sum_vals  = values.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
        mean_all  = ee.Number(sum_vals).divide(ee.Number(total))

        n = counts.length().get([0])
        indices = ee.List.sequence(0, ee.Number(n).subtract(1))

        def _between_class_variance(i):
            i    = ee.Number(i)
            wB   = counts.slice(0, 0, i).reduce(ee.Reducer.sum(), [0]).get([0])
            wF   = counts.slice(0, i, n).reduce(ee.Reducer.sum(), [0]).get([0])
            sumB = (values.slice(0, 0, i)
                         .multiply(counts.slice(0, 0, i))
                         .reduce(ee.Reducer.sum(), [0]).get([0]))
            meanB = ee.Number(sumB).divide(ee.Number(wB).max(1e-9))
            meanF = (ee.Number(sum_vals).subtract(sumB)
                                        .divide(ee.Number(wF).max(1e-9)))
            bcv   = (ee.Number(wB).multiply(wF)
                                  .multiply(meanB.subtract(meanF).pow(2)))
            return bcv

        bcv_list = ee.Array(indices.map(_between_class_variance))
        # v6/v7 fix: argmax() returns ee.List, not ee.Array
        best_idx  = ee.Number(ee.List(bcv_list.argmax()).get(0))
        threshold = ee.Number(values.get([best_idx]))

        t = threshold.getInfo()
        return float(t) if t is not None else None

    except Exception:
        return None


def build_burn_mask(dnbr, aoi, fire_geometry, burn_date, fire_year):
    """
    Build burn mask + severity raster.

    Primary:  Otsu threshold on dNBR.
              Sanity bounds: [OTSU_LOWER_BOUND, OTSU_UPPER_BOUND]
              i.e. [-0.5, 0.9]  — upper bound 0.9 covers chaparral/mallee.
    Fallback: MCD64A1 burn date raster (±MCD64A1_WINDOW_DAYS days).
              Returns mask_type='mcd64_fallback'; burn_severity=None.

    Returns dict:
        burn_mask    : ee.Image (1=burned, 0=unburned)
        burn_severity: ee.Image or None
        mask_type    : 'otsu' | 'mcd64_fallback'
        threshold    : float or None
    """
    threshold = compute_otsu_threshold(dnbr, aoi)

    use_otsu = (
        threshold is not None
        and OTSU_LOWER_BOUND <= threshold <= OTSU_UPPER_BOUND
    )

    if use_otsu:
        burn_mask = dnbr.gt(threshold).rename("BurnMask")

        # Sanity check: at least MIN_BURN_PATCH_PIXELS pixels
        pixel_count = (burn_mask
                       .reduceRegion(
                           reducer=ee.Reducer.sum(),
                           geometry=aoi,
                           scale=SCALE_M,
                           maxPixels=1e9
                       ).get("BurnMask"))
        is_null  = ee.Algorithms.IsEqual(pixel_count, None).getInfo()
        px_count = 0 if is_null else ee.Number(pixel_count).getInfo()

        from config import MIN_BURN_PATCH_PIXELS
        if px_count >= MIN_BURN_PATCH_PIXELS:
            burn_severity = _severity_from_dnbr(dnbr).updateMask(burn_mask)
            return {
                "burn_mask":     burn_mask,
                "burn_severity": burn_severity,
                "mask_type":     "otsu",
                "threshold":     threshold,
            }

    # ---- MCD64A1 fallback ------------------------------------------------
    bd   = ee.Date(burn_date)
    mcd  = (ee.ImageCollection("MODIS/061/MCD64A1")
              .filterBounds(fire_geometry)
              .filterDate(
                  bd.advance(-MCD64A1_WINDOW_DAYS, "day"),
                  bd.advance( MCD64A1_WINDOW_DAYS, "day")
              )
              .select("BurnDate")
              .max())    # .max() not .first() — catches cross-tile pixels

    burn_doy_start = bd.advance(-MCD64A1_WINDOW_DAYS, "day").getRelative("day", "year")
    burn_doy_end   = bd.advance( MCD64A1_WINDOW_DAYS, "day").getRelative("day", "year")

    mcd_mask = (mcd.gte(ee.Number(burn_doy_start))
                   .And(mcd.lte(ee.Number(burn_doy_end)))
                   .rename("BurnMask"))

    return {
        "burn_mask":     mcd_mask,
        "burn_severity": None,         # not available from MCD64A1 fallback
        "mask_type":     "mcd64_fallback",
        "threshold":     None,
    }


# =============================================================================
# 4.  LAND COVER  (MODIS MCD12Q1)
# =============================================================================

def get_igbp_land_cover(aoi, fire_year):
    """
    Annual IGBP land cover at year-1 (pre-fire).
    Uses MCD12Q1 — NOT WorldCover (WorldCover is single 2021 epoch, wrong
    for 2019-2020 fires).
    """
    lc_year = str(int(fire_year) - 1)
    lc = (ee.ImageCollection("MODIS/061/MCD12Q1")
            .filterDate(f"{lc_year}-01-01", f"{lc_year}-12-31")
            .first()
            .select("LC_Type1")
            .rename("LandCover"))
    return lc.clip(aoi)


# =============================================================================
# 5.  ANNUAL RECOVERY TIME-SERIES
# =============================================================================

def _annual_composite_for_ts(aoi, center_date_str, max_cloud):
    """
    Build a single-year composite for the annual time-series.
    Fully server-side (no .getInfo()) — safe inside ee.List.map()
    via the three-tier build_composite() path.
    """
    center = ee.Date(center_date_str)
    start  = center.advance(-PRE_FIRE_DAYS, "day")
    end    = center.advance( PRE_FIRE_DAYS, "day")
    return build_composite(aoi, start, end, max_cloud)


def build_annual_timeseries(aoi, fire_date_str, burn_mask, max_cloud,
                             pre_ndvi_mean, pre_evi_mean,
                             post_ndvi_mean, post_evi_mean,
                             fire_year, igbp_lc_mode,
                             burn_area_km2):
    """
    Build annual NDVI + EVI time-series for RECOVERY_YEARS_AFTER years.

    Includes:
      - pre/post baseline means per row (for recovery completeness index)
      - mask_type forwarded from burn mask
      - large-fire scale degradation (> LARGE_FIRE_KM2 → TS_SCALE_LARGE)

    Returns list of ee.Feature ready for export as a FeatureCollection.

    NO .getInfo() calls inside the loop — all composites use the
    three-tier server-side path.  The zero-band guard in build_composite()
    ensures .gt() and .normalizedDifference() never receive a bandless image.
    """
    ts_scale = TS_SCALE_LARGE if burn_area_km2 > LARGE_FIRE_KM2 else TS_SCALE_NORMAL
    features = []

    fire_dt = ee.Date(fire_date_str)

    for yr_offset in range(1, RECOVERY_YEARS_AFTER + 1):
        center = fire_dt.advance(yr_offset, "year")
        center_str = center.format("YYYY-MM-dd").getInfo()  # one .getInfo() per year

        comp = _annual_composite_for_ts(aoi, center_str, max_cloud)

        # Zero-band / empty composite guard
        # build_composite() guarantees correct band structure, but the image
        # may be fully masked.  Use a server-side conditional so .gt() /
        # .normalizedDifference() always receive a valid image.
        fallback = _safe_zero_band_image()
        comp = ee.Image(
            ee.Algorithms.If(
                comp.bandNames().size().eq(0),
                fallback,
                comp
            )
        )

        ndvi = compute_ndvi(comp).updateMask(burn_mask)
        evi  = compute_evi(comp).updateMask(burn_mask)

        ndvi_mean_val = (ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi, scale=ts_scale, maxPixels=1e9
        ).get("NDVI"))

        evi_mean_val = (evi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi, scale=ts_scale, maxPixels=1e9
        ).get("EVI"))

        # Null-safe extraction  — ee.Algorithms.If(value) treats 0.0 as falsy
        # so we use IsEqual(None) for all null checks.
        def _safe_get(val, default=-9999.0):
            return ee.Number(
                ee.Algorithms.If(
                    ee.Algorithms.IsEqual(val, None),
                    default,
                    val
                )
            )

        feat = ee.Feature(None, {
            "fire_year":       fire_year,
            "igbp_lc_mode":    igbp_lc_mode,
            "year_offset":     yr_offset,
            "center_date":     center_str,
            "ndvi_mean":       _safe_get(ndvi_mean_val),
            "evi_mean":        _safe_get(evi_mean_val),
            "pre_ndvi_mean":   pre_ndvi_mean,
            "pre_evi_mean":    pre_evi_mean,
            "post_ndvi_mean":  post_ndvi_mean,
            "post_evi_mean":   post_evi_mean,
            # Recovery completeness index:
            # (yr_NDVI − post_NDVI) / (pre_NDVI − post_NDVI)
            # computed downstream in Python from the four columns above
        })
        features.append(feat)

    return ee.FeatureCollection(features)


# =============================================================================
# 6.  AREA-BY-CLASS TABLE
# =============================================================================

def compute_area_by_class(burn_mask, land_cover, aoi, fire_year, igbp_lc_mode):
    """
    Area (km²) per IGBP class within burn scar.
    Returns ee.FeatureCollection (one row per IGBP class).

    Band order fix: GEE requires data band BEFORE grouping band.
    Correct order: [area_km2 (band 0), LandCover (band 1)]
    groupField=1 → group by LandCover (last band).
    Wrong order (old): [LandCover (0), area_km2 (1)], groupField=0
    → "Group input must come after weighted inputs" error.
    """
    burned_lc = land_cover.rename("LandCover").updateMask(burn_mask)
    # Data band first, grouping band last — required by GEE grouped reducer
    area_img = (ee.Image.pixelArea().divide(1e6).rename("area_km2")
                  .addBands(burned_lc))

    stats = area_img.reduceRegion(
        reducer=ee.Reducer.sum().group(
            groupField=1,          # band 1 = LandCover (grouping band, must be last)
            groupName="igbp_class"
        ),
        geometry=aoi,
        scale=SCALE_M,
        maxPixels=1e9,
        bestEffort=True,
    ).get("groups")

    def _to_feature(group):
        d = ee.Dictionary(group)
        return ee.Feature(None, {
            "igbp_class":   d.get("igbp_class"),
            "area_km2":     d.get("sum"),
            "fire_year":    fire_year,
            "igbp_lc_mode": igbp_lc_mode,
        })

    return ee.FeatureCollection(ee.List(stats).map(_to_feature))


# =============================================================================
# 7.  SEVERITY × IGBP CROSS-TABLE
# =============================================================================

def compute_severity_igbp(burn_severity, land_cover, aoi, fire_year, igbp_lc_mode):
    """
    Cross-tabulation of burn severity class vs IGBP land cover class.
    Skipped when burn_severity is None (MCD64A1 fallback fires).
    Returns ee.FeatureCollection or None.

    Implementation: combined-class encoding avoids nested .group().group().
    Nested grouped reducers in GEE produce ambiguous key ordering — the
    second .group() becomes the OUTER key in the result dict, causing:
      "Dictionary does not contain key: severity"
    when extraction code expects severity as the outer key.

    Fix: encode severity + igbp into one integer:
        combined = severity * 100 + igbp
    Single .group() on combined band, decode in Python.
    Severity 1-7, IGBP 1-17 → combined 101-1717, no collisions.
    """
    if burn_severity is None:
        return None

    # Encode both classes into one integer band (no nesting needed)
    combined = (burn_severity.toInt().multiply(100)
                             .add(land_cover.toInt())
                             .rename("combined"))

    # Data band first (band 0), grouping band last (band 1) — GEE requirement
    area_img = (ee.Image.pixelArea().divide(1e6).rename("area_km2")
                  .addBands(combined))

    stats = area_img.reduceRegion(
        reducer=ee.Reducer.sum().group(
            groupField=1,
            groupName="combined_class"
        ),
        geometry=aoi,
        scale=SCALE_M,
        maxPixels=1e9,
        bestEffort=True,
    ).get("groups")

    def _to_feature(group):
        d              = ee.Dictionary(group)
        combined_val   = ee.Number(d.get("combined_class")).toInt()
        severity_class = combined_val.divide(100).toInt()
        igbp_class     = combined_val.mod(100)
        return ee.Feature(None, {
            "severity_class": severity_class,
            "igbp_class":     igbp_class,
            "area_km2":       d.get("sum"),
            "fire_year":      fire_year,
            "igbp_lc_mode":   igbp_lc_mode,
        })

    return ee.FeatureCollection(ee.List(stats).map(_to_feature))

def scale_to_int16(image):
    """
    Scale float rasters (range ≈ -1 to 1) to Int16 × 10000.
    Reduces GeoTIFF file size ~4× vs Float32.
    """
    return image.multiply(10000).round().toInt16()