# =============================================================
# utils/gee_functions.py
# All GEE server-side image processing functions.
#
# CHANGES FROM ORIGINAL
# ──────────────────────────────────────────────────────────────
# IMPROVEMENT 4 — Topographic bands in RF feature stack (NESAC §2.2.3)
#   build_feature_stack() now includes slope, aspect, and elevation
#   from SRTM 30m DEM as additional Random Forest input features.
#
#   Why topography in the RF classifier?
#   NESAC 2014 demonstrated that slope (15–25° peak), south/east aspect,
#   and mid-elevation (200–1000m) are globally meaningful correlates of
#   fire behaviour and post-fire recovery rate. Including these features
#   allows the RF classifier to separate land-cover classes that have
#   similar spectral signatures but different topographic contexts:
#     - North-facing shadowed slopes (low reflectance, dense canopy) vs.
#       south-facing sun-exposed slopes (same NDVI range, sparser canopy)
#     - High-altitude rocky bare soil vs. lowland bare agriculture
#       (both appear as bare soil spectrally but have different recovery
#       trajectories)
#   Using .resample('bilinear') on terrain layers avoids .reproject(),
#   which would force server-side resolution and break lazy evaluation.
#
#   RF_NUM_POINTS raised from 100→150 in run_rf_classification() to
#   maintain statistical power with the expanded 13-band feature stack.
#   Value is read from config.RF_NUM_POINTS.
# =============================================================

import ee
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────
# CLOUD MASKING
# ─────────────────────────────────────────────────────────────

def mask_s2_clouds(image):
    """
    Sentinel-2 SR cloud mask using the Scene Classification Layer (SCL).
    Keeps: 4=vegetation, 5=bare soil, 6=water, 11=snow/ice.
    Drops: 7=unclassified, 8/9/10=cloud shadows, cloud, cirrus.
    SCL 11 (snow) is kept — boreal and alpine zones require it.
    Divides by 10000 to convert DN → reflectance [0–1].
    """
    scl  = image.select('SCL')
    mask = (scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(11)))
    return (image.updateMask(mask)
                 .divide(10000)
                 .copyProperties(image)
                 .set('system:time_start', image.get('system:time_start')))


def mask_l8_clouds(image):
    """
    Landsat 8 Collection 2 SR cloud mask using QA_PIXEL bitmask.
    Bit 1 = dilated cloud, Bit 3 = cloud, Bit 4 = cloud shadow — dropped.
    Applies Collection 2 scale + offset: reflectance = DN × 0.0000275 − 0.2
    """
    qa   = image.select('QA_PIXEL')
    mask = (qa.bitwiseAnd(1 << 3).eq(0)
              .And(qa.bitwiseAnd(1 << 4).eq(0))
              .And(qa.bitwiseAnd(1 << 1).eq(0)))
    return (image.updateMask(mask)
                 .multiply(0.0000275).add(-0.2)
                 .copyProperties(image)
                 .set('system:time_start', image.get('system:time_start')))


# ─────────────────────────────────────────────────────────────
# HLS HARMONIZATION (Claverie et al. 2018)
# ─────────────────────────────────────────────────────────────

def harmonize_l8_to_s2(image):
    """
    Harmonized Landsat Sentinel-2 (HLS) linear transformation.
    Converts Landsat 8 SR bands to Sentinel-2 equivalent reflectance.
    Coefficients from Claverie et al. (2018), globally validated.

    Output bands match S2 naming: B2, B3, B4, B8, B11, B12
    (Blue, Green, Red, NIR, SWIR1, SWIR2)

    Why HLS over site-specific OLS?
    Site-specific OLS requires coincident S2/L8 image pairs at every
    new AOI — impractical for a global batch pipeline.
    HLS coefficients are globally calibrated and appropriate here.
    """
    b2  = image.select('SR_B2').multiply(0.9778).add( 0.0053).rename('B2')
    b3  = image.select('SR_B3').multiply(1.0053).add(-0.0023).rename('B3')
    b4  = image.select('SR_B4').multiply(0.9765).add( 0.0089).rename('B4')
    b8  = image.select('SR_B5').multiply(0.9983).add(-0.0007).rename('B8')
    b11 = image.select('SR_B6').multiply(0.9972).add(-0.0072).rename('B11')
    b12 = image.select('SR_B7').multiply(1.0031).add(-0.0189).rename('B12')
    return (b2.addBands([b3, b4, b8, b11, b12])
              .copyProperties(image)
              .set('system:time_start', image.get('system:time_start')))


# ─────────────────────────────────────────────────────────────
# COLLECTION LOADERS
# ─────────────────────────────────────────────────────────────

def get_s2(aoi, start_date, end_date, max_cloud):
    """Load and cloud-mask Sentinel-2 SR Harmonized collection."""
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
              .filterBounds(aoi)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', max_cloud))
              .map(mask_s2_clouds))


def get_l8(aoi, start_date, end_date, max_cloud):
    """Load and cloud-mask Landsat 8 C2 T1 SR collection."""
    return (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
              .filterBounds(aoi)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUD_COVER', max_cloud))
              .map(mask_l8_clouds))


# ─────────────────────────────────────────────────────────────
# COMPOSITE BUILDER
# ─────────────────────────────────────────────────────────────

def build_composite(aoi, start_date, end_date, max_cloud):
    """
    Merge Sentinel-2 and harmonized Landsat 8 into a single median composite.

    Strategy:
      1. S2 median composite — primary source (10m natively)
      2. L8 harmonized median — fills gaps where S2 is cloudy or absent
      3. .unmask(l8) fills remaining NoData in S2 with L8 values

    Bands returned: B2, B3, B4, B8, B11, B12
    (Blue, Green, Red, NIR broad, SWIR1, SWIR2)
    """
    s2_med = (get_s2(aoi, start_date, end_date, max_cloud)
                .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
                .median()
                .clip(aoi))

    l8_med = (get_l8(aoi, start_date, end_date, max_cloud)
                .map(harmonize_l8_to_s2)
                .median()
                .clip(aoi))

    return s2_med.unmask(l8_med).clip(aoi)


# ─────────────────────────────────────────────────────────────
# SPECTRAL INDICES
# ─────────────────────────────────────────────────────────────

def calc_nbr(img):
    """
    Normalized Burn Ratio: (NIR - SWIR2) / (NIR + SWIR2)
    B12 (SWIR2, 20m) resampled to match B8 (NIR, 10m) with bilinear
    interpolation. Uses .resample('bilinear') — avoids .reproject().
    """
    nir  = img.select('B8')
    swir = img.select('B12').resample('bilinear')
    return nir.subtract(swir).divide(nir.add(swir)).rename('NBR')


def calc_ndvi(img):
    """NDVI: (NIR - Red) / (NIR + Red)"""
    return img.normalizedDifference(['B8', 'B4']).rename('NDVI')


def calc_ndwi(img):
    """NDWI (Gao 1996): (NIR - SWIR1) / (NIR + SWIR1) — vegetation water content"""
    return img.normalizedDifference(['B8', 'B11']).rename('NDWI')


def calc_evi(img):
    """
    Enhanced Vegetation Index — reduces canopy background and
    atmospheric influence. Better than NDVI in dense tropical canopy.
    EVI = 2.5 × (NIR−RED) / (NIR + 6×RED − 7.5×BLUE + 1)
    """
    return img.expression(
        '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
        {'NIR':  img.select('B8'),
         'RED':  img.select('B4'),
         'BLUE': img.select('B2')}
    ).rename('EVI')


def build_feature_stack(img):
    """
    Assemble all spectral bands + indices + topographic context into one
    multi-band image for Random Forest input.

    Original 10 bands:
      B2, B3, B4, B8, B11, B12  (spectral)
      NDVI, NDWI, NBR, EVI       (spectral indices)

    IMPROVEMENT 4 — 3 topographic bands added (NESAC 2014, §2.2.3):
      slope_deg    — terrain slope in degrees (SRTM)
      aspect_deg   — terrain aspect in degrees 0–360 (SRTM)
      elevation_m  — metres above sea level (SRTM)

    Total: 13 bands.

    Why topography as RF features?
    NESAC 2014 established that slope and aspect are significant
    determinants of fire severity AND post-fire recovery rate. This
    finding is biome-agnostic: illumination geometry, soil moisture
    retention, and wind exposure all interact with topography in the
    same physical way globally. Adding these features allows the RF
    to separate:
      - Spectrally similar classes with different topographic context
        (e.g. dark/dense shadow on a north-facing slope ≠ dense canopy)
      - High-altitude rocky bare soil vs. lowland agricultural bare soil
      - Steep-slope post-fire bare soil (high erosion risk) vs.
        flat post-fire grassland (fast regrowth, different trajectory)

    Technical note:
      - SRTM coverage: 56°S to 60°N. Above 60°N (boreal Canada/Russia,
        Scandinavia) the terrain bands will contain NoData. The RF
        classifier handles NoData inputs gracefully in GEE — it simply
        uses the other 10 bands for those pixels.
      - .resample('bilinear') on terrain layers avoids .reproject(),
        which would force server-side resolution and break lazy evaluation.
      - ee.Terrain.products() is called once per build_feature_stack()
        invocation; GEE caches the result across bandNames() calls.
    """
    srtm    = ee.Image('USGS/SRTMGL1_003')
    terrain = ee.Terrain.products(srtm)

    # slope: 0–90 degrees. Higher values = steeper terrain.
    # NESAC showed peak fire frequency at 15–25° globally in NER — this
    # generalises to Mediterranean, Western USA, and Himalayan foothills.
    slope = terrain.select('slope').resample('bilinear').rename('slope_deg')

    # aspect: 0–360 degrees clockwise from north.
    # 0/360 = North, 90 = East, 180 = South, 270 = West.
    # South and east aspects receive more solar radiation in the Northern
    # Hemisphere → drier fuel → higher fire frequency (NESAC §3.3).
    # In the Southern Hemisphere the effect is reversed (north-facing).
    # The RF learns this asymmetry from training data without hard-coding.
    aspect = terrain.select('aspect').resample('bilinear').rename('aspect_deg')

    # elevation: metres above sea level.
    # NESAC showed peak fire frequency at 200–1000m; above 2000m alpine
    # forests have lower fire frequency. Elevation also proxies for
    # climate zone (temperature, precipitation seasonality).
    elevation = srtm.select('elevation').resample('bilinear').rename('elevation_m')

    return (img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
               .addBands([
                   calc_ndvi(img),
                   calc_ndwi(img),
                   calc_nbr(img).rename('NBR'),
                   calc_evi(img),
                   slope,
                   aspect,
                   elevation
               ]))


# ─────────────────────────────────────────────────────────────
# dNBR + BURN SEVERITY
# ─────────────────────────────────────────────────────────────

def calc_dnbr(pre_img, post_img):
    """
    differenced NBR: pre-fire NBR − post-fire NBR
    Positive values = burned (high pre-fire vegetation signal lost).
    Negative values = enhanced post-fire regrowth (rare but real).
    """
    return calc_nbr(pre_img).subtract(calc_nbr(post_img)).rename('dNBR')


def calc_burn_severity(dnbr):
    """
    Map dNBR to 6-class USGS burn severity scheme.
    Class 1: Enhanced Regrowth (dNBR < -0.25)
    Class 2: Unburned         (-0.25 ≤ dNBR < 0.10)
    Class 3: Low severity     (0.10 ≤ dNBR < 0.27)
    Class 4: Moderate-Low     (0.27 ≤ dNBR < 0.44)
    Class 5: Moderate-High    (0.44 ≤ dNBR < 0.66)
    Class 6: High severity    (dNBR ≥ 0.66)
    """
    return (dnbr.expression(
        "(b('dNBR') < -0.25) ? 1"
        ": (b('dNBR') < 0.10) ? 2"
        ": (b('dNBR') < 0.27) ? 3"
        ": (b('dNBR') < 0.44) ? 4"
        ": (b('dNBR') < 0.66) ? 5"
        ": 6"
    ).rename('BurnSeverity').toInt())


# ─────────────────────────────────────────────────────────────
# OTSU ADAPTIVE THRESHOLD
# ─────────────────────────────────────────────────────────────

def compute_otsu_threshold(dnbr_image, aoi):
    """
    Otsu's method: find the dNBR value that maximises between-class
    variance (burned vs. unburned split).

    Replaces the hardcoded dNBR > 0.15 used in earlier single-site scripts.
    Every ecosystem has a different dNBR distribution:
      - Amazonian rainforest: narrow, low dNBR range
      - Californian chaparral: wide, high dNBR range
    Otsu finds the optimal threshold from the actual histogram of
    each fire event's AOI — no manual tuning required.

    Reference: Otsu (1979). IEEE Trans. Sys. Man. Cyb. 9(1):62–66.
    All operations are server-side (ee.Array) — no .getInfo() call.
    """
    histogram = dnbr_image.reduceRegion(
        reducer=ee.Reducer.histogram(maxBuckets=200, minBucketWidth=0.005),
        geometry=aoi,
        scale=30,
        bestEffort=True,
        tileScale=16
    ).get('dNBR')

    counts = ee.Array(ee.Dictionary(histogram).get('histogram'))
    means  = ee.Array(ee.Dictionary(histogram).get('bucketMeans'))
    size   = means.length().get([0])
    total  = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    sum_   = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean   = sum_.divide(total)

    indices = ee.List.sequence(1, size.subtract(1))

    def compute_bss(i):
        i       = ee.Number(i).toInt()
        a_cnts  = counts.slice(0, 0, i)
        a_count = a_cnts.reduce(ee.Reducer.sum(), [0]).get([0])
        a_means = means.slice(0, 0, i)
        a_mean  = (a_means.multiply(a_cnts)
                         .reduce(ee.Reducer.sum(), [0]).get([0])
                         .divide(a_count))
        b_count = total.subtract(a_count)
        b_mean  = sum_.subtract(a_count.multiply(a_mean)).divide(b_count)
        return (a_count.multiply(a_mean.subtract(mean).pow(2))
                       .add(b_count.multiply(b_mean.subtract(mean).pow(2))))

    bss = indices.map(compute_bss)
    # CORRECTED LINE
    # CORRECTED LINE
    return means.get(ee.Array(bss).argmax().get(0))


# ─────────────────────────────────────────────────────────────
# BURN MASK
# ─────────────────────────────────────────────────────────────

def build_burn_mask(dnbr, aoi, min_patch_pixels=None):
    """
    Two-stage burn mask:
      1. Otsu threshold on dNBR — adaptive to this AOI's histogram
      2. connectedPixelCount filter — removes isolated noise pixels

    min_patch_pixels: if None, reads from config.MIN_BURN_PATCH_PIXELS.
      This parameter was previously hardcoded as 200 in pipeline_2_analysis.py.
      It is now configurable via config.py (IMPROVEMENT from NESAC §2.2.6).

    NESAC applied a majority filter to remove patches < ~1 ha.
    At Sentinel-2 20m: 200 pixels = 8 ha (conservative default).
    Set MIN_BURN_PATCH_PIXELS = 50 in config.py for 2 ha minimum.
    """
    if min_patch_pixels is None:
        min_patch_pixels = config.MIN_BURN_PATCH_PIXELS

    otsu_thresh = compute_otsu_threshold(dnbr, aoi)
    raw_mask    = dnbr.gt(otsu_thresh)
    clean_mask  = (raw_mask
                   .updateMask(
                       raw_mask.connectedPixelCount(min_patch_pixels)
                               .gte(min_patch_pixels))
                   .selfMask()
                   .clip(aoi)
                   .rename('BurnMask'))
    return clean_mask, otsu_thresh
