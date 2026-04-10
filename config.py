# =============================================================
# config.py  —  Global Fire Recovery Pipeline
# All parameters live here. Neither pipeline file needs editing
# unless you're changing core logic.
#
# CHANGES FROM ORIGINAL
# ──────────────────────────────────────────────────────────────
# Added five new parameters from NESAC (2014) methodology review:
#   MIN_FIRMS_CONFIDENCE   — suppress false FIRMS detections globally
#   MAX_NON_FOREST_FRAC    — flag / filter agricultural burns
#   MIN_BURN_PATCH_PIXELS  — moved from pipeline_2 hardcode to config
#   RF_NUM_POINTS          — raised from 100→150 for 13-band feature stack
# =============================================================

import os

# ── GEE authentication ─────────────────────────────────────────
# Replace with your GEE-registered project ID.
GEE_PROJECT = "YOUR_GEE_PROJECT_ID"   # e.g. "ee-yourname-firerecovery"

# ── Google Drive export folder ─────────────────────────────────
EXPORT_FOLDER = "GEE_GlobalFireRecovery"

# ── Local paths ────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
INVENTORY_DIR   = os.path.join(BASE_DIR, "outputs", "inventory")
PER_FIRE_DIR    = os.path.join(BASE_DIR, "outputs", "per_fire")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "outputs", "checkpoint.txt")

# ── Fire year range ────────────────────────────────────────────
# FIRMS VIIRS: 2012–present. Sentinel-2: 2015–present.
# 2019–2023 gives full S2+L8 dual-sensor coverage AND a
# 3-year recovery tail within the archive (e.g. 2019 fire → 2022 recovery).
FIRE_YEARS = list(range(2019, 2024))   # [2019, 2020, 2021, 2022, 2023]

# ── MCD64A1 burned area filter ────────────────────────────────
# Primary fire detection uses MODIS MCD64A1 Burned Area Monthly product.
# Resolution: 500m. One pixel = 0.25 km².
#
# MIN_BURN_PIXELS: minimum number of 500m MODIS pixels in a connected
# burned-area cluster for it to qualify as a "significant" fire.
# 500 pixels × 0.25 km²/pixel = 125 km² minimum.
# Adjust down to 200 (50 km²) for data-sparse regions (e.g. Scandinavia).
MIN_BURN_PIXELS = 100

# MAX_DOY_GAP: maximum difference in MODIS burn DOY between adjacent
# pixels for them to be considered part of the same fire event.
# Pixels that burned >MAX_DOY_GAP days apart are treated as separate events
# even if spatially adjacent.
# 15 days is appropriate for fast-moving fire seasons.
# Use 7 for highly active fire corridors (Cerrado, W. Africa).
MAX_DOY_GAP = 15

# ── FIRMS FRP enrichment filter ───────────────────────────────
# After MCD64A1 defines the burn polygon, FIRMS T21 is summed over
# the polygon to score fire intensity. Stored as 'total_t21' in the CSV.
# It is NOT used to filter out fires — every MCD64A1 polygon is kept.
#
# MIN_FRP_MW: optional hard filter on total_t21 score.
# Set to 0 to disable (keep all MCD64A1 fires regardless of FIRMS detection).
MIN_FRP_MW = 0   # set to 0 to keep all MCD64A1 fires (recommended default)

# ── NEW: FIRMS confidence threshold ───────────────────────────
# METHODOLOGICAL IMPROVEMENT (NESAC 2014, §2.2.4):
# FIRMS VIIRS fire pixels include a confidence score (0–100).
# NESAC validated that retaining only detections with confidence ≥ 60%
# suppresses false positives from:
#   - ocean/land sunglint at low solar angles
#   - agricultural smoke hazes misclassified as active fire pixels
#   - industrial thermal sources (smelters, flares)
# This threshold applies ONLY to the FIRMS T21 enrichment step.
# It does NOT affect MCD64A1 burn polygon detection — the inventory
# polygon boundaries are always derived from MCD64A1 surface reflectance.
# Setting to 0 disables the filter and reproduces original pipeline behaviour.
MIN_FIRMS_CONFIDENCE = 60

# ── NEW: Non-forest burn fraction ─────────────────────────────
# METHODOLOGICAL IMPROVEMENT (NESAC 2014, §2.2.4):
# NESAC masked FIRMS points against a forest boundary to exclude
# agricultural burns. For a biome-agnostic global pipeline, a binary
# forest mask is too restrictive (Cerrado savanna mosaics, SE Asian
# agroforestry, etc. would be incorrectly excluded).
#
# Instead, pipeline_1 computes 'non_forest_frac' per polygon:
#   non_forest_frac = 1 - (forest pixels / total burn pixels)
#   0.0 = entirely forested burn
#   1.0 = no forest detected — likely agricultural or grassland fire
#
# MAX_NON_FOREST_FRAC: polygons with non_forest_frac > this value are
# dropped from the inventory.
# 1.0 = keep everything (default — recommended for global studies)
# 0.5 = drop polygons where >50% of area was non-forest before the fire
# 0.3 = strict forest-fire-only mode (use for carbon/biodiversity studies)
#
# ESA WorldCover v200 class 10 (Tree cover) defines "forest" here.
# Temporal note: WorldCover 2021 labels are used for all fires 2019–2023.
# In rapidly deforesting areas this may over-retain or over-exclude
# certain polygons — acceptable for a first-pass global inventory.
MAX_NON_FOREST_FRAC =1.0   # set to 1.0 to keep all fires (default)

# ── NEW: Minimum burn mask patch size ─────────────────────────
# METHODOLOGICAL IMPROVEMENT (NESAC 2014, §2.2.6):
# NESAC applied a majority filter to remove burnt patches smaller than
# ~1 ha to reduce overestimation from spurious dNBR pixels.
# This was previously hardcoded as 200 pixels in pipeline_2_analysis.py.
# Exposed here for global configurability.
#
# At Sentinel-2 20m resolution: 1 pixel = 400 m² = 0.04 ha
#   50 pixels  =  2.0 ha  — catches small but real forest burns
#  200 pixels  =  8.0 ha  — conservative default
#  500 pixels  = 20.0 ha  — large scar studies only
#
# NESAC used ~1 ha; 50 pixels (2 ha) is a reasonable global minimum.
# Keep at 200 (8 ha) to avoid including small agricultural burn patches
# that pass the Otsu threshold.
MIN_BURN_PATCH_PIXELS = 200

# ── NEW: Random Forest training sample size ───────────────────
# Raised from 100 to 150 because build_feature_stack() now outputs
# 13 bands (up from 10 in the original) after adding slope, aspect,
# and elevation. More features require more training samples to maintain
# statistical power and avoid overfitting in sparse biomes.
# Rule of thumb: ≥10 samples per feature per class.
# 5 classes × 13 features × 10 samples = 650 minimum theoretical.
# 150 per class × 5 classes = 750 → safely above the minimum.
RF_NUM_POINTS = 1500

# ── Analysis parameters ────────────────────────────────────────
AOI_BUFFER_M         = 20000   # metres buffer around fire centroid
RECOVERY_YEARS_AFTER = 3       # years after fire end for recovery composite
POST_FIRE_DAYS_START = 10      # smoke-clearance grace period (days)
POST_FIRE_DAYS_END   = 90      # post-fire window end (days after fire)
MAX_CLOUD_PCT        = 20      # max cloud % — pre-fire & recovery
MAX_CLOUD_POST       = 35      # relaxed threshold — post-fire (smoke/ash)
MAX_CLUSTERS_PER_RUN = 50      # hard cap on clusters per biome per year-month

# ── Task pacing ────────────────────────────────────────────────
TASK_POLL_SECONDS    = 60      # GEE task status poll interval
MAX_CONCURRENT_TASKS = 5       # wait when this many tasks are running
SLEEP_BETWEEN_FIRES  = 20      # seconds between sequential submissions

# ── Biome bounding boxes: [lon_min, lat_min, lon_max, lat_max] ─
BIOMES = {
    # Amazon + South America
    "amazon":          [-82.0,-20.0, -35.0,  8.0],
    "cerrado":         [-60.0,-25.0, -35.0, -5.0],
    # South / Southeast Asia
    "south_asia":      [ 68.0,  6.0,  97.0, 37.0],
    "southeast_asia":  [ 95.0, -8.0, 141.0, 28.0],
    # Australia
    "australia_east":  [ 138.0,-44.0,154.0,-10.0],
    "australia_sw":    [ 112.0,-36.0,130.0,-15.0],
    # Western USA / Mexico
    "western_usa":     [-125.0,30.0,-100.0, 50.0],
    "mexico":          [-118.0,14.0, -86.0, 32.0],
    # Sub-Saharan Africa
    "west_africa":     [-18.0,  0.0,  25.0, 20.0],
    "east_africa":     [ 25.0,-12.0,  52.0, 15.0],
    "southern_africa": [ 12.0,-35.0,  52.0,-10.0],
    # Boreal
    "boreal_canada":   [-140.0,48.0, -52.0, 70.0],
    "boreal_russia":   [  60.0,50.0, 180.0, 72.0],
    "scandinavia":     [   5.0,55.0,  32.0, 72.0],
    # Mediterranean + North Africa
    "mediterranean":   [  -5.0,28.0,  42.0, 48.0],
}

# ── ESA WorldCover remapping ────────────────────────────────────
# Raw ESA class IDs → 5-class analysis scheme
WORLDCOVER_FROM = [10, 20, 30, 40, 50, 60, 80, 95]
WORLDCOVER_TO   = [ 1,  2,  2,  4,  4,  3,  5,  1]
CLASS_NAMES = {
    1: "Forest_DenseVeg",
    2: "Scrub_Grassland",
    3: "Bare_Soil",
    4: "Agriculture_Settlement",
    5: "Water",
}

# ── MODIS MCD12Q1 IGBP land cover class names ──────────────────
# Used to decode the 'igbp_lc_mode' field in the inventory CSV.
# igbp_lc_mode = dominant (modal) IGBP class within a burn polygon
# for the year BEFORE the fire, from MODIS/061/MCD12Q1 LC_Type1.
#
# Usage in postprocessing:
#   import config
#   class_name = config.IGBP_CLASS_NAMES.get(int(row['igbp_lc_mode']), 'Unknown')
#
# Forest classes (1–5) are the primary targets for vegetation recovery
# analysis. Classes 6–9 (woody non-forest) are also fire-adapted and
# ecologically important. Classes 12–16 indicate non-natural burns
# (agriculture, urban) that may be filtered depending on study goals.
IGBP_CLASS_NAMES = {
    0:  "Unknown",                           # sentinel for missing MCD12Q1 data
    1:  "Evergreen_Needleleaf_Forest",       # boreal pine/fir; Pacific coast
    2:  "Evergreen_Broadleaf_Forest",        # tropical rainforest: Amazon, Congo, SEA
    3:  "Deciduous_Needleleaf_Forest",       # boreal larch: Siberia, Canada
    4:  "Deciduous_Broadleaf_Forest",        # temperate: oak, beech, maple
    5:  "Mixed_Forest",                      # transition/ecotone zones
    6:  "Closed_Shrublands",
    7:  "Open_Shrublands",                   # Mediterranean maquis, fynbos, steppe
    8:  "Woody_Savannas",                    # Africa, Cerrado — fire-adapted
    9:  "Savannas",
    10: "Grasslands",
    11: "Permanent_Wetlands",                # peat swamps — high carbon risk fires
    12: "Croplands",
    13: "Urban_and_Built-up",
    14: "Cropland_Natural_Vegetation_Mosaic",
    15: "Snow_and_Ice",
    16: "Barren",
    17: "Water_Bodies",
}

# Convenience grouping for downstream filtering:
#   IGBP_FOREST_CLASSES    — true forest; primary recovery study targets
#   IGBP_WOODED_CLASSES    — fire-adapted woody non-forest (savanna, shrub)
#   IGBP_NONVEGETATED      — cropland, urban, barren (likely anthropogenic fires)
IGBP_FOREST_CLASSES     = {1, 2, 3, 4, 5}
IGBP_WOODED_CLASSES     = {6, 7, 8, 9}
IGBP_NONVEGETATED       = {12, 13, 16}
