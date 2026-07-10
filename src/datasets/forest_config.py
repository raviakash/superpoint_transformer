import numpy as np


########################################################################
#                              Labels                                  #
########################################################################

FOREST_NUM_CLASSES = 16

# Raw class IDs in the LAZ files: 0-13 are contiguous, then 22 and 23.
# We remap to a clean 0-15 train-ID space; anything else becomes void (16).
_MAX_RAW_ID = 24
ID2TRAINID = np.full(_MAX_RAW_ID, FOREST_NUM_CLASSES, dtype=np.int64)
for _i in range(14):
    ID2TRAINID[_i] = _i  # classes 0-13 stay as-is
ID2TRAINID[22] = 14      # lower canopy tree stems
ID2TRAINID[23] = 15      # lower canopy tree branches

CLASS_NAMES = [
    'shrub',                   # 0
    'ground_vegetation',       # 1
    'branches_leaves',         # 2
    'stems',                   # 3
    'dead_downwood',           # 4
    'stump',                   # 5
    'stakes_fences_poles',     # 6
    'people',                  # 7
    'others',                  # 8
    'rocks',                   # 9
    'standing_dead_stem',      # 10
    'ivy',                     # 11
    'understorey_branches',    # 12
    'understorey_stems',       # 13
    'lower_canopy_stems',      # 14  (raw class 22)
    'lower_canopy_branches',   # 15  (raw class 23)
    'void',                    # 16  (ignored / unlabelled)
]

CLASS_COLORS = np.asarray([
    [ 86, 200,  86],   # 0  shrub               — bright green
    [180, 120,  60],   # 1  ground_vegetation    — earthy brown
    [ 34,  85,  34],   # 2  branches_leaves      — dark green
    [139,  69,  19],   # 3  stems                — saddle brown
    [101,  67,  33],   # 4  dead_downwood        — dark brown
    [105, 105, 105],   # 5  stump                — dim gray
    [255, 165,   0],   # 6  stakes_fences_poles  — orange
    [255,  20, 147],   # 7  people               — deep pink
    [128, 128, 128],   # 8  others               — mid gray
    [119, 136, 153],   # 9  rocks                — slate gray
    [211, 211, 211],   # 10 standing_dead_stem   — light gray
    [124, 252,   0],   # 11 ivy                  — lawn green
    [ 60, 150,  60],   # 12 understorey_branches — medium green
    [120,  80,  40],   # 13 understorey_stems    — medium brown
    [100,  60,  30],   # 14 lower_canopy_stems   — darker brown
    [  0, 100,   0],   # 15 lower_canopy_branches— dark green
], dtype=np.uint8)

# Semantic segmentation only — all 16 classes are "stuff" (no instances)
STUFF_CLASSES = list(range(FOREST_NUM_CLASSES))
THING_CLASSES = []

########################################################################
#                       RGB normalisation target                       #
########################################################################

# Per-channel (R, G, B) mean and std computed across all 42 LAZ files in
# data/forest/raw/{train,val,test}/.  Used to normalise out-of-distribution
# RGB from sensors with different radiometry (e.g. MLS vs TLS).
TRAIN_RGB_MEAN = (53302.8, 35858.2, 45588.8)   # uint16 scale (0–65535)
TRAIN_RGB_STD  = (15544.4, 28797.1, 21598.4)


########################################################################
#                            Data splits                               #
########################################################################

# All 14 plots contribute to every stage via the embedded Split field.
# We pre-split the LAZ files (scripts/prepare_forest_data.py) and assign
# unique cloud IDs per stage so SPT processes each subset independently.
_PLOT_IDS = [f'plot_{i:02d}' for i in range(1, 15)]

TILES = {
    'train': _PLOT_IDS,                          # → raw/train/plot_XX.laz
    'val':   [f'{p}_val'  for p in _PLOT_IDS],   # → raw/val/plot_XX.laz
    'test':  [f'{p}_test' for p in _PLOT_IDS],   # → raw/test/plot_XX.laz
}
