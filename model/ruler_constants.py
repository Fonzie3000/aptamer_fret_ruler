"""
ruler_constants.py
==================

Measured efficiencies and fixed structural constants of the helical ruler.

EXP_E and EXP_SD are the measured FRET efficiencies and their replicate
standard deviations, one per stem length in EXP_BP (base pairs). JUNCTION
(a position, in angstrom) and STEM_AXIS (a direction) are in the coordinate
frame of PDB 6C63 chain A.
"""

import numpy as np

# Measured FRET efficiencies and replicate standard deviations
EXP_BP = [6, 8, 10, 12, 14, 16, 18]
EXP_E = np.array([0.732, 0.814, 0.716, 0.542, 0.452, 0.585, 0.484])
EXP_SD = np.array([0.013, 0.006, 0.007, 0.014, 0.035, 0.023, 0.014])

# Aptamer-stem junction, DSSR pair 6 of 6C63 chain A (C6-G31).
# The G-quadruplex side, treated as the anchored end.
JUNCTION = np.array([23.2653, 43.3558, -10.4172])

# Mean base-pair normal of the Mango II stem, in the 6C63 frame.
# The individual pair normals lie within about 13 degrees of this.
STEM_AXIS = np.array([0.6991, -0.6794, -0.2230])
