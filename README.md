# XRISM/Resolve Additional Screening Tool

## Overview
- This script helps you verify and apply additional screening to your XRISM/Resolve data.
- If you use this script in your publication, please cite Mochizuki et al. 2025, JATIS (DOI:10.1117/1.JATIS.11.4.042002).
- Should you encounter any bugs, please don't hesitate to contact me at mochizuki_at_ac.jaxa.jp.

## How to use
1. Download the XRISM/Resolve data from the HEASARC archive
2. Write the Obs-ID and the path to the data, respectively.
3. Run plot_deriv_rise_tick.py as <code>python plot_deriv_rise_tick.py</code> to confirm the data and its current screening status.
4. Run screening.sh as <code>./screening.sh</code> to apply the additional screening to your data.

## Enviroment
This code requires Python 3 and Heasoft version 6.34 or newer.

## History
2025-07-01: Version 0.0 released.
