# Mapping energy metabolism systems in the human brain
This repository contains code and data used to run the analyses in "Mapping energy metabolism systems in the human brain", available on [bioRxiv](https://doi.org/10.1101/2025.03.17.643763). 

## scripts
The [scripts](scripts/) folder conatins all the scripts required to run the analyses described in the manuscript.

## data
The [data](data/) folder includes all data needed to run the scripts:
- Structural classes and functional network assignments (Mesulam, Von Economo-Koskinas and Yeo-Krienen) for Schafer-400 and Schaefer-100 parcellations.
- Structural and functional connectivity metrices.
- Cell- and Layer-specific marker genes.
- All other maps used in the analysis including group-average PET and MEG maps and functional connectivity gradients can be retrieved using the neuromaps package with code provided in [scripts](scripts/s14_prepare_brain_maps.py).

## results
The [results](results/) folder includes outputs generated from the scripts, including the energy expression matrices and mean expression maps.
