# Volumetric PPI

## Citation
As-sya'bani, J. A. (2026). Volumetric Wind Field Reconstruction of Nacelle-based PPI Scanning Long-Range Lidar. Zenodo. https://doi.org/10.5281/zenodo.18436130

## Overview
Volumetric PPI is a python-based algorithm to reconstruct volumetric wind field from plan position indicator (PPI) scanning long-range doppler wind lidar. Typical PPI scan reconstruction yields planar wind field. This reconstruction approach take multiple scans from a 10 minutes window then derives wind field from multiple perspectives:
- horizontal slices (similar to PPI scans)
- vertical slices (similar to RHI scans)
- cross-sectional slices

<img width=800px src="/Figures/volumetric-ppi.png">

