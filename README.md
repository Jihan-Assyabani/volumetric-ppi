# Volumetric PPI

Volumetric PPI is a python-based algorithm to reconstruct volumetric wind field from plan position indicator (PPI) scanning long-range doppler wind lidar. Typical PPI scan reconstruction yields planar wind field. This reconstruction approach take multiple scans from 10~minutes window then derives wind field from other perspectives:
- horizontal slices at multiple heights
- vertical slices (similar to range-height indicator scans)
- cross-sectional slices

<img width=800px src="/Figures/volumetric-ppi.png">

## Introduction
