# Robust Sequential Tracking via Bounded Information Geometry

This repository contains the official Python implementations and Monte Carlo benchmarks for the paper **Robust Sequential Tracking via Bounded Information Geometry and Non-Parametric Field Actions** by Carlos C. Rodriguez.

## Overview
Standard state-of-the-art estimators (like unconstrained Kalman filters and MAP estimators) operate on unbounded parameter spaces. When confronted with extreme, structured outliers, they suffer from infinite variance drag and mean divergence. 

This repository implements the **Information Tracker**. By operating directly on the statistical manifold equipped with the plain Information Metric, we utilize strictly invariant Delta (or nu) Information Separations to physically truncate the infinite tails of the spatial distribution. The active parameter space compresses into a strictly finite, normalizable probability droplet (1 - nu * alpha * I_delta >= 0). This geometry analytically severs extreme outliers by assigning exactly zero weight, without relying on infinite-tailed distributional assumptions or heuristic Mahalanobis gating.

## Benchmarks Included
1. `lidar_tracking/` - 6D kinematic tracking of a maneuvering target under severe reflection ghost contamination (+40m spatial offset). 
2. `crypto_order_flow/` - Robust mean tracking of 1-minute ETH-USD close prices, suppressing transient liquidation cascades while adapting to structural regime shifts.
3. `quantum_tomography/` - Native positive semi-definite reconstruction of a single qubit density matrix from noisy Pauli observables.

## Dependencies
The code relies strictly on standard scientific computing libraries to ensure transparency and ease of reproduction.

```bash
pip install numpy scipy matplotlib
