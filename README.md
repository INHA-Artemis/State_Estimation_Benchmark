# IMU/GPS State Estimation Benchmark Library

This project provides a **Python-based benchmark library for IMU/GPS state estimation algorithms**.  
The goal is to implement and compare multiple filtering methods under a **unified and modular interface**.

## Features

- Implementation of multiple state estimation algorithms:
  - Extended Kalman Filter (**EKF**)
  - Unscented Kalman Filter (**UKF**)
  - Particle Filter (**PF**)
  - Invariant Extended Kalman Filter (**InEKF**)

- **Unified interface** for all filters
- Easy configuration:
  - IMU on/off
  - GPS on/off
  - Filter parameter tuning
- **Real-time estimation visualization**
- **Performance analysis and comparison across datasets**

## State Models

The library supports two state representations:

- **3D model**
  
  \[
  (x, y, \psi)
  \]

- **6D model**

  Full 6-DoF state estimation.

## Goal

The objective of this project is to build a **modular Python benchmark library** that:

1. Implements EKF, UKF, PF, and InEKF under a common interface  
2. Supports flexible sensor configurations (IMU/GPS on/off)  
3. Provides clear visualization and performance evaluation tools  
4. Enables easy comparison across multiple datasets  

## References

The implementation is inspired by existing libraries and tutorials:

- navlie  
- FilterPy  
- Stone Soup tutorials  
- robot_localization  
- DRIFT