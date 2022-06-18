# FLOWMath

[![](https://img.shields.io/badge/docs-dev-blue.svg)](http://flow.byu.edu/FLOWMath.jl/dev/)
![](https://github.com/byuflowlab/FLOWMath.jl/workflows/Run%20tests/badge.svg)

*A collection of mathematical functions and convenience methods*.  These are simple methods we use regularly in our lab that aren't readily available in the Julia ecosystem.  

Authors: Andrew Ning and Taylor McDonnell

Quadrature
- trapezoidal integration

Root Finding
- Brent's method

Interpolation
- Akima spline
- linear interpolation
- 2D/3D/4D interpolation from recursive 1D iterpolation

Smoothing
- absolute value
- Kreisselmeier-Steinhauser constraint aggregation function
- sigmoid blending
- cubic/quintic polynomial blending

[Complex step safe](https://doi.org/10.1145/838250.838251) versions of
- `abs`: `abs_cs_safe`
- `abs2`: `abs2_cs_safe`
- `norm`: `norm_cs_safe`
- `dot`: `dot_cs_safe`
- `atan` (two argument form): `atan_cs_safe`

### Install

```julia
] add FLOWMath
```

### Run Unit Tests

```julia
] test FLOWMath
```
