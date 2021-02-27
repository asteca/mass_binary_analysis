
TODO:

1. Assign (m1, m2) masses, put both stars into the single systems' pool, and  *then* estimate the single system's mass.
2. Use ASteCA's binary probability to identify binaries


Process:

1. Cluster data is read from file in `input/` folder
2. The IMF's CDF is generated outside of the `for` blocks
3. For each combination of `(z,a,e,d)` parameters
4. Load the isochrone for that `(z,a)`
5. Move the isochrone using the `(e,d)` values
6. Interpolate extra points into the isochrone
7. 




