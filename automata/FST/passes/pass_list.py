import compute_algebras

# These passes should work every time inputs are passed to them.
# This module contains a static list of such passes that can be run
# over various IRs

# GraphIR.
ComputeAlgebras = compute_algebras.ComputeAlgebraPass()
