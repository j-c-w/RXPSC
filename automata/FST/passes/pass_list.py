import compute_algebras
import prefix_split
import splitter

# These passes should work every time inputs are passed to them.
# This module contains a static list of such passes that can be run
# over various IRs

# GraphIR.
ComputeAlgebras = compute_algebras.ComputeAlgebraPass()

PrefixSplit = prefix_split.PrefixSplitPass()
Splitter = splitter.SplitterPass()
