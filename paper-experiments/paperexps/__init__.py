"""Empirical evaluation for the "Reshaping Diversity Planning" paper.

One module per stage:

- ``run_pool``       one FI plan pool -> one result JSON (experiments 1-5)
- ``generate_slurm`` scan the pools, emit slurm job arrays in the aspbench style
- ``aggregate``      fold the per-pool JSONs into per-experiment CSVs
- ``plots``          paper figures from the CSVs
"""

__version__ = "0.1.0"
