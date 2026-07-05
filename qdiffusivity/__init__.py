"""
qdiffusivity
Quantile-based estimation of local diffusivities in molecular dynamics
simulations of nanoconfined liquids
"""

# Add imports here
from importlib.metadata import version

from .density import (
    TransverseDensityKDE,
    epanechnikov_kernel,
    kde_1d,
    select_bandwidth,
    sheather_jones_bw,
    silverman_bw,
)

__version__ = version("qdiffusivity")

__all__ = [
    "TransverseDensityKDE",
    "epanechnikov_kernel",
    "kde_1d",
    "select_bandwidth",
    "sheather_jones_bw",
    "silverman_bw",
    "__version__",
]
