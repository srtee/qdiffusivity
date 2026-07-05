"""
qdiffusivity
Quantile-based estimation of local diffusivities in molecular dynamics
simulations of nanoconfined liquids
"""

# Add imports here
from importlib.metadata import version

from .binned import (
    TransverseDensityBinned,
    TransverseDiffusivityBinned,
    cic_assign,
    resolve_bins,
)
from .density import (
    TransverseDensityKDE,
    epanechnikov_kernel,
    kde_1d,
    select_bandwidth,
    sheather_jones_bw,
    silverman_bw,
)
from .diffusivity import (
    TransverseDiffusivityKDE,
    build_cdf,
    gaussian_kernel,
    kde_estimate,
    select_diff_bandwidth,
)

__version__ = version("qdiffusivity")

__all__ = [
    "TransverseDensityBinned",
    "TransverseDensityKDE",
    "TransverseDiffusivityBinned",
    "TransverseDiffusivityKDE",
    "build_cdf",
    "cic_assign",
    "epanechnikov_kernel",
    "gaussian_kernel",
    "kde_1d",
    "kde_estimate",
    "resolve_bins",
    "select_bandwidth",
    "select_diff_bandwidth",
    "sheather_jones_bw",
    "silverman_bw",
    "__version__",
]
