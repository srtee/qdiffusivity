"""Tests for the density QKDE module."""

import numpy as np
import pytest

from qdiffusivity.density import (
    TransverseDensityQKDE,
    epanechnikov_kernel,
    kde_1d,
    select_bandwidth,
    silverman_bw,
)


def test_epanechnikov_kernel():
    """Normalised, compact support, non-negative, peak at zero."""
    x = np.linspace(-3.0, 3.0, 200_001)
    assert np.trapezoid(
        epanechnikov_kernel(x, 1.0), x
    ) == pytest.approx(1.0, abs=1e-4)
    assert epanechnikov_kernel(np.array([1.6]), 1.5)[0] == 0.0
    assert epanechnikov_kernel(
        np.array([0.0]), 1.5
    )[0] == pytest.approx(0.75 / 1.5)
    assert np.all(epanechnikov_kernel(x, 2.0) >= 0.0)


def test_bandwidth_selection():
    """Silverman, Sheather-Jones, select_bandwidth, and edge cases."""
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 1.0, size=10_000)
    assert silverman_bw(z) == pytest.approx(1.06 * 10_000 ** (-0.2), rel=0.3)
    assert silverman_bw(np.array([1.0])) == 0.1
    from qdiffusivity.density import sheather_jones_bw

    z_bimodal = np.concatenate(
        [rng.normal(-3.0, 0.5, 2000), rng.normal(3.0, 0.5, 2000)]
    )
    assert 0 < sheather_jones_bw(z_bimodal, -10, 10) <= silverman_bw(z_bimodal)
    assert select_bandwidth(z, -5, 5, method=0.42) == pytest.approx(0.42)
    with pytest.raises(ValueError):
        select_bandwidth(z, -5, 5, method="bogus")


def test_kde_1d():
    """Integration to 1, bandwidth validation, ESS bounds, empty data."""
    rng = np.random.default_rng(1)
    z = rng.normal(50.0, 5.0, size=20_000)
    z_eval = np.linspace(20.0, 80.0, 2001)
    rho, n_eff = kde_1d(z, z_eval, 0.5, 0.0, 100.0)
    assert np.trapezoid(rho, z_eval) == pytest.approx(1.0, abs=0.02)
    # ESS is positive in the well-sampled interior and bounded by N.
    interior = (z_eval > 40) & (z_eval < 60)
    assert np.all(n_eff[interior] > 0)
    assert np.all(n_eff <= z.size)
    with pytest.raises(ValueError):
        kde_1d(z, z_eval, 0.0, 0.0, 100.0)
    rho_e, n_eff_e = kde_1d(np.array([]), np.linspace(0, 10, 11), 0.5, 0, 10)
    assert np.all(rho_e == 0.0) and np.all(n_eff_e == 0.0)


@pytest.mark.parametrize("grouping", ["atoms", "residues"])
def test_density_qkde(density_universe, grouping):
    """Run, attributes, normalisation, auto bandwidth, auto boundaries."""
    n_atoms, n_res, n_frames = 200, 10, 5
    u = density_universe(
        n_atoms=n_atoms, n_res=n_res,
        n_frames=n_frames, Lz=100.0,
    )
    ag = u.select_atoms("all")
    kde = TransverseDensityQKDE(
        ag,
        dim=2,
        z_bot=0.0,
        z_top=100.0,
        n_points=80,
        bandwidth="auto",
        grouping=grouping,
    )
    kde.run()
    assert kde.rho.shape == (80,)
    assert np.all(kde.rho >= 0.0)
    assert np.trapezoid(kde.rho, kde.z_eval) == pytest.approx(1.0, abs=0.05)
    assert kde.n_frames_used == n_frames
    if grouping == "residues":
        expected_n = n_res * n_frames
    else:
        expected_n = n_atoms * n_frames
    assert kde.n_total == expected_n
    assert np.isfinite(kde.bandwidth) and kde.bandwidth > 0


def test_density_qkde_auto_boundaries(density_universe):
    u = density_universe(n_atoms=100, n_frames=2, Lz=40.0)
    kde = TransverseDensityQKDE(
        u.select_atoms("all"), dim=2, n_points=40, bandwidth=0.3
    )
    kde.run()
    assert kde.z_eval[0] > 0.0
    assert kde.z_eval[-1] < 40.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 5},
        {"grouping": "molecules"},
        {"bandwidth": "bogus"},
    ],
)
def test_density_qkde_validation(density_universe, kwargs):
    u = density_universe(n_atoms=10, n_res=2, n_frames=1, Lz=10.0)
    with pytest.raises(ValueError):
        TransverseDensityQKDE(u.select_atoms("all"), **kwargs)


def test_density_qkde_inverted_boundaries(density_universe):
    u = density_universe(n_atoms=10, n_res=2, n_frames=1, Lz=10.0)
    kde = TransverseDensityQKDE(
        u.select_atoms("all"), dim=2, z_bot=10.0, z_top=0.0, bandwidth=0.5
    )
    with pytest.raises(ValueError):
        kde.run()
