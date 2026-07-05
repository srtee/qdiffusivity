"""Tests for the diffusivity QKDE module."""

import numpy as np
import pytest

import qdiffusivity
from qdiffusivity.diffusivity import (
    LocalDiffusivityQKDE,
    build_cdf,
    gaussian_kernel,
    kde_estimate,
    select_diff_bandwidth,
)


def test_gaussian_kernel():
    """Normalised, symmetric, peak at zero."""
    x = np.linspace(-10.0, 10.0, 200_001)
    assert np.trapezoid(
        gaussian_kernel(x, 1.0), x
    ) == pytest.approx(1.0, abs=1e-4)
    k = gaussian_kernel(np.array([-3, -1, 0, 1, 3.0]), 1.5)
    assert k[0] == pytest.approx(k[-1])
    assert k[2] > k[1] > k[0]
    assert k[2] == pytest.approx(1.0 / (1.5 * np.sqrt(2 * np.pi)))


def test_bandwidth_selection():
    """Silverman, SJ for both kernels, select_diff_bandwidth, invalid."""
    rng = np.random.default_rng(10)
    u = rng.uniform(0.0, 1.0, size=5_000)
    from qdiffusivity.diffusivity import sheather_jones_bw, silverman_bw

    assert silverman_bw(u) > 0
    for kernel in ("gaussian", "epanechnikov"):
        h = sheather_jones_bw(u, kernel=kernel)
        assert np.isfinite(h) and h > 0
    h_unif = sheather_jones_bw(np.linspace(0, 1, 5000), kernel="gaussian")
    assert h_unif <= silverman_bw(np.linspace(0, 1, 5000))
    assert select_diff_bandwidth(u, method=0.42) == pytest.approx(0.42)
    with pytest.raises(ValueError):
        select_diff_bandwidth(u, method="bogus", kernel="gaussian")


def test_build_cdf():
    """Monotonic, in-bounds, inverse roundtrip, rho integrates to 1,
    rho_prime sign at mode, empty raises."""
    rng = np.random.default_rng(13)
    z = rng.normal(50.0, 5.0, size=20_000)
    P, P_inv, rho, rho_prime, _, _ = build_cdf(z)
    z_grid = np.linspace(40, 60, 101)
    u_grid = P(z_grid)
    assert np.all(np.diff(u_grid) >= -1e-12)
    assert np.all(u_grid >= 0) and np.all(u_grid <= 1)
    z_back = P_inv(P(z_grid))
    assert np.allclose(z_back, z_grid, atol=0.1)
    rho_vals = rho(z_grid)
    assert np.all(rho_vals >= 0)
    assert np.trapezoid(rho_vals, z_grid) == pytest.approx(1.0, abs=0.1)
    rho_p = rho_prime(np.array([45.0, 50.0, 55.0]))
    assert rho_p[0] > 0 and rho_p[2] < 0
    assert rho_p[1] == pytest.approx(0.0, abs=2e-3)
    with pytest.raises(ValueError):
        build_cdf(np.array([]))


def test_kde_estimate():
    """Constant estimator, weighted mean, bandwidth validation, empty,
    ESS bound, kernels agree at wide bandwidth."""
    rng = np.random.default_rng(16)
    u = rng.uniform(0.0, 1.0, size=5_000)
    D, D_std, n_eff = kde_estimate(
        u, np.full(5000, 0.3), np.array([0.25, 0.5, 0.75]), h=0.05
    )
    assert np.allclose(D, 0.3, atol=1e-10)
    assert np.all(n_eff > 0)
    # Two clusters.
    u2 = np.concatenate([np.full(2000, 0.2), np.full(2000, 0.8)])
    d2 = np.concatenate([np.full(2000, 1.0), np.full(2000, 2.0)])
    D2, _, _ = kde_estimate(u2, d2, np.array([0.2, 0.8]), h=0.05)
    assert D2[0] == pytest.approx(1.0, abs=0.05)
    assert D2[1] == pytest.approx(2.0, abs=0.05)
    with pytest.raises(ValueError):
        kde_estimate(u, np.ones(5000), np.array([0.5]), h=0.0)
    De, _, ne = kde_estimate(
        np.array([]), np.array([]), np.linspace(0.1, 0.9, 5), h=0.1
    )
    assert np.all(De == 0) and np.all(ne == 0)
    # Kernels agree at wide bandwidth.
    d3 = rng.normal(1.0, 0.2, size=2000)
    Dg, _, _ = kde_estimate(
        u2[:2000], d3, np.array([0.5]), h=2.0,
        kernel="gaussian",
    )
    Dep, _, _ = kde_estimate(
        u2[:2000], d3, np.array([0.5]), h=2.0, kernel="epanechnikov"
    )
    assert Dg[0] == pytest.approx(np.mean(d3), abs=0.05)
    assert Dep[0] == pytest.approx(np.mean(d3), abs=0.05)


@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov"])
def test_diffusivity_qkde(diff_universe, kernel):
    """Run, attributes, known-D recovery, auto bandwidth."""
    D_perp_true, D_para_true = 0.05, 0.1
    u = diff_universe(
        n_atoms=400,
        n_frames=40,
        Lz=60.0,
        D_perp=D_perp_true,
        D_para=D_para_true,
        seed=21,
    )
    ag = u.select_atoms("all")
    kde = LocalDiffusivityQKDE(
        ag,
        dim=2,
        n_points=60,
        bandwidth="auto",
        kernel=kernel,
    )
    kde.run()
    assert kde.D_perp.shape == (60,)
    assert kde.D_para.shape == (60,)
    assert kde.n_increments == 400 * 39
    assert kde.dt == pytest.approx(1.0)
    assert np.all(kde.D_perp >= 0)
    bulk = (kde.z_eval > 20) & (kde.z_eval < 40)
    valid = kde.n_eff_perp > 5
    assert np.median(kde.D_perp[bulk & valid]) == pytest.approx(
        D_perp_true, rel=0.3
    )
    assert np.median(kde.D_para[bulk & valid]) == pytest.approx(
        D_para_true, rel=0.3
    )


@pytest.mark.parametrize(
    "kwargs", [{"dim": 5}, {"bandwidth": "bogus"}, {"kernel": "tri"}]
)
def test_diffusivity_qkde_validation(diff_universe, kwargs):
    u = diff_universe(n_atoms=10, n_frames=3, Lz=10.0)
    with pytest.raises(ValueError):
        LocalDiffusivityQKDE(u.select_atoms("all"), **kwargs)


def test_diffusivity_qkde_single_frame_raises(diff_universe):
    u = diff_universe(n_atoms=10, n_frames=1, Lz=10.0)
    kde = LocalDiffusivityQKDE(
        u.select_atoms("all"), n_points=10, bandwidth=0.1
    )
    with pytest.raises(ValueError):
        kde.run()


def test_diffusivity_qkde_explicit_dt(diff_universe):
    u = diff_universe(n_atoms=50, n_frames=5, Lz=40.0, seed=24)
    kde = LocalDiffusivityQKDE(
        u.select_atoms("all"), n_points=20, bandwidth=0.1, dt=2.0
    )
    kde.run()
    assert kde.dt == pytest.approx(2.0)


@pytest.mark.parametrize("ito", [False, True])
def test_ito_correction(diff_universe, ito):
    """Default-off (None), on (finite non-neg array), reduces D_perp,
    no effect on D_para, recovers known D, zero bias for uniform density."""
    u = diff_universe(n_atoms=200, n_frames=20, Lz=60.0, seed=32)
    ag = u.select_atoms("all")
    kde = LocalDiffusivityQKDE(
        ag,
        dim=2,
        n_points=40,
        bandwidth=0.08,
        ito_correction=ito,
    )
    kde.run()
    if not ito:
        assert kde.ito_bias is None
        return
    assert kde.ito_bias is not None
    assert np.all(kde.ito_bias >= 0)
    # Compare with uncorrected.
    kde_unc = LocalDiffusivityQKDE(ag, dim=2, n_points=40, bandwidth=0.08)
    kde_unc.run()
    assert np.all(kde.D_perp <= kde_unc.D_perp + 1e-12)
    assert np.allclose(kde.D_para, kde_unc.D_para)
    # Bulk bias should be tiny (uniform density -> rho' ~ 0).
    bulk = (kde.z_eval > 20) & (kde.z_eval < 40)
    assert np.all(kde.ito_bias[bulk] < 1e-3)
    # Still recovers known D in bulk.
    u2 = diff_universe(n_atoms=400, n_frames=40, Lz=60.0, D_perp=0.05, seed=35)
    kde2 = LocalDiffusivityQKDE(
        u2.select_atoms("all"),
        n_points=60,
        bandwidth=0.08,
        ito_correction=True,
    )
    kde2.run()
    bulk2 = (kde2.z_eval > 20) & (kde2.z_eval < 40)
    valid2 = kde2.n_eff_perp > 5
    assert np.median(kde2.D_perp[bulk2 & valid2]) == pytest.approx(
        0.05, rel=0.3
    )


def test_package_exports():
    """All public names are importable from qdiffusivity."""
    for name in (
        "TransverseNumDensityQKDE",
        "TransverseNumDensityQBinned",
        "TransverseMassDensityQKDE",
        "TransverseMassDensityQBinned",
        "LocalDiffusivityQKDE",
        "LocalDiffusivityQBinned",
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
    ):
        assert hasattr(qdiffusivity, name)
