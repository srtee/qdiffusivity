"""Unit and regression tests for the diffusivity KDE module."""

import numpy as np
import pytest

import qdiffusivity
from qdiffusivity.diffusivity import (
    LocalDiffusivityKDE,
    build_cdf,
    gaussian_kernel,
    kde_estimate,
    select_diff_bandwidth,
    sheather_jones_bw,
    silverman_bw,
)

# ---------------------------------------------------------------------------
# Gaussian kernel
# ---------------------------------------------------------------------------


def test_gaussian_kernel_normalised():
    x = np.linspace(-10.0, 10.0, 200_001)
    h = 1.0
    integral = np.trapezoid(gaussian_kernel(x, h), x)
    assert integral == pytest.approx(1.0, abs=1e-4)


def test_gaussian_kernel_symmetric():
    h = 1.5
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    k = gaussian_kernel(x, h)
    assert k[0] == pytest.approx(k[-1])
    assert k[1] == pytest.approx(k[-2])
    assert k[2] > k[1] > k[0]


def test_gaussian_kernel_peak_at_zero():
    h = 2.0
    k0 = gaussian_kernel(np.array([0.0]), h)[0]
    assert k0 == pytest.approx(1.0 / (h * np.sqrt(2.0 * np.pi)))


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------


def test_silverman_bw_basic():
    rng = np.random.default_rng(10)
    u = rng.uniform(0.0, 1.0, size=10_000)
    h = silverman_bw(u)
    assert h > 0
    assert h == pytest.approx(
        1.06 * np.std(u, ddof=1) * 10_000 ** (-0.2), rel=0.3
    )


def test_silverman_bw_small_input():
    assert silverman_bw(np.array([0.5])) == 0.05
    assert silverman_bw(np.array([])) == 0.05


@pytest.mark.parametrize("kernel", ["gaussian", "epanechnikov"])
def test_sheather_jones_bw_positive(kernel):
    rng = np.random.default_rng(11)
    u = rng.uniform(0.0, 1.0, size=5_000)
    h = sheather_jones_bw(u, kernel=kernel)
    assert np.isfinite(h)
    assert h > 0


def test_sheather_jones_bw_falls_back_for_uniform():
    u = np.linspace(0.0, 1.0, 5_000)
    h_sj = sheather_jones_bw(u, kernel="gaussian")
    h_silver = silverman_bw(u)
    assert h_sj > 0
    assert np.isfinite(h_sj)
    assert h_sj <= h_silver


def test_select_diff_bandwidth_methods():
    rng = np.random.default_rng(12)
    u = rng.uniform(0.0, 1.0, size=2_000)
    h_auto = select_diff_bandwidth(u, method="auto", kernel="gaussian")
    h_silver = select_diff_bandwidth(u, method="silverman")
    h_fixed = select_diff_bandwidth(u, method=0.42)
    assert h_auto > 0
    assert h_silver > 0
    assert h_fixed == pytest.approx(0.42)


def test_select_diff_bandwidth_invalid_method():
    with pytest.raises(ValueError):
        select_diff_bandwidth(
            np.array([0.5, 0.6]), method="notamethod", kernel="gaussian"
        )


# ---------------------------------------------------------------------------
# build_cdf
# ---------------------------------------------------------------------------


def test_build_cdf_monotonic_in_bounds():
    rng = np.random.default_rng(13)
    z = rng.normal(50.0, 5.0, size=20_000)
    P, P_inv, rho, rho_prime, z_sorted, p_vals = build_cdf(z)
    z_grid = np.linspace(40.0, 60.0, 101)
    u_grid = P(z_grid)
    assert np.all(np.diff(u_grid) >= -1e-12)
    assert np.all(u_grid >= 0.0)
    assert np.all(u_grid <= 1.0)
    assert u_grid[0] == pytest.approx(0.0, abs=0.05)
    assert u_grid[-1] == pytest.approx(1.0, abs=0.05)


def test_build_cdf_inverse_roundtrip():
    rng = np.random.default_rng(14)
    z = rng.normal(0.0, 1.0, size=10_000)
    P, P_inv, rho, rho_prime, z_sorted, p_vals = build_cdf(z)
    z_test = np.linspace(-2.0, 2.0, 21)
    u_test = P(z_test)
    z_back = P_inv(u_test)
    assert np.allclose(z_back, z_test, atol=0.1)


def test_build_cdf_rho_positive_and_integrates_to_one():
    rng = np.random.default_rng(15)
    z = rng.normal(50.0, 5.0, size=20_000)
    P, P_inv, rho, rho_prime, z_sorted, p_vals = build_cdf(z)
    z_grid = np.linspace(40.0, 60.0, 2001)
    rho_vals = rho(z_grid)
    assert np.all(rho_vals >= 0.0)
    integral = np.trapezoid(rho_vals, z_grid)
    assert integral == pytest.approx(1.0, abs=0.1)


def test_build_cdf_rho_prime_zero_at_mode():
    # For a symmetric Gaussian-like density the derivative rho' should
    # be ~zero at the mode and have opposite signs on either side.
    rng = np.random.default_rng(19)
    z = rng.normal(50.0, 5.0, size=40_000)
    P, P_inv, rho, rho_prime, z_sorted, p_vals = build_cdf(z)
    rho(np.array([45.0, 50.0, 55.0]))  # exercise rho closure
    rho_p = rho_prime(np.array([45.0, 50.0, 55.0]))
    # Derivative positive below mode, ~zero at mode, negative above.
    assert rho_p[0] > 0
    assert rho_p[2] < 0
    assert rho_p[1] == pytest.approx(0.0, abs=2e-3)


def test_build_cdf_rho_prime_finite():
    rng = np.random.default_rng(20)
    z = rng.uniform(0.0, 100.0, size=20_000)
    P, P_inv, rho, rho_prime, z_sorted, p_vals = build_cdf(z)
    z_grid = np.linspace(10.0, 90.0, 101)
    rho_p = rho_prime(z_grid)
    assert np.all(np.isfinite(rho_p))


def test_build_cdf_empty_raises():
    with pytest.raises(ValueError):
        build_cdf(np.array([]))


# ---------------------------------------------------------------------------
# kde_estimate
# ---------------------------------------------------------------------------


def test_kde_estimate_constant_estimator():
    # If every d_j is the same constant, the KDE should return that
    # constant at every well-sampled evaluation point.
    rng = np.random.default_rng(16)
    u = rng.uniform(0.0, 1.0, size=5_000)
    d = np.full(u.size, 0.3)
    u_eval = np.array([0.25, 0.5, 0.75])
    D, D_std, n_eff = kde_estimate(u, d, u_eval, h=0.05, kernel="gaussian")
    assert D.shape == (3,)
    assert np.allclose(D, 0.3, atol=1e-10)
    assert np.all(D_std == pytest.approx(0.0, abs=1e-9))
    assert np.all(n_eff > 0)


def test_kde_estimate_weighted_mean():
    # Two clusters of equal size with different d values; near the
    # cluster the KDE should approach that cluster's d value.
    u = np.concatenate([np.full(2000, 0.2), np.full(2000, 0.8)])
    d = np.concatenate([np.full(2000, 1.0), np.full(2000, 2.0)])
    u_eval = np.array([0.2, 0.8])
    D, D_std, n_eff = kde_estimate(u, d, u_eval, h=0.05, kernel="gaussian")
    assert D[0] == pytest.approx(1.0, abs=0.05)
    assert D[1] == pytest.approx(2.0, abs=0.05)


def test_kde_estimate_rejects_nonpositive_bandwidth():
    with pytest.raises(ValueError):
        kde_estimate(np.array([0.5]), np.array([1.0]), np.array([0.5]), h=0.0)


def test_kde_estimate_empty_data():
    D, D_std, n_eff = kde_estimate(
        np.array([]), np.array([]), np.linspace(0.1, 0.9, 5), h=0.1
    )
    assert np.all(D == 0.0)
    assert np.all(D_std == 0.0)
    assert np.all(n_eff == 0.0)


def test_kde_estimate_n_eff_bounded_by_n():
    rng = np.random.default_rng(17)
    u = rng.uniform(0.0, 1.0, size=1_000)
    d = rng.normal(1.0, 0.1, size=1_000)
    u_eval = np.array([0.5])
    _, _, n_eff = kde_estimate(u, d, u_eval, h=0.1)
    # Mirror-reflection triples the augmented sample, so n_eff <= 3*n.
    assert n_eff[0] > 0
    assert n_eff[0] <= 3 * 1_000


def test_kde_estimate_kernels_match_at_wide_bandwidth():
    # With a very wide bandwidth both kernels give a near-flat estimate
    # equal to the global mean of d.
    rng = np.random.default_rng(18)
    u = rng.uniform(0.0, 1.0, size=2_000)
    d = rng.normal(1.0, 0.2, size=2_000)
    u_eval = np.array([0.5])
    D_g, _, _ = kde_estimate(u, d, u_eval, h=2.0, kernel="gaussian")
    D_e, _, _ = kde_estimate(u, d, u_eval, h=2.0, kernel="epanechnikov")
    assert D_g[0] == pytest.approx(np.mean(d), abs=0.05)
    assert D_e[0] == pytest.approx(np.mean(d), abs=0.05)


# ---------------------------------------------------------------------------
# LocalDiffusivityKDE AnalysisBase
# ---------------------------------------------------------------------------


def _make_diffusion_universe(
    n_atoms,
    n_frames,
    Lx=30.0,
    Ly=30.0,
    Lz=100.0,
    D_perp=0.05,
    D_para=0.1,
    seed=0,
):
    """Build a Universe whose atoms diffuse with prescribed D_perp/D_para.

    Atoms perform independent Brownian walks: dz ~ N(0, sqrt(2*D_perp*dt)),
    (dx, dy) ~ N(0, sqrt(2*D_para*dt)) per step, with dt = 1.0.  z is kept
    inside [0, Lz] by wrapping; x/y are wrapped for the topology but
    NoJump will unwrap them.  A uniform equilibrium density is used.
    """
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    u = mda.Universe.empty(
        n_atoms=n_atoms,
        n_residues=n_atoms,
        atom_resindex=np.arange(n_atoms),
        trajectory=True,
    )
    u.add_TopologyAttr("masses", values=np.ones(n_atoms))
    u.add_TopologyAttr("types", values=["X"] * n_atoms)
    u.add_TopologyAttr("resids", values=np.arange(1, n_atoms + 1))
    u.add_TopologyAttr("resnames", values=["RES"] * n_atoms)

    rng = np.random.default_rng(seed)
    pos = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
    z = rng.uniform(0.0, Lz, size=n_atoms)
    x = rng.uniform(0.0, Lx, size=n_atoms)
    y = rng.uniform(0.0, Ly, size=n_atoms)
    dt = 1.0
    sigma_perp = np.sqrt(2.0 * D_perp * dt)
    sigma_para = np.sqrt(2.0 * D_para * dt)
    for f in range(n_frames):
        x = (x + rng.normal(0.0, sigma_para, size=n_atoms)) % Lx
        y = (y + rng.normal(0.0, sigma_para, size=n_atoms)) % Ly
        z = (z + rng.normal(0.0, sigma_perp, size=n_atoms)) % Lz
        pos[f, :, 0] = x
        pos[f, :, 1] = y
        pos[f, :, 2] = z

    dims = np.tile([Lx, Ly, Lz, 90.0, 90.0, 90.0], (n_frames, 1))
    u.trajectory = MemoryReader(pos, dimensions=dims)
    return u


def test_diffusion_kde_runs_and_attrs():
    u = _make_diffusion_universe(n_atoms=100, n_frames=20, Lz=80.0, seed=20)
    ag = u.select_atoms("all")
    kde = LocalDiffusivityKDE(
        ag, dim=2, n_points=50, bandwidth=0.1, kernel="gaussian"
    )
    kde.run()
    assert kde.D_perp.shape == (50,)
    assert kde.D_para.shape == (50,)
    assert kde.z_eval.shape == (50,)
    assert kde.u_eval.shape == (50,)
    assert kde.bandwidth == pytest.approx(0.1)
    assert kde.n_increments == 100 * 19
    assert kde.n_frames_used == 20
    assert kde.dt == pytest.approx(1.0)


def test_diffusion_kde_recovers_known_diffusivity():
    # With a uniform equilibrium density and Brownian dynamics the KDE
    # should recover the input diffusivities in the bulk.
    D_perp_true = 0.05
    D_para_true = 0.1
    u = _make_diffusion_universe(
        n_atoms=400,
        n_frames=40,
        Lz=60.0,
        D_perp=D_perp_true,
        D_para=D_para_true,
        seed=21,
    )
    ag = u.select_atoms("all")
    kde = LocalDiffusivityKDE(
        ag, dim=2, n_points=60, bandwidth=0.08, kernel="gaussian"
    )
    kde.run()
    # Bulk = middle third of the gap.
    bulk = (kde.z_eval > 20.0) & (kde.z_eval < 40.0)
    valid = kde.n_eff_perp > 5
    bulk_perp = np.median(kde.D_perp[bulk & valid])
    bulk_para = np.median(kde.D_para[bulk & valid])
    assert bulk_perp == pytest.approx(D_perp_true, rel=0.3)
    assert bulk_para == pytest.approx(D_para_true, rel=0.3)


def test_diffusion_kde_auto_bandwidth():
    u = _make_diffusion_universe(n_atoms=100, n_frames=10, Lz=50.0, seed=22)
    ag = u.select_atoms("all")
    kde = LocalDiffusivityKDE(
        ag, dim=2, n_points=40, bandwidth="auto", kernel="gaussian"
    )
    kde.run()
    assert np.isfinite(kde.bandwidth)
    assert kde.bandwidth > 0


def test_diffusion_kde_epanechnikov_kernel():
    u = _make_diffusion_universe(n_atoms=100, n_frames=10, Lz=50.0, seed=23)
    ag = u.select_atoms("all")
    kde = LocalDiffusivityKDE(
        ag, dim=2, n_points=40, bandwidth=0.1, kernel="epanechnikov"
    )
    kde.run()
    assert np.all(kde.D_perp >= 0.0)
    assert np.all(np.isfinite(kde.D_para))


def test_diffusion_kde_dim_validation():
    u = _make_diffusion_universe(n_atoms=10, n_frames=3, Lz=10.0)
    with pytest.raises(ValueError):
        LocalDiffusivityKDE(u.select_atoms("all"), dim=5)


def test_diffusion_kde_bandwidth_validation():
    u = _make_diffusion_universe(n_atoms=10, n_frames=3, Lz=10.0)
    with pytest.raises(ValueError):
        LocalDiffusivityKDE(u.select_atoms("all"), bandwidth="bogus")


def test_diffusion_kde_kernel_validation():
    u = _make_diffusion_universe(n_atoms=10, n_frames=3, Lz=10.0)
    with pytest.raises(ValueError):
        LocalDiffusivityKDE(u.select_atoms("all"), kernel="triangular")


def test_diffusion_kde_single_frame_raises():
    u = _make_diffusion_universe(n_atoms=10, n_frames=1, Lz=10.0)
    kde = LocalDiffusivityKDE(
        u.select_atoms("all"), n_points=10, bandwidth=0.1
    )
    with pytest.raises(ValueError):
        kde.run()


def test_diffusion_kde_explicit_dt():
    u = _make_diffusion_universe(n_atoms=50, n_frames=5, Lz=40.0, seed=24)
    ag = u.select_atoms("all")
    kde = LocalDiffusivityKDE(
        ag, dim=2, n_points=20, bandwidth=0.1, dt=2.0
    )
    kde.run()
    assert kde.dt == pytest.approx(2.0)


def test_diffusion_kde_exposed_from_package():
    assert hasattr(qdiffusivity, "LocalDiffusivityKDE")
    assert hasattr(qdiffusivity, "build_cdf")
    assert hasattr(qdiffusivity, "gaussian_kernel")
    assert hasattr(qdiffusivity, "kde_estimate")
    assert hasattr(qdiffusivity, "select_diff_bandwidth")


# ---------------------------------------------------------------------------
# Itô correction
# ---------------------------------------------------------------------------


def test_ito_correction_default_is_off():
    u = _make_diffusion_universe(n_atoms=50, n_frames=5, Lz=40.0, seed=30)
    ag = u.select_atoms("all")
    kde = LocalDiffusivityKDE(ag, dim=2, n_points=20, bandwidth=0.1)
    kde.run()
    assert kde.ito_bias is None


def test_ito_correction_sets_attribute():
    u = _make_diffusion_universe(n_atoms=50, n_frames=5, Lz=40.0, seed=31)
    ag = u.select_atoms("all")
    kde = LocalDiffusivityKDE(
        ag, dim=2, n_points=20, bandwidth=0.1, ito_correction=True
    )
    kde.run()
    assert kde.ito_bias is not None
    assert kde.ito_bias.shape == (20,)
    assert np.all(np.isfinite(kde.ito_bias))
    assert np.all(kde.ito_bias >= 0.0)


def test_ito_correction_reduces_D_perp():
    # With ito_correction=True the corrected D_perp should be <= the
    # uncorrected D_perp (the bias is non-negative).
    u = _make_diffusion_universe(n_atoms=200, n_frames=20, Lz=60.0, seed=32)
    ag = u.select_atoms("all")
    kde_unc = LocalDiffusivityKDE(ag, dim=2, n_points=40, bandwidth=0.08)
    kde_unc.run()
    kde_cor = LocalDiffusivityKDE(
        ag, dim=2, n_points=40, bandwidth=0.08, ito_correction=True
    )
    kde_cor.run()
    assert np.all(kde_cor.D_perp <= kde_unc.D_perp + 1e-12)


def test_ito_correction_uniform_density_zero_bias():
    # A uniform equilibrium density has rho' == 0, so the Itô bias
    # should be identically zero and the corrected D_perp should equal
    # the uncorrected D_perp.
    u = _make_diffusion_universe(n_atoms=500, n_frames=20, Lz=60.0, seed=33)
    ag = u.select_atoms("all")
    kde = LocalDiffusivityKDE(
        ag, dim=2, n_points=40, bandwidth=0.1, ito_correction=True
    )
    kde.run()
    bulk = (kde.z_eval > 20.0) & (kde.z_eval < 40.0)
    # In the bulk the density is ~uniform, so rho' ~ 0 and bias ~ 0
    # (residual Poisson noise in rho' gives a small nonzero value).
    assert np.all(kde.ito_bias[bulk] < 1e-3)


def test_ito_correction_does_not_affect_D_para():
    # The parallel estimator has zero Itô bias; the corrected and
    # uncorrected D_para should be identical.
    u = _make_diffusion_universe(n_atoms=100, n_frames=10, Lz=50.0, seed=34)
    ag = u.select_atoms("all")
    kde_unc = LocalDiffusivityKDE(ag, dim=2, n_points=30, bandwidth=0.1)
    kde_unc.run()
    kde_cor = LocalDiffusivityKDE(
        ag, dim=2, n_points=30, bandwidth=0.1, ito_correction=True
    )
    kde_cor.run()
    assert np.allclose(kde_cor.D_para, kde_unc.D_para)


def test_ito_correction_recovers_known_diffusivity():
    # With a uniform equilibrium density the Itô bias is zero, so the
    # corrected estimator should still recover the input D_perp.
    D_perp_true = 0.05
    u = _make_diffusion_universe(
        n_atoms=400,
        n_frames=40,
        Lz=60.0,
        D_perp=D_perp_true,
        D_para=0.1,
        seed=35,
    )
    ag = u.select_atoms("all")
    kde = LocalDiffusivityKDE(
        ag,
        dim=2,
        n_points=60,
        bandwidth=0.08,
        ito_correction=True,
    )
    kde.run()
    bulk = (kde.z_eval > 20.0) & (kde.z_eval < 40.0)
    valid = kde.n_eff_perp > 5
    bulk_perp = np.median(kde.D_perp[bulk & valid])
    assert bulk_perp == pytest.approx(D_perp_true, rel=0.3)
