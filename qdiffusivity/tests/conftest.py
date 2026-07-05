"""Shared pytest fixtures for the qdiffusivity test suite."""

import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.coordinates.memory import MemoryReader


def _make_universe(n_atoms, n_res, n_frames, Lx, Ly, Lz, pos, seed):
    """Build a Universe with masses, types, residues, and a trajectory."""
    u = mda.Universe.empty(
        n_atoms=n_atoms,
        n_residues=n_res,
        atom_resindex=np.arange(n_atoms) % n_res,
        trajectory=True,
    )
    u.add_TopologyAttr("masses", values=np.ones(n_atoms))
    u.add_TopologyAttr("types", values=["X"] * n_atoms)
    u.add_TopologyAttr("resids", values=np.arange(1, n_res + 1))
    u.add_TopologyAttr("resnames", values=["RES"] * n_res)
    dims = np.tile([Lx, Ly, Lz, 90.0, 90.0, 90.0], (n_frames, 1))
    u.trajectory = MemoryReader(pos, dimensions=dims)
    return u


@pytest.fixture
def diff_universe():
    """Universe whose atoms perform independent Brownian walks.

    Atoms diffuse with prescribed D_perp (along z) and D_para (along
    x/y) with dt=1.  z is wrapped to [0, Lz]; x/y are wrapped for the
    topology (NoJump unwraps them during analysis).
    """

    def _build(
        n_atoms=100,
        n_frames=20,
        Lx=30.0,
        Ly=30.0,
        Lz=80.0,
        D_perp=0.05,
        D_para=0.1,
        seed=0,
    ):
        rng = np.random.default_rng(seed)
        pos = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
        z = rng.uniform(0.0, Lz, size=n_atoms)
        x = rng.uniform(0.0, Lx, size=n_atoms)
        y = rng.uniform(0.0, Ly, size=n_atoms)
        sp = np.sqrt(2.0 * D_perp)
        sq = np.sqrt(2.0 * D_para)
        for f in range(n_frames):
            x = (x + rng.normal(0.0, sq, size=n_atoms)) % Lx
            y = (y + rng.normal(0.0, sq, size=n_atoms)) % Ly
            z = (z + rng.normal(0.0, sp, size=n_atoms)) % Lz
            pos[f, :, 0] = x
            pos[f, :, 1] = y
            pos[f, :, 2] = z
        return _make_universe(n_atoms, n_atoms, n_frames, Lx, Ly, Lz, pos, seed)

    return _build


@pytest.fixture
def density_universe():
    """Universe with atoms random-walking in z (no drift, uniform density)."""

    def _build(
        n_atoms=200, n_res=10, n_frames=5,
        Lx=20.0, Ly=20.0, Lz=100.0, seed=0,
    ):
        rng = np.random.default_rng(seed)
        pos = np.empty((n_frames, n_atoms, 3), dtype=np.float64)
        base = rng.uniform(0.0, Lz, size=n_atoms)
        for f in range(n_frames):
            pos[f, :, 0] = rng.uniform(0.0, Lx, size=n_atoms)
            pos[f, :, 1] = rng.uniform(0.0, Ly, size=n_atoms)
            base = (base + rng.normal(0.0, 1.0, size=n_atoms)) % Lz
            pos[f, :, 2] = base
        return _make_universe(n_atoms, n_res, n_frames, Lx, Ly, Lz, pos, seed)

    return _build
