"""Properties that must hold whatever the caller does to batching, threading or ordering.

These are the screens that catch the failure modes threading actually produces: a shared
scratch buffer, a cache without a lock, an offset that assumes one molecule per batch. They
compare a run against another run, so they need no oracle.
"""
import numpy as np
import pytest

import molhume


@pytest.fixture(scope="module")
def base(smiles):
    X = molhume.featurize(smiles, standardize="none", threads=1, batch_size=4096)
    return X


def _same(a, b, what):
    assert a.shape == b.shape, f"{what}: shape {a.shape} vs {b.shape}"
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{what}: NaN pattern differs"
    m = np.isfinite(a)
    if not np.array_equal(a[m], b[m]):
        d = np.abs(a[m] - b[m])
        assert False, f"{what}: {int((d > 0).sum())} value(s) differ, largest {d.max():.6g}"


@pytest.mark.parametrize("threads", [0, 1, 2, 8])
def test_thread_count_does_not_change_values(smiles, base, threads):
    X = molhume.featurize(smiles, standardize="none", threads=threads)
    _same(base, X, f"threads={threads}")


@pytest.mark.parametrize("batch_size", [1, 7, 64, 100_000])
def test_batch_size_does_not_change_values(smiles, base, batch_size):
    X = molhume.featurize(smiles, standardize="none", batch_size=batch_size)
    _same(base, X, f"batch_size={batch_size}")


def test_row_order_does_not_change_values(smiles, base):
    order = np.random.default_rng(0).permutation(len(smiles))
    X = molhume.featurize([smiles[i] for i in order], standardize="none")
    _same(base[order], X, "shuffled input")


def test_a_molecule_alone_matches_the_same_molecule_in_a_batch(smiles, base):
    for i in (0, len(smiles) // 2, len(smiles) - 1):
        X = molhume.featurize([smiles[i]], standardize="none")
        _same(base[i:i + 1], X, f"molecule {i} alone")


def test_repeated_calls_agree(smiles, base):
    X = molhume.featurize(smiles, standardize="none")
    _same(base, X, "second call")


def test_a_repeated_molecule_gives_repeated_rows(smiles):
    X = molhume.featurize([smiles[3]] * 5, standardize="none")
    _same(X[:1].repeat(5, axis=0), X, "same molecule five times")


def test_fingerprint_bits_do_not_depend_on_descriptor_flags(smiles):
    n_all = len(molhume.column_set("full"))          # NOT len(ALL_COLUMNS): that counts `qed`,
    n_few = len(molhume.column_set("full_no_new"))   # which is opt-in and in no set
    a = molhume.featurize(smiles, standardize="none", columns="full")
    b = molhume.featurize(smiles, standardize="none", columns="full_no_new")
    assert np.array_equal(a[:, n_all:], b[:, n_few:])
