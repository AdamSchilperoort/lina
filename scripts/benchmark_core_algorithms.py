#!/usr/bin/env python3
"""Benchmark core lina vs lina_cpp algorithms on identical inputs."""

from __future__ import annotations

import argparse
import time

import numpy as np


def _bench(label, fn, reps: int, warmup: int = 1):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ms = (time.perf_counter() - t0) * 1000.0 / reps
    print(f"{label:<30} {ms:9.3f} ms")
    return ms


def _dummy_dm_state():
    return {"cmd": np.zeros((2, 2), dtype=np.float64)}


def _set_dm(state, cmd):
    state["cmd"] = np.asarray(cmd, dtype=np.float64).copy()


def _take_im(state):
    return np.ones((4, 4), dtype=np.float64) + 0.1 * float(np.sum(state["cmd"]))


def _estimate_ef():
    e = np.zeros((4, 4), dtype=np.complex128)
    e[1, 1] = 1.0 + 2.0j
    e[1, 2] = -0.5 + 0.25j
    return e


def _run_efc(pkg):
    mod = pkg.efc
    data = mod.init_data()
    state = _dummy_dm_state()
    wfs_mask = np.zeros((4, 4), dtype=bool)
    wfs_mask[1, 1] = True
    wfs_mask[1, 2] = True
    dm_mask = np.ones((2, 2), dtype=bool)
    control_matrix = np.eye(4)
    mod.run(
        data,
        take_im_fun=lambda: _take_im(state),
        take_im_params={},
        set_dm_fun=lambda cmd: _set_dm(state, cmd),
        set_dm_params={},
        estimate_ef_fun=_estimate_ef,
        estimate_ef_params={},
        wfs_mask=wfs_mask,
        dm_mask=dm_mask,
        control_matrix=control_matrix,
        num_iterations=1,
        plot_current=False,
        plot_all=False,
    )
    return data


def _run_iefc(pkg):
    mod = pkg.iefc
    data = mod.init_data()
    state = _dummy_dm_state()
    wfs_mask = np.zeros((4, 4), dtype=bool)
    wfs_mask[1, 1] = True
    wfs_mask[1, 2] = True
    probe_modes = np.zeros((2, 2, 2), dtype=np.float64)
    probe_modes[0, 0, 0] = 1.0
    probe_modes[1, 1, 1] = -1.0
    calib_modes = np.zeros((4, 2, 2), dtype=np.float64)
    for i in range(4):
        calib_modes[i].flat[i] = 1.0
    control_matrix = np.eye(4)
    mod.run(
        data,
        take_im_fun=lambda: _take_im(state),
        take_im_params={},
        set_dm_fun=lambda cmd: _set_dm(state, cmd),
        set_dm_params={},
        control_matrix=control_matrix,
        probe_modes=probe_modes,
        probe_amplitude=0.5,
        calib_modes=calib_modes,
        wfs_mask=wfs_mask,
        num_iterations=1,
        gain=0.75,
        leakage=0.0,
        plot_current=False,
        plot_all=False,
        verbose=False,
    )
    return data


def _run_aefc(pkg):
    mod = pkg.aefc
    data = mod.init_data()
    state = _dummy_dm_state()
    wfs_mask = np.zeros((4, 4), dtype=bool)
    wfs_mask[1, 1] = True
    wfs_mask[1, 2] = True
    dm_mask = np.ones((2, 2), dtype=bool)

    class DummyModel:
        Nacts = 4
        wavelength_c = 1.0

        @staticmethod
        def forward(_acts, *_args, **_kwargs):
            return np.zeros((4, 4), dtype=np.complex128)

    def val_and_grad(x, _M, _vars, *_args):
        return float(np.dot(x, x)), 2.0 * x

    mod.run(
        data,
        take_im_fun=lambda: _take_im(state),
        take_im_params={},
        set_dm_fun=lambda cmd: _set_dm(state, cmd),
        set_dm_params={},
        estimate_ef_fun=_estimate_ef,
        estimate_ef_params={},
        M=DummyModel(),
        val_and_grad=val_and_grad,
        wfs_mask=wfs_mask,
        dm_mask=dm_mask,
        num_iterations=1,
        plot_current=False,
        plot_all=False,
    )
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    import lina
    import lina_cpp

    print("\n=== Correctness spot checks (same inputs) ===")
    py_efc, cpp_efc = _run_efc(lina), _run_efc(lina_cpp)
    py_iefc, cpp_iefc = _run_iefc(lina), _run_iefc(lina_cpp)
    py_aefc, cpp_aefc = _run_aefc(lina), _run_aefc(lina_cpp)
    print("efc command allclose :", np.allclose(py_efc["commands"][0], cpp_efc["commands"][0]))
    print("iefc command allclose:", np.allclose(py_iefc["commands"][0], cpp_iefc["commands"][0]))
    print("aefc command allclose:", np.allclose(py_aefc["commands"][0], cpp_aefc["commands"][0]))

    print("\n=== Core algorithm timing (avg per run) ===")
    print(f"{'name':<30} {'time':>9}")
    rng = np.random.default_rng(7)
    A = rng.standard_normal((512, 256), dtype=np.float32)
    cA = (rng.standard_normal((512, 512)) + 1j * rng.standard_normal((512, 512))).astype(np.complex128)

    _bench("numpy.linalg.svd", lambda: np.linalg.svd(A, full_matrices=True), reps=args.reps)
    if hasattr(lina_cpp, "svd_float_cpu"):
        _bench("lina_cpp.svd_float_cpu", lambda: lina_cpp.svd_float_cpu(A), reps=args.reps)
    if hasattr(lina_cpp, "svd_float_gpu"):
        _bench("lina_cpp.svd_float_gpu", lambda: lina_cpp.svd_float_gpu(A), reps=args.reps)

    _bench("numpy centered fft", lambda: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(cA))), reps=args.reps)
    if hasattr(lina_cpp, "fft_cpu"):
        _bench("lina_cpp.fft_cpu", lambda: lina_cpp.fft_cpu(cA), reps=args.reps)
    if hasattr(lina_cpp, "fft_gpu"):
        _bench("lina_cpp.fft_gpu", lambda: lina_cpp.fft_gpu(cA), reps=args.reps)

    _bench("lina.efc.run (1 itr)", lambda: _run_efc(lina), reps=args.reps)
    _bench("lina_cpp.efc.run (1 itr)", lambda: _run_efc(lina_cpp), reps=args.reps)
    _bench("lina.iefc.run (1 itr)", lambda: _run_iefc(lina), reps=args.reps)
    _bench("lina_cpp.iefc.run (1 itr)", lambda: _run_iefc(lina_cpp), reps=args.reps)
    _bench("lina.aefc.run (1 itr)", lambda: _run_aefc(lina), reps=args.reps)
    _bench("lina_cpp.aefc.run (1 itr)", lambda: _run_aefc(lina_cpp), reps=args.reps)


if __name__ == "__main__":
    main()
