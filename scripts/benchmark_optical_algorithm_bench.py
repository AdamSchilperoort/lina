#!/usr/bin/env python3
"""Optical/control algorithm benchmark: mft, efc, iefc, aefc."""

from __future__ import annotations

import argparse
import json
import time
import numpy as np

MFT_INPUTS = (512, 1024, 2048)
MFT_OUTPUTS = (256, 512, 1024, 2048)
MFT_CASES_FULL = [(n_in, n_out) for n_in in MFT_INPUTS for n_out in MFT_OUTPUTS]
MFT_CASES_SMALL = [(512, 256), (1024, 512), (2048, 1024)]


def _load_grid_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_mft_cases(values):
    out = []
    for item in values:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid MFT case: {item!r} (expected [input, output])")
        n_in, n_out = int(item[0]), int(item[1])
        if n_in <= 0 or n_out <= 0:
            raise ValueError(f"Invalid MFT case: {item!r} (dimensions must be positive)")
        out.append((n_in, n_out))
    if not out:
        raise ValueError("MFT case list is empty after parsing.")
    return out


def _resolve_mft_cases(cfg, full_compute):
    if full_compute:
        if "mft_cases_full" in cfg:
            return _normalize_mft_cases(cfg["mft_cases_full"])
        if "mft_cases" in cfg:
            return _normalize_mft_cases(cfg["mft_cases"])
        if "mft_inputs" in cfg and "mft_outputs" in cfg:
            inputs = [int(v) for v in cfg["mft_inputs"]]
            outputs = [int(v) for v in cfg["mft_outputs"]]
            return [(n_in, n_out) for n_in in inputs for n_out in outputs]
        return MFT_CASES_FULL
    if "mft_cases_small" in cfg:
        return _normalize_mft_cases(cfg["mft_cases_small"])
    return MFT_CASES_SMALL


def avg_ms(fn, iters=20, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / max(1, iters)


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


def run_efc(pkg):
    data = pkg.efc.init_data()
    state = _dummy_dm_state()
    wfs_mask = np.zeros((4, 4), dtype=bool)
    wfs_mask[1, 1] = True
    wfs_mask[1, 2] = True
    dm_mask = np.ones((2, 2), dtype=bool)
    control_matrix = np.eye(4)
    pkg.efc.run(
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


def run_iefc(pkg):
    data = pkg.iefc.init_data()
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
    pkg.iefc.run(
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


def run_aefc(pkg):
    data = pkg.aefc.init_data()
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

    pkg.aefc.run(
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
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument(
        "--full-compute",
        action="store_true",
        help="Run full MFT input/output grid (default runs capped subset).",
    )
    ap.add_argument(
        "--sizes-file",
        type=str,
        default="",
        help="Optional JSON file with MFT size grid.",
    )
    args = ap.parse_args()

    import lina
    import lina_cpp

    print("| algorithm | backend | ms |")
    print("| --- | --- | ---:|")

    # mft
    cfg = _load_grid_config(args.sizes_file)
    cases = _resolve_mft_cases(cfg, args.full_compute)
    for n_in, n_out in cases:
        rng_r = np.random.default_rng(1000 + n_in * 10 + n_out)
        rng_i = np.random.default_rng(2000 + n_in * 10 + n_out)
        wf = (rng_r.random((n_in, n_in)) + 1j * rng_i.random((n_in, n_in))).astype(np.complex128)
        iters = max(1, min(args.iters, 2 if (n_in >= 2048 or n_out >= 2048) else args.iters))
        size = f"in{n_in}->out{n_out}"
        print(f"| mft_forward[{size}] | lina | {avg_ms(lambda: lina.props.mft_forward(wf, n_in, n_out, 1.0), iters):.3f} |")
        print(f"| mft_forward[{size}] | lina_cpp | {avg_ms(lambda: lina_cpp.props.mft_forward(wf, n_in, n_out, 1.0), iters):.3f} |")
        fpwf = lina.props.mft_forward(wf, n_in, n_out, 1.0)
        print(f"| mft_reverse[{size}] | lina | {avg_ms(lambda: lina.props.mft_reverse(fpwf, 1.0, n_in, n_in), iters):.3f} |")
        print(f"| mft_reverse[{size}] | lina_cpp | {avg_ms(lambda: lina_cpp.props.mft_reverse(fpwf, 1.0, n_in, n_in), iters):.3f} |")

    # control algorithms
    print(f"| efc.run(1 itr) | lina | {avg_ms(lambda: run_efc(lina), args.iters):.3f} |")
    print(f"| efc.run(1 itr) | lina_cpp | {avg_ms(lambda: run_efc(lina_cpp), args.iters):.3f} |")
    print(f"| iefc.run(1 itr) | lina | {avg_ms(lambda: run_iefc(lina), args.iters):.3f} |")
    print(f"| iefc.run(1 itr) | lina_cpp | {avg_ms(lambda: run_iefc(lina_cpp), args.iters):.3f} |")
    print(f"| aefc.run(1 itr) | lina | {avg_ms(lambda: run_aefc(lina), args.iters):.3f} |")
    print(f"| aefc.run(1 itr) | lina_cpp | {avg_ms(lambda: run_aefc(lina_cpp), args.iters):.3f} |")


if __name__ == "__main__":
    main()
