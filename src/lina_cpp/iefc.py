"""iEFC utilities for lina_cpp.

This module keeps API parity with ``lina.iefc`` but implements hot loops
locally so we can reduce Python overhead while preserving math.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from lina_cpp._pyref import coro_utils, utils
from lina_cpp._pyref.math_module import ensure_np_array, xp

from . import _core

_HAS_CPP_IEFC_CALIBRATE = hasattr(_core, "control_model_iefc_calibrate")


def _extract_cpp_control_model(take_im_params, set_dm_params):
    if not isinstance(take_im_params, dict) or not isinstance(set_dm_params, dict):
        return None
    model_take = take_im_params.get("C")
    model_set = set_dm_params.get("C")
    if model_take is None or model_take is not model_set:
        return None
    if not hasattr(model_take, "_cpp_model"):
        return None
    return model_take


def _should_use_cpp_calibrate(model) -> bool:
    return model is not None and _HAS_CPP_IEFC_CALIBRATE


def calibrate_control_model(
    model,
    wfs_mask,
    probe_modes,
    probe_amplitude,
    calibration_modes,
    calibration_amplitude,
    scale_factors=None,
    initial_command=None,
    set_dm_params=None,
    use_vortex=None,
    plot_responses=False,
):
    """Native C++ iEFC calibration endpoint for ControlModel-backed workflows."""
    if not _HAS_CPP_IEFC_CALIBRATE:
        raise RuntimeError(
            "lina_cpp._core is missing control_model_iefc_calibrate. "
            "Restart the Python kernel after reinstalling lina_cpp."
        )
    if use_vortex is None:
        use_vortex = bool(getattr(model, "use_vortex", True))
    if set_dm_params is None:
        set_dm_params = {}
    channel = int(set_dm_params.get("channel", 1))

    dm_commands = xp.asarray(model.dm_commands)
    if channel < 0 or channel >= int(dm_commands.shape[0]):
        raise ValueError(f"Invalid DM channel index {channel}")

    # Keep native model state in sync once before entering the C++ hot loop.
    if hasattr(model, "_sync_cpp_device"):
        model._sync_cpp_device()
    if hasattr(model, "_sync_cpp_prefpm"):
        model._sync_cpp_prefpm()

    dm_static = xp.sum(dm_commands, axis=0) - dm_commands[channel]
    if initial_command is None:
        initial_channel = xp.zeros_like(dm_commands[channel])
    else:
        initial_channel = xp.asarray(initial_command)
    initial_total = dm_static + initial_channel

    scale_factors_np = (
        None
        if scale_factors is None
        else np.asarray(ensure_np_array(scale_factors), dtype=np.float64)
    )

    # Per-mode progress print, matching the pure-Python reference output.
    _start = time.time()

    def _progress(done, total):
        print(f"\tCalibrated mode {int(done):d}/{int(total):d} in {time.time() - _start:.3f}s",
              end="\r", flush=True)

    response_matrix_np, response_cube_np = _core.control_model_iefc_calibrate(
        model._cpp_model,
        np.asarray(ensure_np_array(wfs_mask), dtype=np.uint8),
        float(probe_amplitude),
        np.asarray(ensure_np_array(probe_modes), dtype=np.float64),
        float(calibration_amplitude),
        np.asarray(ensure_np_array(calibration_modes), dtype=np.float64),
        scale_factors_np,
        np.asarray(ensure_np_array(initial_total), dtype=np.float64),
        bool(use_vortex),
        float(getattr(model, "Imax_ref", 1.0)),
        _progress,
    )
    print()  # newline after the \r progress line

    # Keep model-side state parity with callback-driven calibration.
    model.dm_commands[channel] = initial_channel

    response_matrix = xp.asarray(response_matrix_np)
    response_cube = xp.asarray(response_cube_np)

    if plot_responses:
        nmodes = calibration_modes.shape[0]
        nact = int(getattr(model, "Nact", np.sqrt(calibration_modes.reshape(nmodes, -1).shape[1])))
        ncamsci = wfs_mask.shape[0]
        dm_response_map = xp.sqrt(
            xp.mean(xp.square(response_matrix.dot(calibration_modes.reshape(nmodes, -1))), axis=0)
        )
        dm_response_map = dm_response_map.reshape(nact, nact) / xp.max(dm_response_map)

        fp_response_map = xp.sqrt(xp.mean(xp.abs(response_cube), axis=(0, 1))).reshape(
            ncamsci, ncamsci
        )
        fp_response_map = fp_response_map / xp.max(fp_response_map)
        from matplotlib.colors import LogNorm
        utils.imshow(
            [dm_response_map, fp_response_map],
            titles=["DM Response Map", "Focal Plane Response Map"],
            norms=[LogNorm(1e-2), None],
        )

    return response_matrix, response_cube


def measure_probe_response(
    take_im_fun,
    take_im_params,
    set_dm_fun,
    set_dm_params,
    probe_modes,
    probe_amplitude,
    base_command=None,
    normalize_diff_fun=None,
    normalize_diff_params=None,
    verbose=False,
    plot=False,
    _scaled_probe_modes=None,
):
    """Measure normalized difference images for each probe mode."""
    nprobes = probe_modes.shape[0]
    nact = probe_modes.shape[1]
    if base_command is None:
        base_command = xp.zeros((nact, nact))

    scaled_probes = (
        probe_amplitude * probe_modes if _scaled_probe_modes is None else _scaled_probe_modes
    )
    responses = None

    for i in range(nprobes):
        if verbose:
            print(f"\tMeasuring response of probe {i + 1}/{nprobes}.")

        probe = scaled_probes[i]
        set_dm_fun(base_command + probe, **set_dm_params)
        im_pos = take_im_fun(**take_im_params)

        set_dm_fun(base_command - probe, **set_dm_params)
        im_neg = take_im_fun(**take_im_params)

        diff_im = im_pos - im_neg
        diff_im_ni = (
            diff_im
            if normalize_diff_fun is None
            else normalize_diff_fun(diff_im, **normalize_diff_params)
        )
        probe_response = diff_im_ni / (2 * probe_amplitude)

        if responses is None:
            responses = xp.empty((nprobes,) + probe_response.shape, dtype=probe_response.dtype)
        responses[i] = probe_response

        if plot:
            utils.imshow(
                [probe_modes[i], diff_im_ni],
                titles=[f"Probe Command {i+1}", "Normalized Response"],
                cmaps=["viridis", "magma"],
            )

    set_dm_fun(base_command, **set_dm_params)
    if responses is None:
        return xp.empty((0,))
    return responses


def calibrate(
    take_im_fun,
    take_im_params,
    set_dm_fun,
    set_dm_params,
    wfs_mask,
    probe_modes,
    probe_amplitude,
    calibration_modes,
    calibration_amplitude,
    scale_factors=None,
    initial_command=None,
    normalize_diff_fun=None,
    normalize_diff_params=None,
    plot_responses=False,
):
    """Calibrate iEFC response matrix for a set of calibration modes."""
    model = _extract_cpp_control_model(take_im_params, set_dm_params)
    if _should_use_cpp_calibrate(model):
        print("Calibrating iEFC... (native C++)")
        out = calibrate_control_model(
            model=model,
            wfs_mask=wfs_mask,
            probe_modes=probe_modes,
            probe_amplitude=probe_amplitude,
            calibration_modes=calibration_modes,
            calibration_amplitude=calibration_amplitude,
            scale_factors=scale_factors,
            initial_command=initial_command,
            set_dm_params=set_dm_params,
            plot_responses=plot_responses,
        )
        print("Calibration complete.")
        return out
    if model is not None and not _HAS_CPP_IEFC_CALIBRATE:
        raise RuntimeError(
            "lina_cpp._core missing native iEFC calibrate symbol. "
            "Rebuild/reinstall lina_cpp and restart the kernel to enable "
            "the C++ calibration path."
        )

    print("Calibrating iEFC...")

    nact = probe_modes.shape[1]
    nprobes = probe_modes.shape[0]
    nmodes = calibration_modes.shape[0]
    ncamsci = wfs_mask.shape[0]
    if initial_command is None:
        initial_command = xp.zeros((nact, nact))

    scaled_probes = probe_amplitude * probe_modes
    response_matrix = None
    response_cube = None

    start = time.time()
    for i, calibration_mode in enumerate(calibration_modes):
        dm_mode = calibration_mode.reshape(nact, nact)
        amp = (
            calibration_amplitude * scale_factors[i]
            if scale_factors is not None
            else calibration_amplitude
        )
        response = 0.0

        for s in (1, -1):
            base_command = initial_command + s * amp * dm_mode
            probed_diffs = measure_probe_response(
                take_im_fun,
                take_im_params,
                set_dm_fun,
                set_dm_params,
                probe_modes,
                probe_amplitude,
                base_command=base_command,
                normalize_diff_fun=normalize_diff_fun,
                normalize_diff_params=normalize_diff_params,
                _scaled_probe_modes=scaled_probes,
            )
            response = response + s * probed_diffs / (2 * amp)

        set_dm_fun(initial_command, **set_dm_params)

        if response_matrix is None:
            nmask = int(xp.sum(wfs_mask))
            response_matrix = xp.empty((nmodes, nprobes * nmask), dtype=response.dtype)
            response_cube = xp.empty((nmodes,) + response.shape, dtype=response.dtype)

        response_matrix[i] = response[:, wfs_mask].ravel()
        response_cube[i] = response

        elapsed = time.time() - start
        print(
            f"\tCalibrated mode {i + 1:d}/{nmodes:d} in {elapsed:.3f}s",
            end="\r",
            flush=True,
        )

    print("\nCalibration complete.")

    response_matrix = response_matrix.T
    if plot_responses:
        dm_response_map = xp.sqrt(
            xp.mean(xp.square(response_matrix.dot(calibration_modes.reshape(nmodes, -1))), axis=0)
        )
        dm_response_map = dm_response_map.reshape(nact, nact) / xp.max(dm_response_map)

        fp_response_map = xp.sqrt(xp.mean(xp.abs(response_cube), axis=(0, 1))).reshape(
            ncamsci, ncamsci
        )
        fp_response_map = fp_response_map / xp.max(fp_response_map)
        from matplotlib.colors import LogNorm
        utils.imshow(
            [dm_response_map, fp_response_map],
            titles=["DM Response Map", "Focal Plane Response Map"],
            norms=[LogNorm(1e-2), None],
        )

    return response_matrix, response_cube


def make_response_matrix(response_cube, wfs_mask):
    nmodes = response_cube.shape[0]
    return response_cube[:, :, wfs_mask].reshape(nmodes, -1).T


def init_data(wfs_mask=None, contrast0=None, ni_im0=None):
    return {
        "raw_images": [],
        "ni_images": [],
        "contrasts": [],
        "commands": [],
        "del_commands": [],
        "reg_conds": [],
        "wfs_mask": wfs_mask,
        "ni_im0": ni_im0,
        "contrast0": contrast0,
    }


def run(
    iefc_data,
    take_im_fun,
    take_im_params,
    set_dm_fun,
    set_dm_params,
    response_matrix,
    reg_cond,
    probe_modes,
    probe_amplitude,
    calib_modes,
    wfs_mask,
    num_iterations=3,
    gain=1.0,
    leakage=0.0,
    normalize_diff_fun=None,
    normalize_diff_params=None,
    normalize_metric_fun=None,
    normalize_metric_params=None,
    plot_current=True,
    plot_all=False,
    verbose=True,
    plot_probe_responses=False,
    vmin=1e-10,
    vmax=1e-5,
):
    """Run closed-loop iEFC updates for a fixed response matrix."""
    start = time.time()

    nact = probe_modes.shape[1]
    nmodes = calib_modes.shape[0]
    modal_matrix = calib_modes.reshape(nmodes, -1).T
    scaled_probes = probe_amplitude * probe_modes

    starting_itr = len(iefc_data["commands"]) + 1
    total_command = (
        copy.copy(iefc_data["commands"][-1])
        if len(iefc_data["commands"]) > 0
        else xp.zeros((nact, nact))
    )

    control_matrix = utils.beta_reg(response_matrix, reg_cond)

    for i in range(num_iterations):
        print(f"Running iteration {i + starting_itr} / {num_iterations + starting_itr - 1}")
        diff_ims = measure_probe_response(
            take_im_fun,
            take_im_params,
            set_dm_fun,
            set_dm_params,
            probe_modes,
            probe_amplitude,
            base_command=total_command,
            normalize_diff_fun=normalize_diff_fun,
            normalize_diff_params=normalize_diff_params,
            verbose=verbose,
            plot=plot_probe_responses,
            _scaled_probe_modes=scaled_probes,
        )
        measurement_vector = diff_ims[:, wfs_mask].ravel()

        modal_coeff = -control_matrix.dot(measurement_vector)
        del_command = gain * modal_matrix.dot(modal_coeff).reshape(nact, nact)
        total_command = (1.0 - leakage) * total_command + del_command

        set_dm_fun(total_command, **set_dm_params)
        print("Measuring dark hole state ...")
        metric_im = take_im_fun(**take_im_params)
        metric_im_ni = (
            metric_im
            if normalize_metric_fun is None
            else normalize_metric_fun(metric_im, **normalize_metric_params)
        )
        contrast = coro_utils.compute_contrast(metric_im_ni, wfs_mask)

        iefc_data["raw_images"].append(copy.copy(metric_im))
        iefc_data["ni_images"].append(copy.copy(metric_im_ni))
        iefc_data["contrasts"].append(copy.copy(contrast))
        iefc_data["commands"].append(copy.copy(total_command))
        iefc_data["del_commands"].append(copy.copy(del_command))
        iefc_data["reg_conds"].append(copy.copy(reg_cond))

        if plot_current:
            from IPython.display import clear_output
            from matplotlib.colors import CenteredNorm, LogNorm

            if not plot_all:
                clear_output(wait=True)
            utils.imshow(
                [del_command, total_command, metric_im_ni],
                titles=[
                    f"Iteration {starting_itr + i:d}: $\\delta$DM",
                    "Total DM Command",
                    f"Normalized Image\nMean Contrast = {contrast:.3e}",
                ],
                cmaps=["viridis", "viridis", "magma"],
                pxscls=[None, None, None],
                norms=[CenteredNorm(), None, LogNorm(vmin, vmax)],
            )

    print(f"Completed {num_iterations:d} iterations in {time.time() - start:.3f}s.")
    return iefc_data


def compute_hadamard_scale_factors(
    had_modes,
    scale_exp=1 / 6,
    scale_thresh=4,
    iwa=2.5,
    owa=13,
    oversamp=4,
    plot=False,
):
    """Native C++ implementation of the Hadamard scale-factor calc."""
    return _core.compute_hadamard_scale_factors(
        had_modes,
        float(scale_exp),
        float(scale_thresh),
        float(iwa),
        float(owa),
        int(oversamp),
        bool(plot),
    )
