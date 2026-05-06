import os
import sys
import unittest
import numpy as np
import scipy

from lina import llowfsc, shmim_utils, coro_utils, aefc, utils, math_module


class TestMorePython(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        math_module.update_np(np)
        math_module.update_scipy(scipy)
        shmim_path = "/opt/MagAOX/source/milk/src/ImageStreamIO/build/lib.linux-x86_64-cpython-310"
        if os.path.isdir(shmim_path) and shmim_path not in sys.path:
            sys.path.insert(0, shmim_path)

    def test_llowfsc_make_shear_chops(self):
        camlo_ref = np.arange(16, dtype=float).reshape(4, 4)
        control_mask = np.ones_like(camlo_ref, dtype=bool)
        shear_responses, shear_chops = llowfsc.make_shear_chops(
            camlo_ref, control_mask, shear_pix=0.5, order=1, central_diff=False, return_np=True
        )
        self.assertEqual(shear_responses.shape[1], 2)
        self.assertEqual(shear_chops.shape, (2, 4, 4))

    def test_coro_utils_normalize_and_contrast(self):
        raw = np.array([[2.0, 0.5], [1.0, 3.0]])
        dark = np.zeros_like(raw)
        im_params = {"exp_time": 1.0, "gain": 0.0, "atten": 0.0, "Imax": 1.0}
        ref_params = {"exp_time": 1.0, "gain": 0.0, "atten": 0.0, "Imax": 1.0}
        ni = coro_utils.normalize_coro_im(raw, im_params, ref_params, dark_im=dark)
        mask = np.array([[True, False], [True, True]])
        contrast = coro_utils.compute_contrast(ni, mask, verbose=False)
        self.assertGreater(contrast, 0.0)

    def test_aefc_run_smoke(self):
        class DummyStream:
            def __init__(self, shape):
                self.shape = shape
                self._latest = np.zeros(shape)
            def grab_latest(self):
                return self._latest
            def grab_many(self, n):
                return np.repeat(self._latest[None, ...], n, axis=0)
            def write(self, data):
                self._latest = data

        def dummy_val_and_grad(x, _M, _vars, *_args, **_kwargs):
            grad = 2 * x
            return np.sum(x ** 2), grad

        camsci = DummyStream((4, 4))
        dm = DummyStream((2, 2))
        efc_data = {
            "raw_images": [],
            "dark_images": [],
            "ni_images": [],
            "contrasts": [],
            "efields": [],
            "commands": [],
            "del_commands": [],
            "bfgs_tols": [],
            "reg_conds": [],
            "images": [],
        }
        im_params = {"exp_time": 1.0, "gain": 0.0, "atten": 0.0, "Imax": 1.0}
        ref_params = {"exp_time": 1.0, "gain": 0.0, "atten": 0.0, "Imax": 1.0}
        control_mask = np.zeros((4, 4), dtype=bool)
        control_mask[1, 1] = True
        dm_mask = np.ones((2, 2), dtype=bool)

        class DummyModel:
            Nacts = 4
            wavelength_c = 1.0
            def forward(self, _acts, *_args, **_kwargs):
                return np.zeros((4, 4), dtype=np.complex128)

        def dummy_pwp_run(*_args, **_kwargs):
            field = np.zeros((4, 4), dtype=np.complex128)
            field_vec = np.zeros(2, dtype=np.complex128)
            return field, field_vec

        import lina.pwp as pwp_mod
        pwp_mod.run = dummy_pwp_run

        utils.imshow = lambda *_args, **_kwargs: None
        aefc.run(
            efc_data,
            camsci,
            dm,
            im_params,
            ref_params,
            1,
            np.zeros((4, 4)),
            DummyModel(),
            dummy_val_and_grad,
            control_mask,
            dm_mask,
            pwp_params={},
            Nitr=1,
            delay=0.0,
        )
        self.assertEqual(len(efc_data["commands"]), 1)

    def test_shmim_utils_write_stack(self):
        class DummyStream:
            def __init__(self, shape):
                self.shape = shape
                self._latest = np.zeros(shape)
            def write(self, data):
                arr = np.array(data)
                if arr.shape[-2:] == self.shape:
                    self._latest = arr.reshape(self.shape)
                elif arr.shape == (self.shape[0], self.shape[1], 1):
                    self._latest = arr[:, :, 0]
                else:
                    self._latest = arr.squeeze()
            def grab_many(self, n):
                return np.repeat(self._latest[None, ...], n, axis=0)

        stream = DummyStream((2, 2))
        shmim_utils.write(stream, np.array([[1.0, 2.0], [3.0, 4.0]]))
        stacked = shmim_utils.stack(stream, 1)
        self.assertEqual(stacked.shape, (2, 2))
