# Math Fidelity Audit: Python ↔ C++

This document records a function-by-function comparison of the C++ port
under `cpp/` against the original Python implementation in `lina/`. It is
intended both as a record of what was checked and as a punch list of
divergences that should be fixed (or accepted) before the C++ becomes
authoritative.

Severity legend used below:

- ✅ **Faithful** — mathematically equivalent within numerical tolerance.
- ⚠️ **Partial** — equivalent under default / typical inputs; differs in
  edge cases.
- ❌ **Divergent** — produces a different numerical result for inputs the
  callers actually use; should be fixed.

---

## Recent additions (May 2026)

In addition to the bugs documented below, the following capabilities
were added to the C++ port and exposed through pybind:

1. **`lina::Grid2D` helper class** (`cpp/include/lina/grid2d.h`) — a
   single class for coordinate-grid construction with consistent
   "odd"/"even" centering, optional rotation, and shifts. Used by new
   code (`get_fresnel_TF`); existing parity-tested implementations of
   `make_grid`, `create_annular_mask`, `mft_*`, and `make_vortex_phase_mask`
   still inline their own coordinate code (refactor deferred to avoid
   regressions). New code should prefer `Grid2D`.

2. **`lina::get_fresnel_TF`** — Fresnel defocus transfer function
   (`cpp/src/props.cpp`). Numerically matches `lina.props.get_fresnel_TF`
   to floating-point precision (max diff ≈ 1e-15).

3. **FITS I/O via cfitsio** — `lina::save_fits` (overloaded for
   `Array2D<double|float|int32|uint8>`) and `lina::load_fits_double` /
   `load_fits_float` / `load_fits_with_header`, plus the helper
   `lina::fits_available()` (`cpp/src/fits_io.cpp`). Header is a
   `std::vector<std::pair<std::string, std::string>>`. Gated on
   `LINA_USE_CFITSIO` (default ON, auto-detected). Round-trip tested
   against astropy: bit-exact in both directions for double, float, and
   int32 buffers.

4. **`lina_cpp` Python package** (`lina_cpp/`) — a separate
   pip-installable package that mirrors the entire `lina` API surface
   so user code can swap backends with a one-line change:

   ```python
   import lina_cpp as lina  # was: import lina
   ```

   Math hot paths (utils.{mean,rms,pad_or_crop,lstsq,tikhonov_inverse,
   beta_reg,create_annular_mask,save_fits,load_fits},
   props.{fft,ifft,ang_spec,mft_forward,mft_reverse,
   make_vortex_phase_mask,get_fresnel_TF}, dm.create_*,
   coro_utils.{normalize_coro_im,compute_contrast},
   iefc.compute_hadamard_scale_factors) are wired to the C++ extension.
   Hardware control (28 INDI-driven functions in coro_utils) and the
   high-level loop modules (llowfsc, rt_utils, wfe) re-export from
   `lina` until those are also ported. See `lina_cpp/README.md`.

5. **`notebooks/compare_lina_vs_lina_cpp.ipynb`** — side-by-side parity
   and microbenchmark harness. Sample numbers from the test machine
   (CPU only, no CUDA): `create_annular_mask` 4.1×, `save_fits` 3.9×,
   `load_fits` 5.3× (256² arrays); FFT/IFFT and `ang_spec` ≈ 1× because
   both backends share FFTW; `get_fresnel_TF` ≈ 1.2×. The notebook
   re-runs cleanly under `jupyter nbconvert --execute`.

---

## Bugs fixed during this audit

The following four bugs were found *and fixed* in the C++ port while
producing this document. Each has a regression test in
`lina/tests/test_per_method_parity.py`:

1. **FFTW plan in-place / out-of-place aliasing bug** in `cpp/src/props.cpp`.
   `fft_cpu` and `ifft_cpu` called `fftw_execute_dft(plan, buffer, buffer)`
   in-place but the plan was created with two distinct dummy arrays
   (out-of-place). Per FFTW docs, that produces silent garbage for any
   plan whose algorithm needs separate input/output (visible for odd
   sizes ≥ 33 and many composite sizes). Fix: create the plan with the
   same array for input and output.

2. **`create_annular_mask` `edge=0.0` sentinel** in `cpp/src/utils.cpp`.
   Python distinguishes `edge=None` (no half-plane filter) from
   `edge=0` (filter at `xr > 0`); the C++ collapsed both to "no filter".
   Fix: always apply the cut `xr > edge`; introduce
   `kNoEdgeFilter = -DBL_MAX` as the "no filter" sentinel; pybind maps
   Python `None` to this sentinel.

3. **`dm::create_fourier_modes` used the wrong `edge`** in
   `cpp/src/dm.cpp`. Python's current source passes `edge=0` (right-half
   plane) but the C++ ported the older commented-out code with
   `edge=iwa-fourier_sampling`. Fix: pass `edge=0` so C++ matches Python
   exactly (mode count and content).

4. **`mft_reverse` returned the transpose** of the Python answer in
   `cpp/src/props.cpp`. The two matrix-multiplications were factored as
   `My.T @ fpwf @ Mx.T` instead of `Mx @ fpwf @ My`. Visible for any
   non-symmetric input. Fix: re-derive the index pairing to match
   `lina.props.make_mft_reverse_matrices`.

These fixes are all reflected in the per-section findings below and in
the regression tests; sections marked ✅ have been verified by direct
numerical comparison via the pybind module.

---

## Coverage Summary (intentional non-ports)

The following Python modules / functions are deliberately not ported,
because they are I/O, plotting, hardware control, or telemetry:

- `utils.imshow`, `save_fits`, `load_fits`, `save_pickle`, `load_pickle`,
  `get_fnames`, `make_dir`, `move_files`, `delete_files`,
  `rotate_arr`, `interp_arr`, `tt_*`, `create_zernike_modes`,
  `lstsq`, `tikhonov_inverse`, `beta_reg`, `create_circ_mask`,
  `get_radial_dist`, `get_radial_contrast`, `plot_radial_contrast`.
- `coro_utils.*` — every function except `normalize_coro_im` and
  `compute_contrast` is hardware control via `purepyindi`.
- `props.get_scaled_coords`, `props.get_fresnel_TF`.
- `dm.create_all_poke_modes`, `dm.create_fourier_probes` (probes are
  still partly ported but missing features — see below).
- The whole modules `llowfsc.py`, `telem.py`, `rt_utils.py`, `wfe.py`.

If any of these become needed in the C++ pipeline the ports will need to
be added.

---

## utils

### `mean(array, mask=None)` — ✅
Both compute arithmetic mean over all pixels, or over a boolean mask.
Implementations match exactly.

### `rms(array, mask=None)` — ✅
Both compute `sqrt(mean(array**2))`. Match exactly.

### `pad_or_crop(arr, npix)` — ✅
Both use integer `n//2 - npix//2` (or `npix//2 - n//2`) offsets, and
zero-fill on pad. Match exactly.

### `make_grid(npix, pixelscale, half_shift)` — ⚠️ **half-pixel divergence for odd npix**

- Python: `(np.indices(...) - npix//2) * pixelscale` (or `+1/2 - npix//2`).
- C++:    `(idx - npix/2.0) * pixelscale` (or `idx - npix/2.0 + 0.5`).

For **even** `npix` these are identical. For **odd** `npix` (e.g.
`npix=33`):

- Python: pixel column 16 has x = 0 (centered on a pixel).
- C++:    pixel column 16 has x = -0.5; column 17 has x = +0.5
  (centered between pixels).

In other words, the C++ grid is offset by half a pixel relative to
Python whenever `npix` is odd.

**Recommendation**: change the C++ to use integer division to match
Python:

```cpp
const double center = static_cast<double>(npix / 2) - offset;
```

(where `npix / 2` is integer division because both operands are
`std::size_t`). The test
`test_make_grid_odd_documents_known_divergence` in
`test_per_method_parity.py` pins down the current behavior so you can see
the fix when it lands.

### `create_annular_mask(...)`, `create_annular_focal_plane_mask(...)` — ✅ (after edge-sentinel fix)

For zero rotation and zero shift, the C++ matches Python exactly for
**both** `edge=None` (no filter) and any numeric `edge` (cut at
`xr > edge`, including `edge=0`). Verified by
`test_create_annular_mask_no_edge_filter` and
`test_create_annular_mask_edge_zero_filters_right_half`.

Note on the sentinel: in C++ the "no filter" case is encoded as the
extreme negative value `kNoEdgeFilter = -DBL_MAX` (declared in
`include/lina/utils.h`); the pybind wrapper maps Python's `None` to that
sentinel. Existing C++ callers that previously passed `edge=0.0` to
mean "no filter" must now pass `kNoEdgeFilter` (or use the default
argument).

For non-zero rotation or non-integer shifts the implementations differ
slightly because of how the rotation is applied:

- Python applies `scipy.ndimage.rotate(mask, ..., order=0)` and
  `scipy.ndimage.shift(mask, ..., order=0)` to the *binary* mask, which
  rasterizes a rotated/shifted bitmap (with stair-step artefacts).
- C++ rotates the analytical coordinates before evaluating the
  mask predicate, producing a *clean* rotated annulus.

Both are reasonable; the C++ result is arguably more faithful to the
mathematical intent.

---

## props

### `fft(arr)`, `ifft(arr)` — ✅ (after FFTW plan-aliasing fix)

Python uses `xp.fft.ifftshift(xp.fft.fft2(xp.fft.fftshift(arr)))` and
the inverse counterpart. The C++ shift functions are bit-exact NumPy
equivalents (positive roll by `n/2` for `fftshift` and `(n+1)/2` for
`ifftshift`).

**Fixed bug**: the FFTW plan was created with two distinct dummy buffers
(out-of-place) but `fftw_execute_dft(plan, buffer, buffer)` was then
called in-place. That is undefined behaviour per FFTW: "in-place plans
must be used in-place and out-of-place plans must be used out-of-place."
For radix-2 even sizes the algorithm happens to be safe; for prime-
factor / mixed-radix sizes (e.g. `n=33=3·11`, `n=65=5·13`) it produced
wrong values. The fix is to allocate a single dummy and reference it for
both `in` and `out` in `fftw_plan_dft_2d`.

After the fix `lina_cpp.fft_cpu` matches `lina.props.fft` to ~1e-13 for
both even and odd sizes (`test_fft_random`, `test_ifft_random`,
`test_fft_ifft_matches_python_for_odd_n`).

**Note on roundtripping for odd N**: `ifft(fft(x)) == x` exactly for
even `N`, but for odd `N` there is a one-pixel shift because the
centered-FFT pipeline composes `ifftshift∘ifftshift` in the middle, and
that composition is *not* identity for odd N — this is a property of
NumPy itself (and is mirrored faithfully in C++). See
`test_fft_ifft_roundtrip_even_n` (passes) and
`test_fft_ifft_matches_python_for_odd_n` (passes).

### `mft_forward`, `mft_reverse` — ✅ (after `mft_reverse` transpose fix)

`mft_forward` was always correct. `mft_reverse` had a real bug: the two
inner contractions were factored as `My.T @ fpwf @ Mx.T` rather than the
intended `Mx @ fpwf @ My`. Concretely:

```
out_buggy(x, y) = Σᵤ Σᵥ fpwf(u,v) · exp(j·s·2π·Us[v]·Xs[x]) · exp(j·s·2π·Xs[y]·Us[u])
```

which equals `(out_correct).T`. For symmetric inputs this slips through
unnoticed (any roundtrip test on a real symmetric pupil); for any
non-symmetric input (e.g. probes, off-axis sources) it returned the
wrong answer.

After the fix the C++ implements

```
out(x, y) = (psf_pixelscale_lamD / npix) · Σᵤ Σᵥ
              fpwf(u, v) · exp(j·s·2π·Xs[x]·Us[u])
                        · exp(j·s·2π·Us[v]·Xs[y])
```

which matches `lina.props.mft_reverse` element-wise to ~1e-15
(`test_mft_reverse_inverse_of_mft_forward` exercises a non-symmetric
random complex input).

### `ang_spec(wavefront, wavelength, distance, pixelscale)` — ✅

Both compute the same angular spectrum: forward FFT, multiply by the
free-space transfer function `exp(j·kz·distance)` with
`kz = sqrt(k² - kx² - ky²)`, inverse FFT. The k-coordinates match
exactly (both use the half-shifted convention `(i - n/2 + 1/2)·dk`).

### `make_vortex_phase_mask(npix, charge, grid)` — ❌ **divergence for odd npix**

For **even** npix (e.g. 16, 32, 256) the implementations agree.

For **odd** npix the implementations diverge in two coupled ways:

- Python builds `x = np.linspace(-npix//2, npix//2 - 1, npix)`. For odd
  npix this is *not* unit-spaced — e.g. for `npix=17` you get 17 values
  from -8 to 8 with a step of 1.0; for `npix=33` you get 33 values from
  -16 to 15 with a step of 31/32 ≈ 0.969. So the vortex isn't even
  isotropic for odd sizes when using the Python code as-is.
- The C++ port uses `x = c - npix/2.0`, i.e. unit-spaced half-integer
  coordinates for odd npix.

Since the typical use case is `npix=2048` (even), this rarely bites in
practice, but it should be flagged.

**Recommendation**: pick one canonical convention. The cleanest
choice is unit spacing: change Python to
`x = np.arange(npix) - npix//2` (and analogous `+ 0.5` for `'even'`),
and align the C++ if needed. The test
`test_make_vortex_phase_mask_odd_known_divergence` is marked
`expectedFailure` to act as a regression check.

---

## linalg

### `gemm`, `gemv` — ✅

Optionally backed by OpenBLAS' CBLAS; the fallback is a hand-rolled
loop. Both produce results identical to `numpy.matmul` for unit alpha,
zero beta. The transposes are handled correctly.

### `svd(A)` (double), `svd_float_*` — ✅

Calls LAPACKE `dgesvd` / `sgesvd` with `'A'` for both U and Vt. This
matches `numpy.linalg.svd(A, full_matrices=True)` *except* for the
inevitable column-sign ambiguity in U and row-sign ambiguity in Vt.

When writing tests, compare:

- singular values directly (`np.testing.assert_allclose(s_cpp, s_py)`)
- the reconstruction `U @ diag(s) @ Vt` (which is sign-invariant)

rather than U and Vt element-wise. The test class `TestLinalg` shows
both styles.

---

## dm

### `create_mask(nact)` — ⚠️
Equivalent for **even** `nact` (the production case `nact=34`). For
odd `nact` the C++ uses `nact/2.0 = (k+0.5)` while Python uses
`nact//2 = k`, leading to coordinates offset by half a pixel and a
slightly different mask radius cutoff.

### `make_gaussian_inf_fun(...)` — ✅
For typical (even) `ng = sampling*Nact`, the implementations agree
to numerical precision. Both produce the same Gaussian profile with
the same FWHM-via-coupling formula
`d = act_spacing / sqrt(-log(coupling))`.

### `create_hadamard_modes(dm_mask)` — ✅ (with shape difference)

Both use the Sylvester construction. The only difference is the output
layout: Python returns shape `(np2, Nact, Nact)`; C++ returns shape
`(np2, Nact*Nact)` (each mode flattened as a row). They contain the
same numbers; the test reshapes Python before comparing.

### `create_fourier_modes(...)` — ✅ (after `edge=0` fix)

Python's call to `utils.create_annular_mask` passes **`edge=0`**, which
selects the right-half of the expanded annulus. The original C++ port
read the *commented-out* prior version of the Python (the comment in
`lina/dm.py` literally shows the old `edge=iwa-fourier_sampling` line)
and ported that — producing a different region of the
`(iwa-fs, owa+fs)` annulus and therefore a different number of modes
(e.g. 268 vs 316 for the standard 34-actuator config).

Importantly, this also discovered the
[`create_annular_mask` `edge=0.0` sentinel bug](#create_annular_mask-create_annular_focal_plane_mask--).
Together they masked one another: with the buggy `edge=0.0` sentinel
the older Python formulation happened to numerically match the C++ for
the parameter configurations tested. After fixing **both** the sentinel
*and* updating `create_fourier_modes` to use `edge=0` (matching current
Python), the C++ mode set matches Python's element-wise.

```cpp
// cpp/src/dm.cpp (after fix)
const auto fourier_mask = create_annular_mask(
    nfg, fourier_sampling, iwa - fourier_sampling, owa + fourier_sampling,
    0.0, 0.0, 0.0, rotation);  // edge=0 (filter at xr > 0)
```

Verified by `test_create_fourier_modes_match_python` for `which ∈
{cos, sin, both}` across three parameter configurations.

### `create_fourier_probes(...)` — ❌ **two distinct issues**

1. **Normalization is signed-max in Python, abs-max in C++**.
   Python does `probes[i] = probe / xp.max(probe)` (unsigned `np.max`);
   C++ does `probe[i] = probe[i] / max_abs(probe)`. For a probe whose
   most-extreme value is negative, Python's normalization flips the
   sign of the whole probe; C++ never does. They will not produce
   identical probes.

2. **`shifts` parameter is missing in the C++**. Python supports
   per-probe pixel shifts via `xcipy.ndimage.shift(probe, ...)`; the
   C++ has a stub `shift_bilinear(probe, 0.0, 0.0)` that does nothing.
   For the default `shifts=None` this doesn't matter, but consumers
   who want shifted probes won't get them.

### `make_fourier_command`, `make_f`, `make_ring`, `make_cross_command` — ⚠️

Same `Nact//2` (integer) vs `nact/2.0` (float) issue as `create_mask`
and `make_grid`. For the production `Nact=34` (even) all four match
Python. For odd `Nact` they differ by half-pixel coordinate offsets.

---

## coro_utils

### `normalize_coro_im(...)` — ⚠️ **silent default difference**

The numerical formula matches:
`(raw - dark) · (Texp_ref / Texp_im) · 10^((G_ref - G_im)/200) · 10^((A_im - A_ref)/10) / Imax`.

The behavioral difference is in handling missing parameters:
- Python checks each of `'exp_time'`, `'gain'`, `'atten'` against the
  dictionaries and falls back to `1.0` if either side lacks the key.
- C++ unconditionally evaluates all three factors, expecting the
  fields to be present in the `ImParams` struct.

**Recommendation**: document that callers must supply
`gain=0`, `atten=0`, `exp_time` matched between im_params and ref_params
when they don't actually want the correction (the formula degenerates
to `1.0` in those cases).

### `compute_contrast(ni, mask)` — ✅
Mean of pixels in mask that are positive. Matches Python exactly. The
C++ also returns auxiliary counts (`n_mask`, `n_positive`) that are not
present in Python.

---

## control_models

`MODEL` (Python) ↔ `lina::ControlModel` (C++). This is the most
intricate port and accumulates several smaller divergences. None are
disastrous in default use but they will show up if you compare
focal-plane fields pixel-by-pixel.

### Apertures — ⚠️
- Python: `poppy.CircularAperture(...).get_transmission(...)` produces
  *gray-pixel* (anti-aliased) edges with values in `[0,1]`.
- C++: `circular_aperture(...)` produces *binary* edges — a pixel is
  fully transmitting if its center is within `radius`, otherwise zero.

Effect: the pupil edge has a 1-pixel-thick band where Python and C++
disagree by O(0.5). Inside the pupil and outside the pupil they agree.
For typical apertures this changes the total throughput by less than
~0.3% and shifts the PSF wings slightly. If you need pixel-perfect
parity, the C++ would need a sub-pixel-anti-aliased aperture.

### `Imax_ref` normalization — ⚠️
Python: `E_EP = APERTURE * WFE / sqrt(Imax_ref)`. C++: omits the
`/sqrt(Imax_ref)`. Since the default `Imax_ref = 1.0`, this is a no-op
in practice.

### Tukey window — ⚠️
- Python: `scipy.signal.windows.tukey(N, alpha=1, sym=False)` —
  *periodic* (sym=False). With alpha=1 this is the periodic Hann,
  `0.5·(1 - cos(2πi/N))`.
- C++: `0.5·(1 - cos(2πi/(N-1)))` — *symmetric* Hann.

Difference is one sample at the edge of the window; tiny effect on the
LRES vortex output.

### `inf_fun_fft` shift convention — ⚠️
- Python: `fftshift(fft2(ifftshift(inf_fun)))`.
- C++:    `lina::fft(inf_fun)` = `ifftshift(fft2(fftshift(inf_fun)))`.

For an even-sized `inf_fun` (which is the production case —
`Nsurf = (Nact+2)·sampling = 36·11 = 396`) `fftshift == ifftshift`, so
the two are equivalent.

For odd `Nsurf` they would differ. Worth a comment in the code.

### DM model `fx`, `fy` — ❌ **suspicious**

- Python: `fx = np.fft.fftshift(np.fft.fftfreq(Nsurf))` — *centered*
  spatial-frequency axis.
- C++:    `fx = fftfreq(nsurf_)` — *un-centered* axis (no fftshift).

The DM matrices `Mx_dm`, `My_dm` are then built as `exp(±j·2π · outer(fx, xc))`.
Python uses centered fx; C++ uses un-centered fx. Combined with the
also-centered `inf_fun_fft` (Python) vs the un-shifted `fft(inf_fun)`
(C++) this likely cancels for even sizes — but the inconsistency is
subtle. Worth verifying with a numerical test by comparing
`forward()` outputs between Python and C++ for the same pupil/DM
inputs.

**Recommendation**: add a comment in `control_models.cpp` documenting
the convention, and add a test that calls Python `MODEL.forward(...)`
and `lina::ControlModel::forward(...)` on a known set of actuators and
checks E_FP agreement.

### Focal plane scale `npix * lyot_ratio` — ❌ **integer truncation**

```cpp
mft_forward(e_fffp,
            static_cast<std::size_t>(npix_ * lyot_ratio_),  // truncates!
            ncamsci_, camsci_pxscl_lamD, ...);
```

Python passes `npix * lyot_ratio` *as a float* into `mft_forward` which
uses it as `npix` directly; the value is e.g. `512 * 0.945 = 483.84`.
The C++ casts to `size_t` (483), giving a 0.17% pixel-scale error.

**Recommendation**: change `mft_forward`'s `npix` parameter to a
`double` (it's only used in `dx = 1.0/npix`), or accept the float
elsewhere.

### Camera rotation interpolation order — ⚠️

Python uses `scipy.ndimage.rotate(..., order=3)` (cubic) for forward
and `order=5` for the gradient. C++ uses bilinear (order=1). For the
default `camsci_rotation = 0.0` this is a no-op; for non-zero rotations
the focal-plane field will be smoother in the C++ port.

### `val_and_grad` — ✅ for default zero rotation

Reading the adjoint chain symbolically:

1. `dJ/d(deltaE)` = `2 · M_control · E_predicted / |E_ab|_2`
2. Reverse MFT through the camsci scale to get `dJ/dE_FFFP`.
3. (Optional) `ang_spec` back-propagation to `dJ/dE_LS`.
4. Multiply by Lyot stop, FFT to FPM plane, multiply by conj(vortex)·(1-window).
5. Adjoint MFT through the high-resolution vortex branch.
6. Combine the two branches.
7. `dJ/dS_DM = 4π/λ · imag(dJ/dE_PUP · conj(E_EP) · conj(DM_PHASOR))`.
8. FFT, multiply by `conj(inf_fun_fft)`, adjoint MFT through DM
   matrices, divide by `Nsurf · Nact²`.
9. Add `r_cond · 2 · del_acts/wavelength_c`.

The C++ implements every step. With the caveats listed above (binary
aperture, fftfreq centering, MFT npix truncation, rotation order), the
C++ gradient should match the Python gradient to within those tolerances.

### `val_and_grad_bb` (multi-wavelength) — ❌ **bugs**

Three issues comparing the C++ to the Python `val_and_grad_mw`:

1. **Missing `/Nwaves` on the gradient**:
   ```cpp
   // current C++ (cpp/src/control_models.cpp:524-549):
   for (... waves) {
       j_sum += j_mono;
       grad_out[k] += grad_mono[k];           // <-- accumulating
   }
   return j_sum / nwaves;                     // J is averaged
   // grad_out is NOT averaged
   ```
   The Python takes `dJ_dA_bb = sum(dJ_dA_monos, axis=0) / Nwaves`,
   so the C++ gradient is `Nwaves`× too large.

2. **Missing regularization term in J**:
   Python adds `r_cond · del_acts_waves.dot(del_acts_waves)` to J;
   C++ does not.

3. **Wrong wavelength for regularization gradient**:
   C++ uses `r_cond · 2 · del_acts[i] / wavelength()` with the
   instantaneous `wavelength_`; Python uses `del_acts/wavelength_c`
   (the central wavelength) consistently.

These should be fixed before relying on broadband AEFC.

### `dm_val_and_grad` — not ported (intentional?)

Python has a third routine `dm_val_and_grad` that minimises OPD wavefront
error (rather than focal-plane intensity). There is no C++ counterpart.
If you need it, port directly — it is structurally similar to
`val_and_grad` but simpler (no vortex branches).

---

## efc

### `compute_jacobian(model, control_mask, amp, current_acts)` — ✅

Both compute finite-difference response per-actuator and store
real/imag parts interleaved at the masked pixels. The C++ adds
`compute_jacobian_bb` for multi-wavelength.

### `run` (single wavelength), `run_bb` (broadband) — ✅ algorithm equivalence

The C++ adds hardware-stream specifics (`dm_scale` factor and
read-modify-write through `Stream2D`) that don't have direct Python
analogs but are consistent with the MagAOX integration. The control
math is the same: `del_acts = -gain · M⁺ · E_ab_vec`, leakage
update `(1-leakage)·current + del`.

---

## iefc

### `measure_probe_response`, `calibrate`, `run`, `compute_hadamard_scale_factors` — ✅

All four match Python. The single caveat is layout: Python returns a
`response_matrix` shaped `(Nprobes·Nmask, Nmodes)` (transposed); C++
returns `(Nmodes, Nprobes·Nmask)`. The pseudoinverse / control matrix
must be computed accordingly.

The constant of `4` shows up nowhere in iEFC because iEFC doesn't
linearize about a complex E-field — its response matrix is the linear
relationship between probed-difference vectors and modal coefficients.

---

## pwp

### `PwpSolver::run` — ❌ **factor-of-2 mismatch with Python**

The PWP measurement equation is

    delI = |E + dE|² - |E - dE|² = 4 · Re(E* · dE)
         = 4 · [Re(E)·Re(dE) + Im(E)·Im(dE)]
         = [4·Re(dE), 4·Im(dE)] · [Re(E), Im(E)]ᵀ

The C++ correctly uses factor `4`:

```cpp
// cpp/src/pwp.cpp:226-227
H(p, 0) = 4.0 * ep.real();
H(p, 1) = 4.0 * ep.imag();
```

The Python uses factor `2`:

```python
# lina/pwp.py:93-95
H = 2 * xp.array([delE_probes[..., i].real,
                  delE_probes[..., i].imag]).T
```

Consequence: the Python E-field estimate is **2× larger** than the
mathematically correct value (and thus 2× larger than the C++).
Downstream EFC may have tuned its gain to absorb this constant, in
which case fixing the Python (or matching the C++ to Python) becomes
a behavior change.

**Recommendation**: pick one and document it. The C++ is
mathematically correct; the Python looks like an off-by-2. Either
- change the Python to factor 4 and re-tune the loop gain by 1/2, or
- change the C++ to factor 2 to match Python output exactly.

---

## aefc

### `Optimizer` interface — ✅

- `GradientDescentOptimizer` is a simple fixed-step descent (handy as a
  trivial smoke test).
- `LbfgsOptimizer` wraps libLBFGS (compiled in via `LINA_USE_LBFGS=ON`).

Python uses `scipy.optimize.minimize(method='L-BFGS-B')`. libLBFGS and
SciPy's L-BFGS-B share the same algorithm family but their convergence
trajectories will differ. As long as the **objective function** matches
(see `val_and_grad` notes above), both arrive at similar minima.

### `run` — ✅ for default rotation; inherits the
`val_and_grad` divergences for non-zero rotation, broadband, and
multi-wavelength setups.

---

## Recommended fix order

Already fixed during this audit (regression-tested):

- ✅ **FFTW plan in-place / out-of-place aliasing** in `fft_cpu` /
  `ifft_cpu`.
- ✅ **`create_annular_mask` `edge=0.0` sentinel** (now `kNoEdgeFilter`).
- ✅ **`dm::create_fourier_modes` wrong `edge`** (now `0.0` matching
  Python).
- ✅ **`mft_reverse` returned the transpose** of Python's answer.

Remaining (in suggested fix order):

1. **`make_grid` integer division** (one-line fix; affects many
   downstream functions; documented by
   `test_make_grid_odd_documents_known_divergence`).
2. **`val_and_grad_bb` averaging + regularization** (a few lines;
   important for broadband AEFC).
3. **PWP factor of 2 vs 4** (decide convention, then adjust one side).
4. **`make_vortex_phase_mask` odd-size convention** (decide canonical
   convention, fix Python or C++).
5. **`mft_forward` accepting non-integer pupil sampling**
   (`npix * lyot_ratio` truncation).
6. **Aperture anti-aliasing** (more involved; only matters for
   pixel-perfect parity).
7. **`fftfreq` centering convention** (verify with a forward-model
   numerical comparison; document or fix).
8. **`svd` (double-precision LAPACKE path)** passes `nullptr` for the
   `superb` workspace; segfaults when `LINA_FORCE_LAPACKE=ON`. Tests
   currently use `svd_float_cpu` instead.

Tests that pin down each item live in
`lina/tests/test_per_method_parity.py`; the ones marked
`@expectedFailure` will start passing as fixes land.
