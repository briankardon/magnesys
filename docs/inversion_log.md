# Inversion Development Log

## Setup

**Coil arrangement:** 3-axis unbalanced anti-Helmholtz pairs at coprime frequencies (100, 137, 173 Hz). Each pair has asymmetric currents (e.g. +1.0A / -0.7A) to break mirror symmetry.

**Inversion pipeline:**
1. Lock-in demodulation (multiply by cos at each frequency, average over window)
2. Coarse search via KD-tree on precomputed field table
3. Fine refinement via `scipy.optimize.least_squares`

All results are noiseless unless otherwise noted.

> **Note:** The parameters above describe the early-experiment baseline
> (Exps. 1-13). Later experiments evolve the carriers (up to 1 kHz
> triple), the cage scale (1 m cube), the noise model (`SensorProfile`),
> the trajectory (closed Lissajous), and the inversion (multi-start
> 6-DOF, dual-sensor). Each experiment's section documents its own
> active setup and assumptions.

---

## Experiments 1–5: No probe rotation (3-DOF position only)

All experiments in this section assume the sensor is in a fixed orientation
(no rotation). The only unknowns are x, y, z position.

### Experiment 1: Initial 3-DOF inversion (small scale)

**Scale:** 12cm tracking volume, coils at ±7.5cm
**Grid resolution:** 25³
**Window:** 1 period of lowest frequency (10ms)

| Metric | Error |
|--------|-------|
| Position error (median) | 32 mm |
| Position error (max) | 52 mm |

**Problem identified:** Each coil treated as a separate source (6 entries in field table), but demodulator can only separate by frequency — two coils sharing a frequency get summed. Demodulated values didn't match the per-source field table entries.

---

### Experiment 2: Frequency grouping fix

**Change:** Group sources by frequency in the field table. Two coils at 100 Hz → one combined channel. 6 sources → 3 frequency channels.

**Scale:** 12cm, resolution 30³, 1-period window

| Metric | Result |
|--------|--------|
| Stationary point error | 9.2 mm |
| Trace position median | 24.5 mm |

**Improvement:** Dramatic — demodulation now matches the field table. But still mediocre accuracy.

---

### Experiment 3: Window length sweep (stationary point)

**Test:** Fixed sensor position, varying demodulation window length.

| Window (periods) | Demod error (max) | Position error (from true) |
|-------------------|-------------------|---------------------------|
| 1 | 7.5e-7 | 9.2 mm |
| 2 | 4.4e-7 | 4.6 mm |
| 5 | 1.3e-7 | 1.5 mm |
| 10 | 9.2e-8 | 0.9 mm |

**Finding:** Demodulation accuracy improves monotonically with window length. Cross-talk between coprime frequencies (100/137/173 Hz) needs multiple periods to average out.

---

### Experiment 4: Moving trace with 3-period windows

**Scale:** 12cm, resolution 30³, speed 5 cm/s

| Window (periods) | Estimates | Position error (median) | Position error (max) |
|-------------------|-----------|------------------------|---------------------|
| 1 | 199 | 22.1 mm | 45.8 mm |
| 3 | 65 | **3.3 mm** | 7.6 mm |
| 5 | 39 | 3.1 mm | 7.0 mm |

**Finding:** 3-period windows achieve sub-centimeter accuracy. The sweet spot — longer windows give diminishing returns but fewer estimates.

---

### Experiment 5: Current strength scaling (small scale)

**Test:** Vary current magnitude from 1A to 10A.

| Current | 3-DOF pos. error (median) | 6-DOF pos. error (median) |
|---------|--------------------------|--------------------------|
| 1.0 A | 13.1 mm | 26.2 mm |
| 5.0 A | 13.1 mm | 26.2 mm |
| 10.0 A | 13.1 mm | 26.2 mm |

**Finding:** Current strength has zero effect in the noiseless case. The field scales linearly and the inversion is scale-invariant. Current will matter only when sensor noise is introduced (SNR).

Note: The 6-DOF column in this table was tested at cage scale (Experiment 8) with rotation, included here for the current-scaling comparison only.

---

## Experiments 6–7: With probe rotation (6-DOF / 4-DOF)

Starting from Experiment 6, synthetic **sensor rotation** is applied to the
magnetometer signal. The sensor orientation varies smoothly along the path
(base orientation along path tangent + random perturbations up to ±30°).
This means the measured field is rotated into the sensor's local frame,
and the inversion must recover both position (x, y, z) and orientation
(roll, pitch, yaw).

### Experiment 6: 6-DOF inversion (position + orientation)

**First attempt:** Used same KD-tree coarse search as 3-DOF. Failed badly (98mm position, 145° orientation) because rotation changes the field direction, making the direction-aware KD-tree useless.

**Fixes applied:**
1. Rotation-invariant coarse search using per-channel field **magnitudes** (invariant under rotation)
2. SVD-based initial rotation estimate (Wahba's problem) from the coarse position

| Metric | Error (before fixes) | Error (after fixes) |
|--------|---------------------|---------------------|
| Position error (median) | 98 mm | **9.4 mm** |
| Orientation error (median) | 145° | **31.8°** |

**Scale:** 12cm, 30° max perturbation, 3-period windows

---

### Experiment 7: 4-DOF + IMU (position + yaw, tilt from accelerometer)

**Idea:** Use accelerometer to fix roll/pitch (tilt from gravity), only optimize position + yaw (4 unknowns).

**First attempt:** Roll/pitch extracted via Euler angle formula — wrong convention, didn't match the rotation composition. Result: 96mm error.

**Second attempt:** Tilt extracted geometrically (find rotation mapping gravity vector). Full rotation = R_yaw × R_tilt.

| Mode | Position error (median) | Orientation error (median) |
|------|------------------------|---------------------------|
| 6-DOF (no IMU) | **9.4 mm** | **31.8°** |
| 4-DOF + ideal IMU | 29.0 mm | 106° |

**Finding:** IMU constraint made it *worse* in the noiseless case. Constraining tilt and only solving for yaw makes the residual landscape more complex — the optimizer gets stuck. The full 6-DOF solver has more freedom to find the global minimum. IMU expected to help when noise limits the 6-DOF solver.

---

---

## Experiments 8–10: Cage scale (0.5m cube)

### Experiment 8: Cage scale baseline

**Setup:** Coils at ±30cm, diameter 70cm, tracking volume ±20cm.

| Mode | Position error (median) | Position error (max) |
|------|------------------------|---------------------|
| 3-DOF, 3-period, 50% overlap | 13.1 mm | 24.1 mm |
| 6-DOF, 3-period | 26.2 mm | 125 mm |

**Finding:** Error roughly doubled going from 12cm to 50cm scale. Weaker gradients over the larger volume mean less spatial discrimination per unit of field measurement.

---

### Experiment 9: Coil geometry sweep (cage scale)

**Test:** Different asymmetric configurations, same coil frequencies.

| Config | Pos. error (median) | Pos. error (max) | Orient. error (median) |
|--------|---------------------|------------------|----------------------|
| Symmetric | 26.2 mm | 125 mm | 29.0° |
| Asym diameters only | 451 mm | 508 mm | 160° (broken) |
| Asym diameters + currents | 27.0 mm | 130 mm | 30.7° |
| Asym everything + offsets | 35.4 mm | 314 mm | 35.4° |
| Asym + tilted Y-axis | **25.8 mm** | 145 mm | **29.1°** |

**Findings:**
- Asymmetric diameters alone broke the magnitude-based coarse search
- Symmetric and tilted configs were essentially tied
- Breaking geometric symmetry didn't help as much as expected — coprime frequencies already provide good channel separation

---

### Experiment 10: Multipass inversion

**Idea (from user):** Use first-pass trajectory to get velocity hints, improve second pass.

**Attempt 1 — shorter windows on second pass:**

| Approach | Estimates | Pos. error (median) |
|----------|-----------|---------------------|
| Single pass, 3-period | 132 | 13.1 mm |
| Multipass, 3p → 1p | 399 | 78.7 mm |
| Multipass, 3p → 0.5p | 832 | 88.2 mm |

**Failed:** Short windows have too much frequency cross-talk regardless of initialization quality.

**Attempt 2 — same window size, higher overlap:**

| Approach | Estimates | Pos. error (median) | Pos. error (max) |
|----------|-----------|---------------------|------------------|
| Single, 3p, 50% overlap | 132 | 13.1 mm | 24.1 mm |
| Single, 3p, 75% overlap | 267 | 10.8 mm | 24.7 mm |
| Single, 3p, 90% overlap | 704 | 10.8 mm | 24.1 mm |
| Multipass, 3p, 90% overlap | 657 | 10.9 mm | 24.1 mm |

**Finding:** Higher overlap gives more estimates at slightly better accuracy. Multipass provides no accuracy improvement over single pass with the same overlap — the demodulation window is the bottleneck, not initialization.

---

### Experiment 11: Excitation frequency sweep (cage scale)

**Test:** Vary excitation frequencies from 100 Hz to 1 kHz, keeping 3-period windows.
Sampling rate scaled to 10× highest frequency. Cage scale (0.5m), 30° rotation.

| Frequencies | Window | 3-DOF pos. error (median) | 3-DOF pts | 6-DOF pos. error (median) | 6-DOF orient. error |
|-------------|--------|--------------------------|-----------|--------------------------|---------------------|
| 100/137/173 Hz | 30.0 ms | **13.1 mm** | 132 | **26.2 mm** | **29.0°** |
| 300/411/519 Hz | 10.0 ms | **13.0 mm** | 399 | 459 mm | 158° (broken) |
| 500/687/873 Hz | 6.0 ms | 14.9 mm | 665 | 460 mm | 159° (broken) |
| 1000/1373/1747 Hz | 3.0 ms | 15.1 mm | 1332 | 462 mm | 158° (broken) |

**Findings:**
1. **3-DOF accuracy is flat at ~13mm regardless of frequency.** The accuracy floor is set by field geometry and grid resolution, not demodulation window duration. This disproves the earlier "motion blur" hypothesis.
2. **6-DOF completely fails at 300+ Hz.** The 100 Hz result (26mm) doesn't generalize to higher frequencies. Likely cause: the magnitude-based coarse search and SVD rotation initialization break down with the different sample counts / rotation sequences at higher rates. Needs investigation.
3. **Higher frequencies give proportionally more position estimates** (132 → 1332) at the same accuracy for 3-DOF — useful for temporal resolution.

---

### Experiment 12: Orientation-first 6-DOF (cage scale, frequency sweep)

**Key insight (from user):** Demodulated field directions approximate the
rotated basis vectors — you can estimate orientation *before* knowing
position. This inverts the order of operations vs. the previous approach.

**New pipeline:**
1. Estimate rotation from field directions alone (SVD on unit vectors vs cardinal axes)
2. Un-rotate measurements into approximate lab frame
3. Coarse 3-DOF position search on un-rotated measurements (direction-aware KD-tree)
4. Refine rotation using coarse position (SVD with actual field at that position)
5. Joint 6-DOF refinement from this good starting point

| Frequencies | Window | Pos. error (median) | Pos. error (max) | Orient. error (median) | Orient. error (max) |
|-------------|--------|---------------------|------------------|----------------------|---------------------|
| 100/137/173 Hz | 30.0 ms | 26.2 mm | 57.9 mm | 29.0° | 60.2° |
| 300/411/519 Hz | 10.0 ms | **23.2 mm** | 91.0 mm | **28.6°** | 63.8° |
| 500/687/873 Hz | 6.0 ms | 26.2 mm | 90.0 mm | 29.0° | 60.4° |

**Findings:**
1. **Fixed the 6-DOF failure at higher frequencies.** Previous approach gave 460mm at 300+ Hz; orientation-first gives 23-26mm across all frequencies tested.
2. **Position accuracy roughly constant at ~25mm median** regardless of frequency — confirms the floor is geometric, not temporal.
3. **Max errors (90mm) occur at cage edges** where field directions deviate most from cardinal axes, degrading the direction-based rotation estimate.
4. **Orientation accuracy ~29° median** — consistent across frequencies.

---

### Experiment 13: Fixed window duration, varying frequency (cage scale)

**Key insight (from user):** Previous frequency sweep (Exp. 11) kept
period count constant (3 periods), so higher frequencies got shorter
windows with the same number of cycles — no improvement.
This test keeps the **window duration fixed at 30ms** so higher
frequencies get more cycles and thus cleaner demodulation.

| Frequencies | Periods in 30ms | 3-DOF pos. error (median) | 3-DOF pos. error (max) | 6-DOF pos. error (median) | Orient. error (median) |
|-------------|----------------|--------------------------|----------------------|--------------------------|----------------------|
| 100/137/173 Hz | 3.0 | 13.1 mm | 24.1 mm | 26.2 mm | 29.0° |
| 300/411/519 Hz | 9.0 | 8.4 mm | 16.8 mm | 15.2 mm | 27.5° |
| 500/687/873 Hz | 15.0 | **5.1 mm** | 9.8 mm | **10.2 mm** | 28.0° |
| 1000/1373/1747 Hz | 30.0 | **1.9 mm** | 4.0 mm | **3.9 mm** | 28.1° |

**Findings:**
1. **Position accuracy scales inversely with frequency** when window duration is fixed. More periods = cleaner demodulation = better position. The previous "geometric floor" was actually a demodulation cross-talk floor.
2. **At 1 kHz: 1.9mm (3-DOF), 3.9mm (6-DOF)** — well under the 1cm target, even with 30° sensor rotation.
3. **Orientation accuracy stays constant at ~28°** regardless of frequency — it's limited by the geometric ambiguity of 3-axis gradient fields, not demodulation quality.
4. **Max error at 1 kHz is only 4mm (3-DOF)** — the system is well-conditioned across the entire tracking volume at this frequency.

---

### Experiment 14: Noise injection (1 kHz, cage scale)

**Sensor model:** MLX90393 (Melexis) 3-axis Hall-effect magnetometer.
- Package: 3×3×0.9 mm QFN
- Resolution: 0.161 µT, RMS noise ~0.5 µT typical
- Sample rate: up to 1 kHz, current ~100 µA
- Cost: ~$2–3
- Noise modeled as additive white Gaussian, σ = 0.5 µT per axis per sample

> **Caveat (added later):** This σ was treated as fixed at all sample
> rates, and the simulation assumed 50 kSPS — neither of which the real
> MLX90393 can actually deliver in 3-axis burst mode (capped near
> ~500 Hz at minimum OSR).  Sensor-realistic experiments should use the
> `SensorProfile` abstraction (`source/sensor_profile.py`), which scales
> per-sample σ from a noise spectral density (nT/√Hz) and clamps to the
> sensor's max ODR.  See `--list-sensors` in `invert.py`.

**Setup:** 1 kHz excitation, 30ms window (30 periods), 50 kHz sample rate,
cage scale (0.5m), 30° rotation, 1500 samples averaged per window.

**Noise level sweep (1A drive):**

| σ (µT) | SNR | 3-DOF pos. error (median) | 3-DOF max | 6-DOF pos. error (median) | 6-DOF max |
|--------|-----|--------------------------|-----------|--------------------------|-----------|
| 0 | ∞ | 1.9 mm | 4.0 mm | 3.9 mm | 11.8 mm |
| 0.1 | 11 | 2.2 mm | 4.8 mm | 4.1 mm | 12.4 mm |
| 0.5 | 2 | 4.0 mm | 9.4 mm | 6.3 mm | 26.0 mm |
| 2.0 | 1 | 14.0 mm | 29.2 mm | 21.1 mm | 94.8 mm |
| 5.0 | 0 | 35.4 mm | 73.4 mm | 54.8 mm | 301.0 mm |

**Current scaling sweep (σ=0.5 µT, MLX90393 typical):**

| Current | Field (mean) | SNR | 3-DOF pos. error (median) | 3-DOF max | 6-DOF pos. error (median) | 6-DOF max |
|---------|-------------|-----|--------------------------|-----------|--------------------------|-----------|
| 1 A | 1.1 µT | 2 | 4.0 mm | 9.4 mm | 6.3 mm | 26.0 mm |
| 5 A | 5.5 µT | 11 | 2.2 mm | 4.8 mm | 4.1 mm | 12.4 mm |
| 10 A | 11.0 µT | 22 | 2.0 mm | 4.2 mm | 3.8 mm | 12.1 mm |
| 20 A | 22.0 µT | 44 | 1.9 mm | 4.0 mm | 3.8 mm | 11.9 mm |

**Findings:**
1. **Even at SNR=2 (1A, σ=0.5 µT), the system achieves 4mm/6.3mm.** Lock-in averaging over 1500 samples per window (50 kHz × 30ms) provides ~39× noise reduction (√1500).
2. **At 5A (SNR=11), accuracy is nearly at the noiseless floor** — 2.2mm vs 1.9mm. Diminishing returns beyond 10A.
3. **5A is the practical sweet spot** — well under 1cm for both modes, easily achievable electronically.
4. **The system is remarkably noise-tolerant** — the lock-in detection makes the dominant noise source (sensor white noise) much less impactful than expected.

---

## Experiments 15: Sensor-realistic re-run at 1 m cube cage

**Motivation:** Exp. 13–14 used a fixed σ = 0.5 µT noise floor and an
implicit 50 kSPS sample rate — neither achievable on the MLX90393 in
3-axis burst mode. This re-run uses the new `SensorProfile` abstraction
so the per-sample σ is derived from each part's noise spectral density
at its actual max ODR, and scales the cage from 0.5 m to 1 m on a side
(±50 cm coils, ±40 cm tracking volume).

Sensors compared: **MLX90394** (Hall, 70 nT/√Hz, 1 kHz max ODR) and
**AK09940A** (TMR, 30 nT/√Hz, 2.5 kHz max ODR). The original Exp. 13
1 kHz / 1.7 kHz carrier rows are *physically unreachable* on either
part — the highest carrier triple either part can support with Nyquist
headroom is 300/411/519 Hz on the AK09940A.

**Setup:** 1 m cube cage, 30 ms window (Exp. 13 standard), 30°
sensor rotation perturbation, 3 s trajectory, field table 25³.
Reproducible via `python demos/experiment_15_sensor_realistic.py`.

### Phase A — sensor + carrier sweep (5 A drive)

| Sensor | Carrier (Hz) | fs (Hz) | σ/sample | SNR | Mode | Pos err (med / max) | Orient err (med / max) |
|--------|--------------|---------|---------:|----:|------|---------------------|------------------------|
| MLX90394 | 100/137/173 | 1000 | 1.57 µT | 2.7 | 3-DOF | 56.1 / 166.4 mm | n/a |
| MLX90394 | 100/137/173 | 1000 | 1.57 µT | 2.7 | 6-DOF | 119.2 / 376.0 mm | 22.9° / 107.3° |
| AK09940A | 100/137/173 | 2500 | 1.06 µT | 3.6 | 3-DOF | 31.8 /  61.6 mm | n/a |
| AK09940A | 100/137/173 | 2500 | 1.06 µT | 3.6 | 6-DOF | 84.9 / 210.4 mm | 11.8° / 179.2° |
| AK09940A | 300/411/519 | 2500 | 1.06 µT | 3.6 | 3-DOF | **27.6 / 73.0 mm** | n/a |
| AK09940A | 300/411/519 | 2500 | 1.06 µT | 3.6 | 6-DOF | 75.8 / 211.8 mm | 11.3° / 171.5° |
| ideal    | 300/411/519 | 50k  |     0  |  ∞  | 3-DOF | **16.9 / 38.4 mm** | n/a |
| ideal    | 300/411/519 | 50k  |     0  |  ∞  | 6-DOF | 64.7 / 98.4 mm  | 6.5° / 170.1° |

### Phase B — drive current sweep (each sensor's best carrier, 3-DOF)

| Sensor | Carrier (Hz) | Drive | Field (mean) | SNR | Pos err (med / max) |
|--------|--------------|------:|-------------:|----:|---------------------|
| MLX90394 | 100/137/173 |  1 A | 2.6 µT | 1.7 | 244.3 / 691.8 mm |
| MLX90394 | 100/137/173 |  5 A | 4.2 µT | 2.7 |  56.1 / 166.4 mm |
| MLX90394 | 100/137/173 | 10 A | 7.3 µT | 4.6 |  37.5 / 111.2 mm |
| AK09940A | 300/411/519 |  1 A | 1.8 µT | 1.7 | 107.4 / 289.8 mm |
| AK09940A | 300/411/519 |  5 A | 3.8 µT | 3.6 |  27.6 /  73.0 mm |
| AK09940A | 300/411/519 | 10 A | 7.0 µT | 6.6 | **20.4 / 54.3 mm** |

### Findings

1. **Neither sensor hits ≤1 cm in a 1 m cube.** Best result is
   AK09940A @ 10 A drive at ~20 mm median error (3-DOF). The 1 cm
   target was achievable in a 0.5 m cube (Exp. 14) but the 1 m cube is
   roughly an order of magnitude harder at the same drive current.
2. **AK09940A clearly outperforms MLX90394** — about 2× lower median
   error at the same drive current, mostly because its higher max ODR
   lets the carrier triple sit at 300+ Hz instead of 100 Hz, giving
   more periods per 30 ms window.
3. **The grid resolution is now the dominant floor at this scale.**
   Even the noiseless / unconstrained-fs ideal case sits at 16.9 mm
   median (3-DOF). Grid spacing at res 25 in a 0.9 m volume is ~38 mm,
   and least-squares refinement only partially escapes that. To push
   below ~5 mm, expect to need res 35–50 or a finer second-pass grid.
4. **6-DOF is broken at this scale.** Even the ideal sensor gives
   65 mm median and orientation max errors of 170°+ — a regression
   from the 0.5 m cube's 4 mm / 28° in Exp. 14. The orientation-first
   SVD likely fails near cage edges where field directions deviate
   most from the cardinal axes; this needs a separate investigation.
5. **1 A drive is unusable** (SNR ≈ 1.7, errors 100–700 mm). The
   feasibility frontier for the 1 m cage starts at ~5 A.

### Implications for the bird tracker

To hit ≤1 cm in a 1 m cube cage, additional levers are needed:

- **Higher drive current** beyond 10 A (limited by coil heating).
- **Larger or more numerous coils** to flatten the field-gradient
  fall-off in the cage interior.
- **Higher carrier frequency**, which means abandoning integrated
  digital sensors. An analog AMR/TMR die plus a 50+ kSPS Σ-Δ ADC
  could run a true 1 kHz lock-in like the original simulation
  assumed.
- **Higher field-table resolution / iterative refinement** in the
  inversion to remove the ~17 mm geometric floor.
- **Fixing the 6-DOF degradation at cage scale** before relying on
  rotation tracking.

Treat this experiment as the realistic floor of what the named parts
can do *as drop-in replacements* at this cage scale; the sub-cm goal
needs at least one of the levers above.

---

## Experiment 16: Analog frontend recovers the 1 kHz lock-in regime

**Motivation:** Exp. 15 showed that integrated digital magnetometers
hit a wall at the 1 m cube scale — their max ODRs cap the achievable
carrier frequency well below 1 kHz, leaving only 3–15 carrier periods
in a 30 ms demodulation window. The original Exp. 13/14 simulation
assumed a real 1 kHz lock-in (fs = 50 kSPS). This experiment models a
realistic wearable signal chain that actually delivers that:

  * **Sensor:** analog TMR die (e.g. NVE AAL024 or MMT MMR3-50J)
  * **Amp:** AD8421 instrumentation amp, gain ~500×, AC-coupled to
    reject the Earth field DC
  * **Anti-alias:** RC low-pass at ~10 kHz
  * **ADC:** nRF52832 SAADC, 3-channel scan, ~50 kSPS/axis, 14-bit
    oversampled

Registered as the `TMR_analog_nRF52` SensorProfile: 5 nT/√Hz (a
realistic system-level estimate including sensor + amp + ADC + board
pickup), 50 kHz max ODR. Field table resolution bumped to 35³ to drop
Exp. 15's grid-quantization floor from ~17 mm to ~4 mm so we can see
the actual sensor-limited error.

### Carrier sweep at 5 A drive, 3-DOF

| Carrier (Hz) | fs | σ/sample | Periods in 30 ms | Pos err (med / max) |
|--------------|------|---------:|-----------------:|---------------------|
| 100/137/173    | 50 kHz | 0.79 µT |   3 | 20.7 / 48.0 mm |
| 300/411/519    | 50 kHz | 0.79 µT |   9 | 17.1 / 45.0 mm |
| 500/687/873    | 50 kHz | 0.79 µT |  15 | 12.8 / 29.8 mm |
| 1000/1373/1747 | 50 kHz | 0.79 µT |  30 | **5.5 / 14.4 mm** |
| (ideal, 1 kHz triple) | 50 kHz | 0 | 30 | 4.0 / 8.9 mm |

The Exp. 13 finding holds at 1 m cube and with realistic noise: more
periods per window monotonically improves accuracy, with the analog
chain reaching the geometric floor at the 1 kHz triple.

### Drive-current sweep at 1 kHz triple, 3-DOF

| Drive | Field (mean) | SNR | Pos err (med / max) |
|------:|-------------:|----:|---------------------|
|  1 A | 1.4 µT | 1.8 | 20.6 / 46.6 mm |
|  5 A | 3.6 µT | 4.6 |  5.5 / 14.4 mm |
| 10 A | 6.9 µT | 8.8 |  **4.4 / 10.9 mm** |

5 A already clears the ≤1 cm median target with ~14 mm max; 10 A
brings the system within ~10 % of the ideal noiseless floor.

### 6-DOF: still broken at cage scale

| Run | Pos err (med / max) | Orient err (med / max) |
|-----|---------------------|------------------------|
| TMR_analog_nRF52, 1 kHz, 5 A | 62.4 / 77.2 mm | 6.0° / 171.0° |
| ideal, 1 kHz, 5 A            | 61.6 / 71.2 mm |  6.0° / 169.7° |

The 6-DOF result is independent of the sensor — even the noiseless
ideal sees the same ~60 mm position error and 170°+ orientation max.
This confirms what Exp. 15 hinted at: the orientation-first SVD
initializer fails near cage edges at this geometric scale, and the
joint refinement can't recover. This is a pipeline issue, not a
sensor issue, and needs a separate fix.

> **Update from Exp. 19:** The "fails near cage edges" diagnosis turned
> out to be wrong. The actual cause is the SVD-based rotation
> initialiser landing in a 180°-flipped basin and `prev_rotvec`
> carrying that wrong rotation forward through the trace; cage
> position is incidental. The fix (rotation-invariant coarse search
> plus multi-start refinement around cardinal-axis 180° flips) is in
> Exp. 19 — with it, 6-DOF position median drops from 62 mm to ~14 mm
> and orientation median from 6° to ~2° on the same 1 m cube cage.

### Findings

1. **The analog signal chain hits the ≤1 cm median target at the 1 m
   cube scale**, even with conservative 5 nT/√Hz system-level noise.
   At 5 A drive: 5.5 mm median / 14 mm max (3-DOF, with 30° rotation
   on the sensor).
2. **Going from integrated digital → analog frontend is worth ~5×
   in median position error.** AK09940A best at 5 A: 27.6 mm.
   TMR_analog_nRF52 at 5 A: 5.5 mm — same drive current, same cage,
   same window, just a faster sample rate that lets the carrier sit
   at 1 kHz instead of 300 Hz.
3. **At 5 A the system is already noise-limited**, not geometry-
   limited. The gap between TMR_analog_nRF52 (5.5 mm) and the ideal
   sensor (4.0 mm) at the same configuration is ~1.5 mm — small.
   Going to 10 A closes most of that.
4. **1 A drive gives SNR ~1.8 and is unusable** even with the analog
   frontend. The minimum practical drive current at 1 m cube is
   around 3–5 A.
5. **The nRF52832 SAADC is not the bottleneck.** Its 200 kSPS shared
   across 3 channels (~66 kSPS/axis) is plenty for a 1 kHz lock-in
   with 30+ samples per period. The 14-bit oversampled mode plus
   ~500× preamp gain puts the ADC noise floor below the sensor.
6. **6-DOF needs its own debugging pass** before the analog chain is
   usable for tracking a rotating bird. The position errors are
   acceptable in median (≤65 mm) but the orientation outliers (170°
   max) indicate periodic catastrophic failures.

### Implications

For the budgerigar tracker spec:

- **Position tracking under 1 cm median is achievable** in a 1 m cube
  with the analog TMR + nRF52832 chain at ≥5 A drive current,
  *provided the sensor stays at fixed orientation* (3-DOF mode).
- **Adding a free-rotation sensor is currently blocked** by the
  6-DOF pipeline regression — fixing that should be the next
  priority before hardware prototyping.
- **Power budget is roughly:** ~15 mW (analog frontend) +
  ~5–10 mW (nRF52832 active) + BLE radio bursts. Compatible with
  small Li-ion cells for multi-hour runtime in a budgie tag.
- **Mass budget** for the sensor head + amp + nRF52832 module
  + battery is ~1.5 g — within the 5 % rule for a 30–40 g
  budgerigar.

---

## Experiment 17: Smaller cage, same coils — disproves edge hypothesis

**Motivation:** Exp. 15-16 showed the 6-DOF pipeline produces ~60 mm
position medians and ~170° orientation max errors at the 1 m cube
scale, even with an ideal noiseless sensor. The Exp. 12 finding
suggested cage-edge regions degrade the orientation-first SVD
initializer. This experiment tests whether constraining the bird to a
0.75 m interior tracking volume (coils unchanged) recovers the 6-DOF
performance seen in the original 0.5 m cube of Exp. 14.

**Setup:** identical to Exp. 16 (1.4 m diameter coils at ±0.5 m,
TMR_analog_nRF52, 1 kHz triple, 30 ms window, 30° rotation, res 35³)
except the bird's path is restricted to ±0.30 m and the field-table
search bounds are tightened to ±0.40 m, modeling a 0.75 m cube cage.

| Sensor | Carrier (Hz) | Drive | Mode | Pos err (med/max) | Orient err (med/max) |
|--------|--------------|------:|------|-------------------|----------------------|
| TMR_analog_nRF52 | 1000/1373/1747 |  5 A | 3-DOF | 5.46 / 14.14 mm | n/a |
| ideal            | 1000/1373/1747 |  5 A | 3-DOF | 3.54 /  7.96 mm | n/a |
| TMR_analog_nRF52 | 1000/1373/1747 |  5 A | 6-DOF | 65.9 / 81.3 mm  | 6.6° / 171.6° |
| ideal            | 1000/1373/1747 |  5 A | 6-DOF | 65.5 / 77.2 mm  | 6.2° / 171.6° |
| TMR_analog_nRF52 | 1000/1373/1747 | 10 A | 6-DOF | 65.6 / 77.4 mm  | 6.3° / 171.6° |
| TMR_analog_nRF52 |  100/ 137/ 173 |  5 A | 6-DOF | 71.8 / 129.0 mm | 7.5° / 173.5° |

### Findings

1. **The cage-edge hypothesis is wrong.** Restricting the bird to the
   cage interior had **no effect** on 6-DOF position or orientation
   error — both stayed essentially identical to Exp. 16's 1 m cube
   numbers (62.4 / 77.2 mm). 3-DOF, by contrast, was unchanged at
   5.5 mm — confirming the position pipeline itself isn't broken.
2. **The orientation max error is *deterministic*.** Four different
   6-DOF rows (different sensors, different drive currents, same cage
   geometry) all converge to **171.6°** maximum orientation error.
   This is not random window-to-window noise; there's a specific
   trajectory configuration that consistently triggers a near-180°
   flip in the SVD-based rotation init.
3. **Drive current doesn't help.** Going from 5 A to 10 A
   (SNR 3.9 → 7.4) leaves the 6-DOF errors unchanged — confirming
   the failure is structural, not noise-limited.
4. **Lower carrier is *worse*** (72 / 129 mm at 100 Hz triple vs 66 /
   81 mm at 1 kHz), consistent with cross-talk during demodulation
   amplifying the orientation pathology rather than fixing it.

### Likely root cause

The deterministic 171.6° outliers strongly point to a specific point
on the synthetic trajectory where rotation generation is pathological.
The most likely candidate is the **ping-pong endpoint** in
`synthesize_signal`: when the path direction reverses at the bounce
point, the path tangent flips by ~180°, which discontinuously flips
the *base* orientation in `generate_rotations`. The optimizer can lock
onto the wrong side of this discontinuity, giving ~180° orientation
error at exactly that window.

If this is correct, the 6-DOF results in Exp. 14 (4.1 mm / 28°) were
"lucky": the 0.5 m cube path was short enough relative to the 3 s
duration that the bird never reached the bounce point. At 1 m / 0.75 m
cube the path is much longer, the bird *does* bounce, and the
discontinuity gets sampled.

### Implications

- Stop chasing geometry-driven explanations for the 6-DOF failure —
  it's a pipeline issue, not a cage issue.
- Two natural next steps converge on roughly the same fix:
  - **Hardware:** add a second magnetometer at fixed PCB offset.
    Two sensors must agree on the rotation, which makes the
    catastrophic flip impossible (the wrong-rotation hypothesis is
    inconsistent with sensor 2's measurement). This also adds
    gradient information for position.
  - **Software:** use the previous window's rotation as a fallback
    initialization when the SVD-based init disagrees with it by
    more than a threshold. Cheap to try, and might be enough on
    its own.
- Either fix is probably worth pursuing before chasing the underlying
  rotation-generator bug — the inversion pipeline should be robust to
  whatever orientation trajectories the bird actually produces.

> **Update from Exp. 19:** The diagnosis above was wrong twice over. The
> ping-pong code does not actually trigger at this duration (path
> length ≈ 3 m, traversal ≈ 0.3 m), so tangent flips at endpoints aren't
> the issue. The real bug is in the 6-DOF rotation initialiser, which
> Exp. 19 isolated and fixed via multi-start refinement. See Exp. 19.

---

## Experiment 18: Dual-magnetometer backpack — and a synthetic-data discovery

**Motivation:** Per Exp. 15-17, single-sensor 6-DOF gives ~62 mm median
position errors and ~170° orientation max errors at the 1 m cube scale,
even with an ideal noiseless sensor. Hypothesis: two magnetometers at
fixed PCB offset on a backpack should add a hard constraint (both
sensors must agree on R) that breaks the orientation ambiguity, while
also adding gradient information for position.

**Implementation:** New `DualSensorConfig` + `_refine_dual_3dof` /
`_refine_dual_6dof` / `invert_trace_dual_3dof` / `invert_trace_dual_6dof`
in `source/inversion.py`. Original single-sensor functions left
unchanged for reproducibility. The 6-DOF dual pipeline initialises
rotation via SVD on both sensors' demodulated vectors stacked together
(Wahba problem with 2K vector pairs instead of K), then jointly
refines position + rotation against both sensors' residuals.

**Setup:** 3 cm baseline along sensor +x, identical trajectory + per-
sensor noise realisation fed through *both* single- and dual-sensor
pipelines for an apples-to-apples comparison. Cage, carriers, drive
sweep, and field-table resolution match Exp. 16-17.

| Sensor | Carrier | Drive | Mode | Single (med/max) | Dual (med/max) | Single orient (med/max) | Dual orient (med/max) |
|--------|---------|------:|------|------------------|----------------|------------------------|----------------------|
| TMR_analog_nRF52 | 1000/1373/1747 |  5 A | 3-DOF |  5.45 / 14.43 mm | **4.72 / 11.64 mm** | n/a | n/a |
| TMR_analog_nRF52 | 1000/1373/1747 |  5 A | 6-DOF | 62.40 / 77.21 mm | 62.24 / 73.58 mm | 6.0° / 171.0° | 6.3° / 170.5° |
| ideal            | 1000/1373/1747 |  5 A | 6-DOF | 61.57 / 71.20 mm | 61.94 / 71.30 mm | 6.0° / 169.7° | 6.1° / 169.8° |
| TMR_analog_nRF52 | 1000/1373/1747 | 10 A | 6-DOF | 61.95 / 73.24 mm | 61.86 / 72.26 mm | 6.1° / 170.4° | 6.3° / 170.1° |
| TMR_analog_nRF52 | 1000/1373/1747 |  1 A | 6-DOF | 69.30 / 113.9 mm | **65.97 / 93.75 mm** | 9.0° / 176.7° | 7.8° / 173.2° |
| TMR_analog_nRF52 |  100/ 137/ 173 |  5 A | 6-DOF | 71.14 / 156.8 mm | 69.87 / 158.0 mm | 7.1° / 171.0° | 7.0° / 171.7° |

### Findings

1. **3-DOF: dual works as expected.** Median 5.45 → 4.72 mm (13 %
   improvement); max 14.43 → 11.64 mm (19 %). Smaller than the √2 ≈
   1.41 noise-reduction prediction because the two sensors share some
   correlated information through the field-table interpolation, but
   the trend is right and the gradient constraint is providing real
   value.
2. **6-DOF: dual provides no meaningful improvement at this test
   scenario.** Position medians and orientation maxes are essentially
   unchanged across the matrix.
3. **The orientation max value is independent of sensor noise.** Ideal
   noiseless sensor: 169.7° max. TMR with 5 nT/√Hz noise at 5 A: 171°.
   Same sensor at 10 A: 170°. Same sensor at 1 A (5× more noise):
   176°. The max is hardly responding to noise at all — a single-digit
   degree change for a 5× noise change. This is decisive evidence that
   what we've been calling the "171° flip" is **not noise-driven, and
   not a single-sensor inversion failure**.
4. **Slight benefit at low SNR.** The 1 A row is the only one where
   dual mag helps materially (113.9 → 93.7 mm max, 18 %), consistent
   with the noise-reduction effect that dominates when SNR is low.

### What's actually causing the ~170° orientation maxes

> **Update from Exp. 19:** The diagnosis below was wrong on two counts.
> (1) The ping-pong code does not actually trigger at this trajectory
> length — path arc-length ≈ 3 m, traversal ≈ 0.3 m in 3 s, so the
> bird never hits an endpoint and there is no tangent flip in
> `generate_rotations`. (2) The real cause of the 62 mm position
> median *and* the ~170° orientation max is the 6-DOF rotation
> initialiser converging into a 180°-flipped basin (see Exp. 19).
> With the multi-start fix, position median drops to ~14 mm and
> orientation median to ~2°. A residual ~178° orientation max persists
> in a small fraction of windows because the cardinal-axis 180° flips
> in the candidate set don't span every Wahba ambiguity — that's a
> separate, narrower issue. The original (incorrect) reasoning is
> kept below for the historical record.

The deterministic orientation maxes near 170° in every 6-DOF row,
including with two sensors and an ideal noiseless magnetometer, point
to an artifact in the synthetic test setup, not a property of the
inversion pipeline. The chain of reasoning:

1. `synthesize_signal` ping-pongs the bird along the path
   (forward → reverse → forward). At each path endpoint, the local
   tangent direction reverses sign in one sample.
2. `generate_rotations` builds the base sensor orientation by aligning
   the sensor's +x axis with the path tangent. When the tangent flips
   sign, the base rotation jumps by ~180° between adjacent samples.
3. The error metric in `orientation_errors` builds a SLERP through
   *all* the per-sample true rotations and queries it at the inversion
   timestamps. A 180° step between adjacent SLERP keypoints makes the
   interpolation between them pathological — the "true" rotation at
   times near the discontinuity is ill-defined.
4. The inversion correctly tracks the actually-applied rotation in
   each window (which is one of the two stable orientations on either
   side of the bounce), but is graded against an interpolated truth
   that doesn't match either side cleanly.

The 6° median orientation error from this same metric is fine because
it's dominated by windows away from the bounce points; the max picks
up the few pathological windows.

The dual-sensor 6-DOF pipeline is therefore **not the bottleneck** —
the bottleneck is the test scenario. Both single- and dual-sensor
inversions are computing the right answer for the field measurements
they're given; the disagreement is between what the inversion can
recover and what the SLERP-truth says it should have.

### What about the 62 mm 6-DOF position median?

This is also large compared to 3-DOF (5 mm), and dual sensors don't
fix it. Likely contributors:

- **Rotation–position coupling in the optimizer.** With 6 parameters
  to fit instead of 3, the residual landscape is more complex; small
  errors in rotation get traded against position to keep the residual
  low.
- **The 30° per-axis rotation perturbation is adversarial.** Real
  bird orientation changes during flight are large but smooth; the
  test's randomized SLERP perturbation visits more of orientation
  space than a real bird would.
- **Bouncing-tangent artifact contributes here too.** Windows where
  the truth is discontinuous likely also produce position errors that
  drag the median up.

Fixing any of these on the test side would let us see what dual mag
actually buys for 6-DOF.

### Implications and next steps

- **Dual-magnetometer backpack at 3 cm baseline is implemented and
  works for 3-DOF.** Modest but real improvement; the pipeline is
  ready for use whenever you choose to add the second sensor.
- **6-DOF can't be evaluated until the test harness is fixed.** The
  current ping-pong + tangent-aligned + SLERP-interpolated setup is
  hiding any real 6-DOF benefit (or harm) under a deterministic
  ~170° artifact.
- Three orthogonal cleanup steps that would let us actually evaluate
  6-DOF:
  1. Replace the ping-pong path with a non-bouncing trajectory
     (e.g., a closed loop, or a path long enough that the bird never
     reaches the end in the simulation duration).
  2. Smooth the rotation at path endpoints in `generate_rotations`
     instead of letting the tangent reverse.
  3. Make the orientation-error metric robust to truth
     discontinuities (e.g., compute error vs the nearest of the two
     adjacent true rotations rather than the SLERP).
- The right next experiment is one of those — probably (1), since
  it also better models a bird actually moving around in a cage.

---

## Experiment 19: Lissajous trajectory exposes (and fixes) the 6-DOF bug

**Motivation:** Exp. 18 left an unsolved puzzle: 6-DOF gave ~62 mm
position medians and ~170° orientation max errors, even with an ideal
sensor and a second magnetometer. Exp. 18's writeup blamed
ping-pong-induced tangent flips at path endpoints, but on inspection
the bird only traverses ~10 % of the path in 3 s — no bouncing
actually triggers. The cause was elsewhere.

This experiment substitutes the open-spline cage path with a closed
**Lissajous trajectory** (`source/path.LissajousPath`, ratios 3:4:5)
traversed in `"loop"` mode (a new `traversal_mode` parameter on the
synthesisers), so any artifact involving path-endpoint tangent
reversal is structurally impossible.

### Phase A: Lissajous *without* fixing the inversion

The path-shape change alone made things dramatically *worse*, not
better. With the original `invert_trace_6dof` / `invert_trace_dual_6dof`:

| Mode | Single (med/max) | Dual (med/max) | Single orient | Dual orient |
|------|------------------|----------------|---------------|-------------|
| 3-DOF, TMR 5A | 5.5 / 14 mm | 4.7 / 12 mm | n/a | n/a |
| 6-DOF, TMR 5A | **534 / 568 mm** | 526 / 548 mm | **176° / 179°** | 177° / 179° |
| 6-DOF, ideal 5A | 534 / 559 mm | 526 / 546 mm | 176° / 179° | 176° / 179° |

3-DOF position is unchanged — that pipeline is healthy. 6-DOF is
catastrophically wrong at *every* window: median orientation 176°
means the inversion converges to a rotation almost exactly 180° from
truth on most windows. This is decisive: the fault is in the 6-DOF
pipeline itself, not in the test harness.

The mechanism: `_estimate_rotation_from_directions`, used as the very
first rotation guess on the first window, assumes channel k's lab
field is along the k-th cardinal axis. Approximately true near the
cage centre, increasingly wrong off-axis. The Lissajous path puts the
bird at non-trivial off-axis positions from the start, the cardinal-
axis assumption is wrong, the SVD lands in a 180°-flipped basin, the
first window converges there, and `prev_rotvec` carries the wrong
rotation forward through the rest of the trace. The previous experiments
(15-18) were "lucky" — their starting positions happened to land in
the right basin.

### Phase B: multi-start fix

Two changes (new `invert_trace_6dof_robust` and
`invert_trace_dual_6dof_robust` in `source/inversion.py`; originals
preserved unchanged):

1. **Coarse position via rotation-invariant magnitudes.** Replace the
   direction-aware `query_coarse` with `query_coarse_rotated` (uses
   per-channel field magnitudes, which are unaffected by sensor
   rotation). The position estimate no longer depends on a possibly-
   wrong rotation guess.
2. **Multi-start refinement.** Build a candidate set: the SVD result
   computed from actual lab fields at the coarse position, plus that
   SVD result composed with each of three 180° flips about the
   cardinal axes, plus the previous window's rotation if available.
   Run the joint least-squares refinement from each candidate; keep
   the lowest-residual solution.

Re-running the matrix with the robust pipeline:

| Sensor | Carrier (Hz) | Drive | Mode | Single (med/max) | Dual (med/max) | Single orient (med/max) | Dual orient (med/max) |
|--------|--------------|------:|------|------------------|----------------|------------------------|----------------------|
| TMR_analog_nRF52 | 1000/1373/1747 |  5 A | 3-DOF |  5.4 / 13 mm | **4.4 / 9.5 mm** | n/a | n/a |
| TMR_analog_nRF52 | 1000/1373/1747 |  5 A | 6-DOF | 16.0 / 45 mm | **14.0 / 29 mm** | 2.3° / 178.6° | 2.2° / 178.9° |
| ideal            | 1000/1373/1747 |  5 A | 6-DOF | 14.4 / 25 mm | 13.9 / 22 mm | 1.9° / 178.9° | 1.7° / 179.0° |
| TMR_analog_nRF52 | 1000/1373/1747 | 10 A | 6-DOF | 14.9 / 28 mm | 13.9 / 25 mm | 2.0° / 178.8° | 1.8° / 178.6° |
| TMR_analog_nRF52 | 1000/1373/1747 |  1 A | 6-DOF | 38 / 573 mm | **26 / 542 mm** | 10.8° / 179.6° | 6.7° / 177.1° |
| TMR_analog_nRF52 |  100/ 137/ 173 |  5 A | 6-DOF | 33 / 112 mm | 30 /  66 mm | 6.1° / 179.6° | 5.2° / 178.8° |

### Findings

1. **The 180° trap is broken on most windows.** Orientation median
   dropped from 176° → ~2° at high SNR. Position median dropped from
   ~530 mm → ~14 mm. This is a structural fix, not a tuning win.
2. **Dual magnetometer now genuinely helps.** With the rotation
   initializer working, the gradient signal at the 3 cm baseline
   provides real benefit:
   - 3-DOF position: 5.4 mm → 4.4 mm median (19 %), 13 mm → 9.5 mm max (27 %)
   - 6-DOF position median: 16 → 14 mm (12 %), max 45 → 29 mm (36 %)
   - 6-DOF position at 1 A drive: median 38 → 26 mm (33 %)
   - Dual orientation also slightly better (2.3° → 2.2° at 5 A,
     6.7° → 10.8° at 1 A).
3. **Residual orientation max errors of ~179° remain on every 6-DOF
   row, including ideal sensor.** The multi-start with three 180°
   flips about cardinal axes does not span all possible Wahba
   ambiguities — the ambiguous rotation is generally a 180° rotation
   about an axis that depends on the field-vector configuration at
   that window. So a small fraction of windows still pick the wrong
   basin. Mechanism is now well-understood; further fix is a richer
   candidate set or a confidence metric.
4. **Low-SNR (1 A) is unstable in 6-DOF.** Position max 542 mm even
   with dual mag suggests the multi-start can't reliably distinguish
   candidates when noise dominates the inter-candidate residual
   differences. Practical implication: 6-DOF needs ≥3-5 A drive for
   reliable tracking on this geometry.

### Implications for the bird tracker

- **6-DOF position tracking is no longer fundamentally broken.**
  At 5-10 A drive with the analog frontend + dual magnetometer, the
  median position error is ~14 mm — close to the 1 cm spec, with a
  clear path to improvement.
- **Median orientation tracking is excellent (~2°)**; max orientation
  outliers (~179°) need additional work but are isolated to a small
  fraction of windows. Worth flagging via a confidence metric and
  filtering or interpolating across them.
- **Dual magnetometer was the right call.** Without the rotation-
  initializer fix the dual sensor couldn't show its benefit, but with
  the fix it is consistently helpful for 6-DOF and modestly helpful
  for 3-DOF.
- **Path-shape choice in test harnesses matters.** Closed (Lissajous)
  trajectories with `"loop"` traversal expose 6-DOF pathologies that
  open ping-pong paths hide. New experiments should default to closed
  paths for orientation-tracking studies.

### Costs

The robust 6-DOF inversion runs ~6× slower per window than the
original (5 candidates × full least-squares refinement each). For a
3 s, 50 kSPS, 30 ms-window trace at res 35³, a single 6-DOF row takes
~18-20 minutes. Tractable for offline analysis; for real-time on the
nRF52832, a confidence-based "only do multi-start when the cheap
single-start residual is high" gating would recover most of the speed
without losing the robustness benefit.

---

## Current best results

These are the headline numbers as of Exp. 19, on the **1 m cube cage**
with the **closed Lissajous trajectory**, the **multi-start 6-DOF
inversion** (`invert_trace_6dof_robust` / `invert_trace_dual_6dof_robust`),
and the **TMR_analog_nRF52** sensor profile (5 nT/√Hz, 50 kSPS) at the
1 kHz/1373 Hz/1747 Hz coprime carrier triple, 30 ms demodulation
window, 30° rotation perturbation. "Single" = one sensor; "dual" =
two magnetometers at 3 cm baseline.

| Scenario | Pos median | Pos max | Orient median | Orient max |
|----------|-----------:|--------:|--------------:|-----------:|
| 3-DOF, TMR, 5 A, single | 5.4 mm | 13 mm | n/a | n/a |
| 3-DOF, TMR, 5 A, dual   | **4.4 mm** | **9.5 mm** | n/a | n/a |
| 6-DOF, TMR, 5 A, single | 16 mm | 45 mm | 2.3° | 178.6° |
| 6-DOF, TMR, 5 A, dual   | **14 mm** | **29 mm** | **2.2°** | 178.9° |
| 6-DOF, ideal, 5 A, dual | 13.9 mm | 22 mm | 1.7° | 179.0° |
| 6-DOF, TMR, 10 A, dual  | 13.9 mm | 25 mm | 1.8° | 178.6° |
| 6-DOF, TMR, 1 A, dual   | 26 mm | 542 mm | 6.7° | 177.1° |

For the **earlier** numbers from Exps. 13-14 (0.5 m cube cage,
fixed-σ noise model, original single-start 6-DOF, MLX90393-style
parameters), see the per-experiment sections — those values are
preserved in their original context.

## Current key findings

1. **Demodulation cross-talk is the main position-accuracy
   bottleneck at low carriers.** Periods/window scales position
   error roughly as `1/√(periods)`. Exp. 13's 30 ms-window result
   (1.9 mm at 30 periods, 13 mm at 3 periods) holds.
2. **An analog signal chain is required to support 1 kHz lock-in.**
   Integrated digital sensors cap at ~250 Hz carrier (Exp. 15-16);
   the analog TMR + AD8421 + nRF52832 SAADC chain (Exp. 16) recovers
   the kHz-lock-in regime and ~2× better position accuracy at 1 m
   cube scale.
3. **6-DOF needs multi-start rotation initialisation.** The
   original SVD-only initializer converges into a 180°-flipped
   basin on smooth trajectories (Exp. 19 Phase A: ideal sensor gives
   176° orientation median). Adding three cardinal-axis 180° flips
   plus the previous window's rotation as candidates, and picking
   the lowest-residual refinement, drops orientation median from
   176° to ~2° (Exp. 19 Phase B).
4. **Dual magnetometers at 3 cm baseline give consistent gains
   once the rotation initialiser is fixed.** Roughly 13-19% better
   3-DOF median, 12-36% better 6-DOF max, and biggest impact at low
   SNR (33 % better at 1 A drive). The implementation is in
   `invert_trace_dual_*` in `source/inversion.py`.
5. **Residual ~178° orientation outliers in a small fraction of
   6-DOF windows.** The cardinal-axis 180° flips don't span every
   Wahba ambiguity; some windows still pick the wrong basin. Median
   is unaffected; max is.
6. **Low SNR breaks 6-DOF.** At 1 A drive, multi-start can't
   reliably distinguish candidates — position max errors blow up
   (542 mm in Exp. 19). Practical floor for 6-DOF: ~3-5 A drive.
7. **`SensorProfile` keeps experiments sensor-comparable.** Per-sample
   σ derives from noise spectral density and clamped sample rate, so
   sweeping profiles produces physically consistent noise. See
   `source/sensor_profile.py` and `python invert.py --list-sensors`.

## Open work

1. **Richer Wahba candidate set or confidence-flagging** to close
   the residual ~178° orientation max-error outliers (item 5 above).
2. **4th coil pair (Helmholtz uniform-field reference)** at a 4th
   coprime frequency, providing position-independent absolute
   orientation. Should largely eliminate the residual outliers and
   add robustness during high-acceleration motion. Discussed with
   user; not yet experimented.
3. **Real-world hardware validation** with the dual-magnetometer
   analog tag (`hardware/dual_mag_tag/`). Schematic netlist and
   symbol library are in place; sensor part choice (HMC1053 vs NVE
   TMR vs single-axis Hall ×3) still pending pricing/availability.
4. **Confidence metric per estimate** based on the joint residual
   after refinement, and outlier filtering / interpolation across
   bad windows.
5. **Multipass inversion at 6-DOF scale** (single-pass tested in
   Exp. 10; not yet revisited with the multi-start fix).
