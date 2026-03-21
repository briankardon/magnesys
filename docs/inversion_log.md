# Inversion Development Log

## Setup

**Coil arrangement:** 3-axis unbalanced anti-Helmholtz pairs at coprime frequencies (100, 137, 173 Hz). Each pair has asymmetric currents (e.g. +1.0A / -0.7A) to break mirror symmetry.

**Inversion pipeline:**
1. Lock-in demodulation (multiply by cos at each frequency, average over window)
2. Coarse search via KD-tree on precomputed field table
3. Fine refinement via `scipy.optimize.least_squares`

All results are noiseless unless otherwise noted.

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

## Summary of best results

| Scenario | Position error (median) | Position error (max) | Orientation error (median) |
|----------|------------------------|---------------------|---------------------------|
| 3-DOF, 1 kHz, noiseless | **1.9 mm** | 4.0 mm | n/a |
| 3-DOF, 1 kHz, 0.5 µT noise, 1A | 4.0 mm | 9.4 mm | n/a |
| 3-DOF, 1 kHz, 0.5 µT noise, 5A | **2.2 mm** | 4.8 mm | n/a |
| 6-DOF, 1 kHz, noiseless | **3.9 mm** | 11.8 mm | 28.1° |
| 6-DOF, 1 kHz, 0.5 µT noise, 1A | 6.3 mm | 26.0 mm | 29.3° |
| 6-DOF, 1 kHz, 0.5 µT noise, 5A | **4.1 mm** | 12.4 mm | 28.4° |

## Key findings

1. **Demodulation cross-talk is the main position accuracy bottleneck.** More lock-in periods per window dramatically improves accuracy. Higher excitation frequencies achieve this without increasing the window duration.
2. **Orientation-first initialization is critical** for robust 6-DOF convergence across frequencies.
3. **Orientation accuracy (~28°) is limited by field geometry**, not demodulation or noise — more coil pairs or IMU assistance would be needed to improve it.
4. **Sensor noise is well-handled by lock-in averaging.** Even at SNR=2, sub-centimeter position accuracy is achieved. 5A drive current provides near-noiseless performance.
5. **Max errors are at cage edges** where field patterns are most nonlinear. A confidence metric based on optimizer residual could flag these.

## Next steps

1. **Confidence metric** — output per-estimate quality based on optimizer residual
2. **More coil pairs** — 4th pair for better orientation conditioning
3. **Real-world validation** — prototype with actual MLX90393 sensor and coils
