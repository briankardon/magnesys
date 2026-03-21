# Inversion Development Log

## Setup

**Coil arrangement:** 3-axis unbalanced anti-Helmholtz pairs at coprime frequencies (100, 137, 173 Hz). Each pair has asymmetric currents (e.g. +1.0A / -0.7A) to break mirror symmetry.

**Inversion pipeline:**
1. Lock-in demodulation (multiply by cos at each frequency, average over window)
2. Coarse search via KD-tree on precomputed field table
3. Fine refinement via `scipy.optimize.least_squares`

All results are noiseless unless otherwise noted.

---

## Experiment 1: Initial 3-DOF inversion (small scale)

**Scale:** 12cm tracking volume, coils at ±7.5cm
**Grid resolution:** 25³
**Window:** 1 period of lowest frequency (10ms)

| Metric | Result |
|--------|--------|
| Position median | 32 mm |
| Position max | 52 mm |

**Problem identified:** Each coil treated as a separate source (6 entries in field table), but demodulator can only separate by frequency — two coils sharing a frequency get summed. Demodulated values didn't match the per-source field table entries.

---

## Experiment 2: Frequency grouping fix

**Change:** Group sources by frequency in the field table. Two coils at 100 Hz → one combined channel. 6 sources → 3 frequency channels.

**Scale:** 12cm, resolution 30³, 1-period window

| Metric | Result |
|--------|--------|
| Stationary point error | 9.2 mm |
| Trace position median | 24.5 mm |

**Improvement:** Dramatic — demodulation now matches the field table. But still mediocre accuracy.

---

## Experiment 3: Window length sweep (stationary point)

**Test:** Fixed sensor position, varying demodulation window length.

| Window (periods) | Demod error (max) | Position error |
|-------------------|-------------------|----------------|
| 1 | 7.5e-7 | 9.2 mm |
| 2 | 4.4e-7 | 4.6 mm |
| 5 | 1.3e-7 | 1.5 mm |
| 10 | 9.2e-8 | 0.9 mm |

**Finding:** Demodulation accuracy improves monotonically with window length. Cross-talk between coprime frequencies (100/137/173 Hz) needs multiple periods to average out.

---

## Experiment 4: Moving trace with 3-period windows

**Scale:** 12cm, resolution 30³, speed 5 cm/s

| Window (periods) | Estimates | Median | Max |
|-------------------|-----------|--------|-----|
| 1 | 199 | 22.1 mm | 45.8 mm |
| 3 | 65 | **3.3 mm** | 7.6 mm |
| 5 | 39 | 3.1 mm | 7.0 mm |

**Finding:** 3-period windows achieve sub-centimeter accuracy. The sweet spot — longer windows give diminishing returns but fewer estimates.

---

## Experiment 5: Current strength scaling (small scale)

**Test:** Vary current magnitude from 1A to 10A.

| Current | 3-DOF median | 6-DOF median |
|---------|-------------|-------------|
| 1.0 A | 13.1 mm | 26.2 mm |
| 5.0 A | 13.1 mm | 26.2 mm |
| 10.0 A | 13.1 mm | 26.2 mm |

**Finding:** Current strength has zero effect in the noiseless case. The field scales linearly and the inversion is scale-invariant. Current will matter only when sensor noise is introduced (SNR).

---

## Experiment 6: 6-DOF inversion (position + orientation)

**First attempt:** Used same KD-tree coarse search as 3-DOF. Failed badly (98mm position, 145° orientation) because rotation changes the field direction, making the direction-aware KD-tree useless.

**Fixes applied:**
1. Rotation-invariant coarse search using per-channel field **magnitudes** (invariant under rotation)
2. SVD-based initial rotation estimate (Wahba's problem) from the coarse position

| Metric | Before fixes | After fixes |
|--------|-------------|-------------|
| Position median | 98 mm | **9.4 mm** |
| Orientation median | 145° | **31.8°** |

**Scale:** 12cm, 30° max perturbation, 3-period windows

---

## Experiment 7: 4-DOF + IMU (position + yaw, tilt from accelerometer)

**Idea:** Use accelerometer to fix roll/pitch (tilt from gravity), only optimize position + yaw (4 unknowns).

**First attempt:** Roll/pitch extracted via Euler angle formula — wrong convention, didn't match the rotation composition. Result: 96mm error.

**Second attempt:** Tilt extracted geometrically (find rotation mapping gravity vector). Full rotation = R_yaw × R_tilt.

| Mode | Position median | Orientation median |
|------|-----------------|-------------------|
| 6-DOF (no IMU) | **9.4 mm** | **31.8°** |
| 4-DOF + ideal IMU | 29.0 mm | 106° |

**Finding:** IMU constraint made it *worse* in the noiseless case. Constraining tilt and only solving for yaw makes the residual landscape more complex — the optimizer gets stuck. The full 6-DOF solver has more freedom to find the global minimum. IMU expected to help when noise limits the 6-DOF solver.

---

## Experiment 8: Cage scale (0.5m cube)

**Setup:** Coils at ±30cm, diameter 70cm, tracking volume ±20cm.

| Mode | Median | Max |
|------|--------|-----|
| 3-DOF, 3-period, 50% overlap | 13.1 mm | 24.1 mm |
| 6-DOF, 3-period | 26.2 mm | 125 mm |

**Finding:** Error roughly doubled going from 12cm to 50cm scale. Weaker gradients over the larger volume mean less spatial discrimination per unit of field measurement.

---

## Experiment 9: Coil geometry sweep (cage scale)

**Test:** Different asymmetric configurations, same coil frequencies.

| Config | Position median | Position max | Orient. median |
|--------|-----------------|-------------|----------------|
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

## Experiment 10: Multipass inversion

**Idea (from user):** Use first-pass trajectory to get velocity hints, improve second pass.

**Attempt 1 — shorter windows on second pass:**

| Approach | Estimates | Median |
|----------|-----------|--------|
| Single pass, 3-period | 132 | 13.1 mm |
| Multipass, 3p → 1p | 399 | 78.7 mm |
| Multipass, 3p → 0.5p | 832 | 88.2 mm |

**Failed:** Short windows have too much frequency cross-talk regardless of initialization quality.

**Attempt 2 — same window size, higher overlap:**

| Approach | Estimates | Median | Max |
|----------|-----------|--------|-----|
| Single, 3p, 50% overlap | 132 | 13.1 mm | 24.1 mm |
| Single, 3p, 75% overlap | 267 | 10.8 mm | 24.7 mm |
| Single, 3p, 90% overlap | 704 | 10.8 mm | 24.1 mm |
| Multipass, 3p, 90% overlap | 657 | 10.9 mm | 24.1 mm |

**Finding:** Higher overlap gives more estimates at slightly better accuracy. Multipass provides no accuracy improvement over single pass with the same overlap — the demodulation window is the bottleneck, not initialization.

---

## Summary of accuracy floors (noiseless)

| Scenario | Position (median) | Orientation (median) |
|----------|-------------------|---------------------|
| 3-DOF, small scale (12cm) | **3.3 mm** | n/a |
| 3-DOF, cage scale (50cm) | **10.9 mm** | n/a |
| 6-DOF, small scale (12cm) | **9.4 mm** | **31.8°** |
| 6-DOF, cage scale (50cm) | **26.2 mm** | **29.0°** |

## Key bottleneck

**Demodulation window duration.** At 100 Hz lowest frequency, a 3-period window = 30ms. At 10 cm/s bird speed, the sensor moves 3mm during the window. This motion blurs the demodulated measurement, setting a hard accuracy floor. Higher excitation frequencies would directly improve this.

## Next steps

1. **Noise injection** — sensor noise may dominate over the motion blur floor
2. **Higher frequencies** — e.g. 500/687/873 Hz → 6ms windows → 0.6mm motion blur
3. **More coil pairs** — 4th pair for better orientation conditioning
