# Magnesys Dual-Magnetometer Bird Tag — Schematic Design

This directory contains the schematic-level description of the
dual-magnetometer backpack tag developed in the inversion experiments
(see `docs/inversion_log.md`, Exps. 18-19). The design is the analog
TMR + AD8421 + nRF52832 SAADC chain with a 3 cm baseline between the
two magnetometers.

Files:

- `magnesys_dual_mag_tag.net` — KiCad netlist (S-expression `.net`
  format) listing all components and their net connections. This can
  be imported into KiCad's Pcbnew via *File → Import → Netlist*, or
  used as a reference when capturing the schematic in Eeschema.
- `magnesys_tag.kicad_sym` — KiCad symbol library with 6 custom
  schematic symbols: `AD8421`, `TLV9001`, `MIC5219-3.3YM5`,
  `NVE_AAH002` (single-axis analog GMR/TMR), `TMR_3AXIS_PLACEHOLDER`
  (generic 8-pin), and `nRF52832_MODULE` (13-pin module breakout
  with named pins).
- `README.md` (this file) — human-readable design description and BOM.

### Adding the symbol library to KiCad

1. *Preferences → Manage Symbol Libraries → Project Specific Libraries
   → Add an existing library to the table* (the folder icon).
2. Browse to `magnesys_tag.kicad_sym` and select it.
3. Give it a nickname (e.g., `magnesys_tag`).
4. Click *OK*. The 6 symbols become available under that nickname in
   the symbol chooser.

## High-level architecture

```
                     +--------- 3.7 V Li-ion battery (BT1)
                     |
                  +--+--+
                  | LDO | (U_LDO, 3.3 V)
                  +--+--+
                     |
       VCC = 3.3 V --+--------------+----------+----------+
                                    |          |          |
       REF = VCC/2 --+              |          |          |
                     |          +---+---+  +---+---+  ... (×8)
                     |          | TMR1  |  | AD8421|  ...
                     |          | (3-ax)|  | inamp |
       GND ----------+----------+---+---+  +---+---+  ...
                                |              |
                                |              v
                                | -- AC coup --> bias to REF --> +/-IN
                                |
                                +-- 3 cm baseline -- TMR2 same chain

       6 amp outputs --> R-C anti-alias --> nRF52832 SAADC inputs
                                            P0.02-0.05, P0.28-0.29
```

## Sections of the design

### Power

| Net  | Description                  |
|------|------------------------------|
| VBAT | Raw Li-ion (3.0–4.2 V)       |
| VCC  | LDO output, 3.3 V regulated  |
| GND  | Ground reference             |
| REF  | VCC/2, mid-supply for in-amp |

- `U_LDO` — 3.3 V LDO regulator (e.g., MIC5219-3.3 or ADP7142). Low
  quiescent current matters for battery life.
- `C_IN_LDO`, `C_OUT_LDO` — input and output bulk decoupling (1 µF
  ceramic each).
- `R_REF1`, `R_REF2` — divider from VCC to GND, 10 kΩ each, generates
  REF = VCC/2 ≈ 1.65 V.
- `C_REF` — 100 nF smoothing on the REF node.
- `U_BUF` — TLV9001 (or similar rail-to-rail single op-amp) wired as
  a unity-gain buffer to drive REF with low impedance. Without this,
  the REF node would be 5 kΩ Thevenin and would couple noise/offset
  through the in-amp REF pins.

### Magnetometers (×2)

`U_MAG1` and `U_MAG2` are 8-pin analog TMR 3-axis magnetometer dies
(e.g., NVE AAL024 series — replace with the exact part you choose). Pin
allocation:

| Pin | Net |
|-----|-----|
| 1 (VCC) | VCC |
| 2 (GND) | GND |
| 3 (X+)  | MAG1_XP / MAG2_XP |
| 4 (X−)  | MAG1_XN / MAG2_XN |
| 5 (Y+)  | MAG1_YP / MAG2_YP |
| 6 (Y−)  | MAG1_YN / MAG2_YN |
| 7 (Z+)  | MAG1_ZP / MAG2_ZP |
| 8 (Z−)  | MAG1_ZN / MAG2_ZN |

Decoupling: one 100 nF ceramic per sensor, placed close to its VCC
pin.

PCB layout note (not in netlist): MAG1 and MAG2 should be placed
**3.0 cm apart along the PCB +x axis**, with the same orientation. PCB
position tolerance ~0.1 mm is well below the inversion's sensitivity
to the offset.

### Per-channel analog frontend (×6)

Six identical channels, one per (sensor, axis): MAG1_X, MAG1_Y, MAG1_Z,
MAG2_X, MAG2_Y, MAG2_Z. For channel `n`:

| Designator       | Function           | Value     |
|------------------|--------------------|-----------|
| `C_AC{n}P`       | AC-coupling cap on +IN leg | 1 µF |
| `C_AC{n}N`       | AC-coupling cap on −IN leg | 1 µF |
| `R_BIAS{n}P`     | +IN bias to REF    | 100 kΩ    |
| `R_BIAS{n}N`     | −IN bias to REF    | 100 kΩ    |
| `U_AMP{n}`       | AD8421 in-amp      |           |
| `R_GAIN{n}`      | Gain resistor (RG) | 100 Ω → G ≈ 100 |
| `C_AMP{n}`       | Amp decoupling     | 100 nF    |
| `R_AA{n}`        | Anti-alias R       | 10 kΩ     |
| `C_AA{n}`        | Anti-alias C       | 1 nF      |

Signal chain per channel:

```
MAGn_(axis)P --[C_AC*P]--+-- AD8421 +IN
                         |
                         +--[R_BIAS*P]-- REF

MAGn_(axis)N --[C_AC*N]--+-- AD8421 -IN
                         |
                         +--[R_BIAS*N]-- REF

AD8421 RG (pins 2,3): R_GAIN
AD8421 REF (pin 5):   REF
AD8421 V+ (pin 7):    VCC (with C_AMP decoupling to GND)
AD8421 V- (pin 8):    GND
AD8421 OUT (pin 6) --[R_AA]--+-- to MCU ADC pin
                              |
                              +--[C_AA]-- GND
```

Gain calculation: AD8421 gain G = 1 + 9.9kΩ / RG. With RG = 100 Ω,
G ≈ 100. Adjust based on final sensor sensitivity and ADC range.

Anti-alias cutoff: f_c = 1 / (2π · R_AA · C_AA) = 1 / (2π · 10k · 1n)
≈ 16 kHz, comfortably below the nRF52832's 50 kSPS Nyquist of 25 kHz
and far above the 1 kHz signal band.

### nRF52832 connection

`J_MCU` is a header connector for an nRF52832 module (e.g., a
Raytac MDBT42Q breakout, or a custom-laid module). Pin assignment:

| Header pin | Net          | nRF52832 GPIO | Function                |
|-----------:|--------------|---------------|-------------------------|
|  1         | VCC          | (VDD)         | 3.3 V supply            |
|  2         | GND          | (VSS)         | Ground                  |
|  3         | ADC_M1X      | P0.02 / AIN0  | MAG1 X-axis amp out     |
|  4         | ADC_M1Y      | P0.03 / AIN1  | MAG1 Y-axis amp out     |
|  5         | ADC_M1Z      | P0.04 / AIN2  | MAG1 Z-axis amp out     |
|  6         | ADC_M2X      | P0.05 / AIN3  | MAG2 X-axis amp out     |
|  7         | ADC_M2Y      | P0.28 / AIN4  | MAG2 Y-axis amp out     |
|  8         | ADC_M2Z      | P0.29 / AIN5  | MAG2 Z-axis amp out     |
|  9         | CAL_BTN      | P0.06         | Calibration button (active low) |
| 10         | LED_STATUS   | P0.07         | Status LED (active high) |
| 11         | SWDIO        | (SWDIO)       | SWD data                |
| 12         | SWCLK        | (SWCLK)       | SWD clock               |
| 13         | nRESET       | (RESET)       | Module reset            |

The module breakout is assumed to handle its own crystal, antenna
matching, and BLE-side decoupling internally. If you're laying out the
nRF52832 chip directly rather than using a module, refer to the Nordic
reference design.

### User-interaction parts

- `SW_CAL` — momentary push-button, one side to CAL_BTN, other to GND.
  Internal pull-up on P0.06 used (configured in firmware).
- `LED1` + `R_LED` — status LED with 330 Ω current limit on
  LED_STATUS.

## Bill of Materials (consolidated)

| Qty | Designator(s)        | Description                              | Suggested part           |
|----:|----------------------|------------------------------------------|--------------------------|
|   1 | BT1                  | Li-ion battery connector (2-pin, 1.25 mm pitch) | Molex 53261 series       |
|   1 | U_LDO                | 3.3 V LDO, low Iq                        | MIC5219-3.3YM5 or ADP7142|
|   1 | U_BUF                | Single op-amp, rail-to-rail              | TLV9001IDBVR             |
|   2 | U_MAG1, U_MAG2       | 3-axis analog TMR magnetometer           | NVE AAL024 (verify pins) |
|   6 | U_AMP1..U_AMP6       | Instrumentation amplifier                | AD8421ARZ                |
|   2 | C_IN_LDO, C_OUT_LDO  | 1 µF / 16 V / X7R / 0603                 |                          |
|  ~9 | C_AMP1..6 + C_REF + C_VCC | 100 nF / 25 V / X7R / 0402         |                          |
|  12 | C_AC1P..6N           | 1 µF / 25 V / X7R / 0603                 | film if low-distortion needed |
|   6 | C_AA1..6             | 1 nF / 50 V / C0G / 0402                 |                          |
|   2 | R_REF1, R_REF2       | 10 kΩ / 0.1 % / 0402                     | matched pair             |
|  12 | R_BIAS1P..6N         | 100 kΩ / 1 % / 0402                      |                          |
|   6 | R_GAIN1..6           | 100 Ω / 0.1 % / 0402                     |                          |
|   6 | R_AA1..6             | 10 kΩ / 1 % / 0402                       |                          |
|   1 | R_LED                | 330 Ω / 5 % / 0402                       |                          |
|   1 | LED1                 | LED, 0603, any color                     |                          |
|   1 | SW_CAL               | SMT tactile button                       | C&K KMR-series           |
|   1 | J_MCU                | 13-pin 0.1″ header                       |                          |

Total parts ≈ 70.  Mass on a 30 × 30 mm 4-layer 0.6 mm PCB plus an
nRF52832 module: roughly 1.5 g excluding battery — well within the
budgerigar 5 % rule.

## Importing into KiCad

The netlist file is `magnesys_dual_mag_tag.net`. To use it:

1. **As a starting point for schematic capture:** open KiCad,
   create a new project, then in Eeschema use the netlist as a
   checklist while drawing — the `(comp ...)` blocks list every
   component you need, and the `(nets ...)` block tells you which
   pins connect together.
2. **Direct import to PCB editor:** in Pcbnew, *File → Import →
   Netlist* and select the file. Pcbnew will populate the board with
   footprints (you may need to assign footprints first via the netlist
   file itself or via a separate footprint-association step).
3. **As reference only:** read the file directly as a text netlist;
   the component and net structure is human-readable.

The TMR sensor symbol is referenced as a generic 8-pin part
(`Connector_Generic:Conn_01x08` style). When you have your final
sensor part chosen, replace it with the matching footprint and
symbol.

## Design caveats / next steps

1. **Reference noise.** The REF node carries the in-amp reference for
   all 6 channels. Any noise here couples into all outputs, so the
   buffered REF (via U_BUF) and a healthy bypass cap matter. If REF
   noise turns out to dominate, consider a dedicated low-noise voltage
   reference IC (e.g., LT6655 — bigger and pricier but lower noise).
2. **Sensor SET/RESET pins.** Most TMR magnetometers don't need
   SET/RESET (unlike AMR), but if your final sensor pick does, those
   pins are not yet wired in this design — add them in the schematic
   capture pass.
3. **ESD protection.** Not included in the netlist. For a wearable
   tag handled by people, add TVS diodes on the SWD pins and battery
   input.
4. **Battery management.** This design assumes the battery is charged
   externally. If you want USB charging, add an MCP73831 or similar.
5. **Footprint assignment.** The netlist suggests footprints (SOIC-8
   for AD8421, SOT-23-5 for the LDO, etc.) but you should verify
   against your specific part numbers before tape-out.
