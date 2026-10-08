# Mechanical mediation through shared apparatus in gravitational-entanglement experiments

Read the [final manuscript](paper/mechanical_mediation.pdf). This research manuscript is
published on [Zenodo](https://doi.org/10.5281/zenodo.23245274), released with its calculations and supporting material.

## Release and verification

The bundled LaTeX and plot data reproduce the final manuscript. On October 8,
2026, a fresh three-pass build produced `paper/main.pdf` with the same text and
page rendering as all 13 pages of the published PDF. PDF timestamps and document
identifiers may differ; the published PDF is preserved byte for byte.

See the [source reproduction record](verification/source-reproduction.json).

On October 4, 2026, `make reproduce` passed in a fresh validation copy using
Python 3.12.3 and pdfTeX 1.40.25 (TeX Live 2023/Debian). 57 unit tests and both outward-arithmetic interval checks passed.
The final LaTeX pass had no unresolved references, warnings, or overfull or
underfull boxes. There are no Lean sources or Lake projects in this bundle.
These checks concern the supplied source and finite examples; they do not
verify every argument in the final PDF.

See the [October 4 validation record](verification/release.json) and
[execution log](verification/build.log) for the earlier source bundle.

Samuel Mausberg, Independent Researcher

This repository accompanies the manuscript above. The source is
`paper/main.tex`, with proofs in `paper/appendices.tex`. All four figures are
native TikZ or pgfplots. No external figure PDFs, proprietary fonts,
or network requests are required to compile the paper.

The paper derives a mechanical-channel certificate from two directed force-port
actions, a force-replay error budget and local null, and finite-time passive
compliance bounds. It retains the exact free-support phase and causal elastic
example. The horizontal clamped-link calculation is a specified design model,
not a measurement of the constrained proposal or an arbitrary mounted table.

## Build the paper

From this directory:

```sh
./build.sh
```

The output is `paper/main.pdf`. This uses `paper/compliance.dat` and
`paper/null_pulses.dat` and does not require Python. A TeX distribution must
provide REVTeX 4.2, amsmath, amssymb, bm, amsthm, graphicx, booktabs, microtype,
TikZ, pgfplots, hyperref, and T1 Computer Modern fonts (CM-Super). The verified
build used pdfTeX 1.40.25 with TeX Live 2023/Debian. The script checks for
unresolved references, LaTeX warnings, and overfull boxes after the last pass.
Underfull-box notices are retained in the log. Logs are written to `results/`.

## Reproduce the calculations and checks

The recorded Python environment was Python 3.13.5. The pinned dependencies
require Python 3.11 or later. A virtual environment is recommended.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
make reproduce
```

Individual steps are:

```sh
python code/calculate.py
python -m unittest discover -s code -p 'test_*.py' -v
python code/design_interval.py
python code/interval_check.py
./build.sh
```

`calculate.py` writes `results/results.json` and the two pgfplots CSV files in
`data/`. It evaluates the gravitational reference trajectories, free capsule,
force-replay example, local response null, horizontal link, thermal covariance,
and the causal free-ended elastic example. `model.py` supplies the scaled causal
initial-value calculation and Gaussian channel. `elastic.py` supplies the exact
polynomial-segment oscillator calculation and elastic bounds. `design.py`
supplies the static Green function and passive compliance estimates.

The interval programs are separate checks. `design_interval.py` encloses the
explicit short-pulse phase and phase-ratio inequalities with outward arithmetic.
`interval_check.py` independently propagates polynomial segments with interval
arithmetic to check the two-response null and its rounding residual. The former
uses 65 decimal places and the latter 70. These checks concern the stipulated
model parameters, not fabrication tolerances or measurement errors.

The tests include independent oscillator ODE checks, force moments and exact
polynomial norms, Gaussian-channel positivity and spectator-system distances,
static Green-function integrals, mode sums, passive inequalities, the finite-gap
identity, causality, source-load coordinates, and the response null. They are
executed consistency checks, not a machine-checked proof of the manuscript.

## Inputs and interpretation

Every published parameter and source is identified in the paper, principally
Sec. IV and Appendices C and E. Geometries, ramp choices, force-port profiles,
material values, and measured reference parameters are not interchangeable.
The vertical-heave reference parameters, including the stated 187 kg suspended
mass, appear only as a numerical control example. They are not treated as the
horizontal susceptibility of a constrained device.

The stainless-steel polynomial is evaluated at 293 K from the cited NIST fit.
The link dimensions, density, boundary conditions, force patches, and thermal
state are explicitly chosen model inputs. The all-mode phase bounds apply to
that longitudinal model. A physical instrument also requires its other load
paths, clamp motion, bending, and control response to be included or calibrated.

The small-splitting gravity calculations use prescribed Newtonian pair paths.
They are not a total error budget for gravity, surface forces, preparation,
readout, or the apparatus as built. The mechanical certificate does not assume
those errors vanish. The room-temperature short-pulse link calculation includes
a covariance large enough to mask the small reference witness; it is a phase
and design-budget comparison, not an experimental entanglement prediction.

## License and citation

The manuscript and original research material use [CC BY 4.0](LICENSES/CC-BY-4.0.txt).
Original code uses [MIT](LICENSES/MIT.txt). Third-party notices remain in effect.
See [LICENSE](LICENSE) for the scope and [CITATION.cff](CITATION.cff) for citation metadata.
