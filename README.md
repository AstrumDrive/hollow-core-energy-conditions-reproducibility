# Reproducibility material for a static spherical hollow-source construction

This repository contains the minimal Python source, machine-readable outputs,
and figure files needed to reproduce the calculations reported in:

**Boundary Constraints and Lapse Freedom in Static Spherical Hollow Sources**

Authors: Nelson Bolívar, Gabriel Abellán, and Ivaylo Vasilev.

The calculations cover the unit-lapse obstruction, the covariant
PG--Israel regularization check, curvature-only closure, the nontrivial-lapse
family, the profile/junction benchmark, and the finite Einstein--Vlasov
constitutive companion. The repository does not contain experimental,
stability, formation, confinement, or transport models.

## Reproduce the reported calculations

Use a clean Python 3.12 environment:

```text
python -m venv .venv
python -m pip install -r requirements.txt
python run_all.py
```

The scripts expect the `derivations/`, `results/`, and `figures/` directories
to retain their relative positions. Each calculation exits with a nonzero
status if one of its stated checks fails. The Einstein--Vlasov branch is the
longest calculation.

The file `MANIFEST.sha256` records SHA-256 checksums for the reproducibility
files. The figures are provided as PDF files so that they remain legible in
print and grayscale workflows.

## Scope

This archive is a computational companion to the theoretical manuscript. It
is not a claim of experimental realization or of a transport-capable warp
drive.
