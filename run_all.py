"""Run the calculations used by the GRG manuscript."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = (
    "verify_obstruction.py",
    "verify_pg_israel_covariant_regularization.py",
    "verify_curvature_only_no_go.py",
    "verify_lapse_only_escape.py",
    "verify_profile_family_junction_benchmark.py",
    "verify_vlasov_constitutive_branch.py",
)


def main() -> int:
    root = HERE
    for name in SCRIPTS:
        path = root / "derivations" / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing reproduction script: {path}")
        print(f"\n=== {name} ===", flush=True)
        result = subprocess.run([sys.executable, str(path)], cwd=root,
                                check=False)
        if result.returncode:
            return result.returncode
    print("\nAll manuscript calculations completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
