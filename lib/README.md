# lib/

Third-party source dependencies, cloned by `src/scripts/setup_lib.sh` (not committed).

| Dir | Upstream | Use |
|---|---|---|
| `PyMUST/` | https://github.com/creatis-ULTIM/PyMUST | SIMUS/PFIELD ultrasound simulation (installed editable) |
| `breastPhantom/` | https://github.com/DIDSR/breastPhantom | VICTRE anthropomorphic breast phantom generator (C++) |
| `breastMass/` | https://github.com/DIDSR/breastMass | VICTRE lesion generator (C++) |
| `VICTRE_PIPELINE/` | https://github.com/DIDSR/VICTRE_PIPELINE | VICTRE Python pipeline (reference) |

k-wave-python, PyTorch and the rest come from pip/conda (`environment.yml`).
