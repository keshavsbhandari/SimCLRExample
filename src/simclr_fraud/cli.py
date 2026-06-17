"""Console entry for the pip-installed ``simclr-run`` command."""

import runpy
import sys
from pathlib import Path


def entrypoint() -> None:
    """Delegate to ``main.py`` in the repository root (same as ``python main.py``).

    Resolves ``main.py`` relative to the installed package location. Requires
    editable install from repo root so ``configs/`` is discoverable by Hydra.

    Raises:
        SystemExit: If ``main.py`` cannot be found next to the package root.
    """
    main_py = Path(__file__).resolve().parents[2] / "main.py"
    if not main_py.is_file():
        raise SystemExit(
            f"Expected entry point at {main_py}. "
            "Install editable from repo root or run `python main.py`."
        )
    sys.argv[0] = str(main_py)
    runpy.run_path(str(main_py), run_name="__main__")
