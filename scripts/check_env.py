from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = [
    "torch",
    "torchaudio",
    "numpy",
    "scipy",
    "matplotlib",
    "sklearn",
    "pytest",
]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")

    missing = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"{module_name}: {version}")
        except ImportError:
            print(f"{module_name}: missing")
            missing.append(module_name)

    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(f"Missing modules: {missing_text}")

    print("Environment looks ready.")


if __name__ == "__main__":
    main()
