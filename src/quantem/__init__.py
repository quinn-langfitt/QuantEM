"""QuantEM: Quantum Error Detection Compiler.

A Python library for automatically integrating quantum error detection
into quantum circuits using various QED protocols.
"""

import sys

# ==== rust stuff ==== #

from . import rust

# Globally define compiled submodules. The normal import mechanism will not find compiled submodules
# in _accelerate because it relies on file paths, but PyO3 generates only one shared library file.
# We manually define them on import so people can directly import chiplets.rust.* submodules
# and not have to rely on attribute access.  No action needed for top-level extension packages.
sys.modules["quantem.rust.sabre"] = rust.sabre

# ==== Public API ==== #

from .compiler import QEDCompiler, QEDStrategy, CompilationResult

__all__ = [
    "QEDCompiler",
    "QEDStrategy", 
    "CompilationResult",
]
