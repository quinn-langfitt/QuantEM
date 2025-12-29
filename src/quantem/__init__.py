'''Copyright © 2025 UChicago Argonne, LLC and Northwestern University All right reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    https://github.com/quinn-langfitt/QuantEM/blob/main/LICENSE.txt

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''

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
