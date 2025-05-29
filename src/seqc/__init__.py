import sys

# ==== rust stuff ==== #

from . import rust

# Globally define compiled submodules. The normal import mechanism will not find compiled submodules
# in _accelerate because it relies on file paths, but PyO3 generates only one shared library file.
# We manually define them on import so people can directly import chiplets.rust.* submodules
# and not have to rely on attribute access.  No action needed for top-level extension packages.
sys.modules["seqc.rust.sabre"] = rust.sabre

__all__ = []
