pub mod sabre;

use pyo3::prelude::*;

#[inline(always)]
#[doc(hidden)]
fn add_submodule<F>(m: &Bound<PyModule>, constructor: F, name: &str) -> PyResult<()>
where
    F: FnOnce(&Bound<PyModule>) -> PyResult<()>,
{
    let new_mod = PyModule::new(m.py(), name)?;
    constructor(&new_mod)?;
    m.add_submodule(&new_mod)
}

#[pymodule]
pub fn rust(m: &Bound<PyModule>) -> PyResult<()> {
    add_submodule(m, sabre::sabre, "sabre")?;
    Ok(())
}

