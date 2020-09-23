use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;

mod file_utils;
mod index_provider;

type WeightT = i64;

#[pyfunction]
fn prepare_indices(
    filename: &str,
    indices: &PyDict,
    counts: Option<&PyArray1<WeightT>>,
) -> PyResult<()> {
    if counts.is_none() {
        eprintln!(
            "{}: No counts provided, assuming uniform distribution",
            filename
        );
    }
    let index = index_provider::from_py(indices, counts)?;
    file_utils::save(filename, &index);
    println!("Saved indices to {}", filename);
    Ok(())
}

/// This module is a Python module implemented in Rust.
#[pymodule]
fn rustlib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<index_provider::IndexProvider>()?;
    m.add_wrapped(wrap_pyfunction!(prepare_indices))?;
    Ok(())
}
