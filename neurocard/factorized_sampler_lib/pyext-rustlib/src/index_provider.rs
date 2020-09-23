use crate::file_utils;
use ndarray::prelude::ArrayView1;
use numpy::{PyArray1, PyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

type KeyT = i64;
type IndexT = i64;
type WeightT = i64;
type KeyVecT = Vec<KeyT>;
type IndexVecT = Vec<IndexT>;

#[derive(Serialize, Deserialize, Debug)]
pub struct Index {
    map: HashMap<KeyVecT, (IndexVecT, Option<Vec<WeightT>>)>,
}

#[pyclass]
pub struct IndexProvider {
    index: HashMap<KeyVecT, (IndexVecT, Option<WeightedIndex<WeightT>>)>,
    null_index: IndexT,
    rng: SmallRng,
}

fn get_key_vec(pytuple: &PyTuple) -> PyResult<KeyVecT> {
    pytuple.iter().map(|pykey| pykey.extract()).collect()
}

fn get_indices_vec(pyarray: &PyArray1<IndexT>) -> IndexVecT {
    pyarray.as_array().iter().copied().collect()
}

fn get_weighted_index(weights: Option<&Vec<WeightT>>) -> Option<WeightedIndex<WeightT>> {
    if let Some(weights) = weights {
        if weights.len() <= 1 {
            None
        } else {
            WeightedIndex::new(weights).ok()
        }
    } else {
        None
    }
}

pub fn from_py(pydict: &PyDict, counts_pyarray: Option<&PyArray1<WeightT>>) -> PyResult<Index> {
    let counts: Option<ArrayView1<WeightT>> = counts_pyarray.map(|a| a.as_array());
    let map: PyResult<HashMap<_, _>> = pydict
        .iter()
        .map(|(pykey, pyvalue)| {
            let key_vec = pykey.extract().and_then(get_key_vec)?;
            let indices = pyvalue.extract().map(get_indices_vec)?;
            let weights = if let Some(counts) = counts {
                let weights = indices.iter().map(|&i| counts[i as usize]).collect();
                Some(weights)
            } else {
                None
            };
            Ok((key_vec, (indices, weights)))
        })
        .collect();
    let map = map?;
    let index = Index { map };
    Ok(index)
}

impl IndexProvider {
    fn _sample(&mut self, key_vec: &Vec<KeyT>) -> IndexT {
        if let Some((indices, weighted_index)) = self.index.get(key_vec) {
            if indices.len() == 1 {
                indices[0]
            } else {
                let mut rng = &mut self.rng;
                if let Some(weighted) = weighted_index {
                    indices[weighted.sample(&mut rng)]
                } else {
                    *indices.choose(&mut rng).unwrap()
                }
            }
        } else {
            self.null_index
        }
    }
}

#[pymethods]
impl IndexProvider {
    #[new]
    fn new(filename: &str, null_index: IndexT) -> PyResult<Self> {
        let saved_index: Index = file_utils::load(filename);
        let index = saved_index
            .map
            .iter()
            .map(|(key_vec, (indices, weights))| {
                (
                    key_vec.to_owned(),
                    (indices.to_owned(), get_weighted_index(weights.as_ref())),
                )
            })
            .collect();
        let rng = SmallRng::from_entropy();
        Ok(Self {
            index,
            null_index,
            rng,
        })
    }

    fn sample_indices(&mut self, keys_pyarray: &PyArrayDyn<KeyT>) -> Py<PyArray1<IndexT>> {
        let keys = keys_pyarray.as_array();
        let sample_vec: IndexVecT = keys
            .outer_iter()
            .map(|v| {
                let key_vec = v.iter().cloned().collect();
                self._sample(&key_vec)
            })
            .collect();
        let gil = pyo3::Python::acquire_gil();
        let ret = PyArray1::from_vec(gil.python(), sample_vec);
        ret.to_owned()
    }
}
