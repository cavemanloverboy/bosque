use bosque::{
    cast::{cast_slice, cast_slice_mut},
    float::{Sqrt, TreeFloat},
};
use numpy::{
    ndarray::{Array2, ArrayViewMut2},
    IntoPyArray, PyArray2, PyReadwriteArray1,
};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn pybosque(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Tree>()?;

    #[pyclass]
    pub struct Tree {
        /// Memory address of data array start
        pub start: usize,

        /// Number of elements
        pub size: usize,

        /// Type of element stored in buffer -- f32, f64, cp32
        pub mode: Mode,
    }

    impl Tree {
        unsafe fn get_data<T: TreeFloat>(&self) -> &'static [[T; 3]] {
            core::slice::from_raw_parts(self.start as *const [T; 3], self.size)
        }
    }

    pub fn build_tree<T: TreeFloat>(
        mut data: ArrayViewMut2<'_, T>,
        mut indices: Option<PyReadwriteArray1<u32>>,
    ) -> PyResult<Tree> {
        if data.ncols() != 3 {
            return Err(PyValueError::new_err("Only 3D is supported"));
        }

        // Get reference to array if contiguous
        let Some(mut data) = data.as_slice_mut() else {
            return Err(PyValueError::new_err("data array is not in C order"));
        };
        // Convert to slice of arrays
        // SAFETY: alingnment of T is same as [T; 3]
        let data: &mut [[T; 3]] = cast_slice_mut(&mut data);

        // Get or generate indices
        if let Some(ref mut indicies) = indices {
            if let Ok(idxs) = indicies.as_slice_mut() {
                println!("building with indices");
                bosque::tree::build_tree_with_indices::<T>(data, idxs);
            } else {
                return Err(PyValueError::new_err("index array is not in C order"));
            }
        } else {
            bosque::tree::build_tree(data);
        };

        let tree = match T::NAME {
            "CP32" => {
                let data = data;
                Tree {
                    start: data.as_ptr() as usize,
                    size: data.len(),
                    mode: Mode::CP32,
                }
            }
            "F32" => {
                let data = data;
                Tree {
                    start: data.as_ptr() as usize,
                    size: data.len(),
                    mode: Mode::F32,
                }
            }
            "F64" => {
                let data = data;
                Tree {
                    start: data.as_ptr() as usize,
                    size: data.len(),
                    mode: Mode::F64,
                }
            }
            _ => unreachable!("only TreeFloat types can be passed in"),
        };

        Ok(tree)
    }

    #[pymethods]
    impl Tree {
        #[new]
        pub fn build<'a>(
            data: PyObject,
            indices: Option<PyReadwriteArray1<u32>>,
            py: Python<'a>,
        ) -> PyResult<Tree> {
            if let Ok(data) = data.downcast::<PyArray2<f64>>(py) {
                if data.dims()[1] != 3 {
                    return Err(PyValueError::new_err("only 3D is supported"));
                }
                return Ok(build_tree(unsafe { data.as_array_mut() }, indices)?);
            }

            if let Ok(data) = data.downcast::<PyArray2<f32>>(py) {
                if data.dims()[1] != 3 {
                    return Err(PyValueError::new_err("only 3D is supported"));
                }
                return Ok(build_tree(unsafe { data.as_array_mut() }, indices)?);
            }

            // if let Ok(data) = data.downcast::<PyArray2<CP32>>(py) {
            //     return Ok(build_tree(data.as_array_mut(), None)?);
            // }

            Err(PyValueError::new_err(
                "data was not a f64, f32, or cp32 array",
            ))
        }

        pub fn query<'a>(
            &self,
            query: PyObject,
            k: PyObject,
            boxsize: Option<[f64; 2]>,
            py: Python<'a>,
        ) -> PyResult<(PyObject, PyObject)> {
            // Parse k as contiguous or sparse set
            let k = KNN::from_py(k.as_ref(py))?;
            let num_k = k.num_k();
            let max_k = k.max();

            match self.mode {
                Mode::F64 => {
                    if let Ok(query) = query.downcast::<PyArray2<f64>>(py) {
                        // Get slice of tree data
                        // SAFETY: downcast + mode = checked type
                        let tree = unsafe { self.get_data::<f64>() };

                        // Check and recast query data
                        if let Ok(query) = unsafe { query.as_slice() } {
                            // Cast single element slice to slice of arays
                            let query: &[[f64; 3]] = cast_slice(query);

                            // Preallocate
                            let dist_sq: Vec<f64> = Vec::with_capacity(k.num_k() * query.len());
                            let indices: Vec<usize> = Vec::with_capacity(k.num_k() * query.len());
                            // Create raw mutable pointers
                            let dist_sq_ptr: usize = dist_sq.as_ptr() as usize;
                            let indices_ptr: usize = indices.as_ptr() as usize;

                            // Parallel iteration through each query
                            query.into_par_iter().enumerate().for_each(|(i, q)| {
                                // Compute nearest neighbors
                                let mut result_bh = if let Some([lo, hi]) = boxsize {
                                    bosque::tree::nearest_k_periodic_bh(tree, q, max_k, lo, hi)
                                } else {
                                    bosque::tree::nearest_k_bh(tree, q, max_k)
                                };

                                // Calculate offset for this thread's output
                                let mut offset = i * num_k + num_k;

                                // Step 4: Directly write results into the subsections of the preallocated vectors
                                let mut current_k = max_k;
                                while let Some((dist_sq, index)) = result_bh.pop() {
                                    if k.contained(current_k) {
                                        offset -= 1;
                                        unsafe {
                                            let dist_sq_dst: *mut f64 =
                                                (dist_sq_ptr as *mut f64).add(offset);
                                            let indices_dst: *mut usize =
                                                (indices_ptr as *mut usize).add(offset);

                                            *dist_sq_dst = dist_sq.sqrt().0;
                                            *indices_dst = index;
                                        }
                                    }
                                    current_k -= 1;
                                }
                            });
                            // println!("queried in {} millis", timer.elapsed().as_millis());

                            let shape = (query.len(), num_k);
                            unsafe {
                                return Ok((
                                    Array2::from_shape_vec_unchecked(shape, dist_sq)
                                        .into_pyarray(py)
                                        .into(),
                                    Array2::from_shape_vec_unchecked(shape, indices)
                                        .into_pyarray(py)
                                        .into(),
                                ));
                            }
                        } else {
                            return Err(PyValueError::new_err(
                                "query array is not contiguous C order",
                            ));
                        }
                    }
                }
                Mode::F32 => {
                    if let Ok(query) = query.downcast::<PyArray2<f32>>(py) {
                        // Get slice of tree data
                        // SAFETY: downcast + mode = checked type
                        let tree = unsafe { self.get_data::<f32>() };

                        // Check and recast query data
                        if let Ok(query) = unsafe { query.as_slice() } {
                            // Cast single element slice to slice of arays
                            let query: &[[f32; 3]] = cast_slice(query);

                            // Preallocate
                            let dist_sq: Vec<f32> = Vec::with_capacity(num_k * query.len());
                            let indices: Vec<usize> = Vec::with_capacity(num_k * query.len());
                            // Create raw mutable pointers
                            let dist_sq_ptr: usize = dist_sq.as_ptr() as usize;
                            let indices_ptr: usize = indices.as_ptr() as usize;

                            // Parallel iteration through each query
                            query.into_par_iter().enumerate().for_each(|(i, q)| {
                                // Compute nearest neighbors
                                let mut result_bh = if let Some([lo, hi]) = boxsize {
                                    bosque::tree::nearest_k_periodic_bh(
                                        tree, q, max_k, lo as f32, hi as f32,
                                    )
                                } else {
                                    bosque::tree::nearest_k_bh(tree, q, max_k)
                                };

                                // Calculate offset for this thread's output
                                let mut offset = i * num_k + num_k;

                                // Directly write results into the subsections of the preallocated vectors
                                let mut current_k = max_k;
                                while let Some((dist_sq, index)) = result_bh.pop() {
                                    if k.contained(current_k) {
                                        offset -= 1;
                                        unsafe {
                                            let dist_sq_dst: *mut f32 =
                                                (dist_sq_ptr as *mut f32).add(offset);
                                            let indices_dst: *mut usize =
                                                (indices_ptr as *mut usize).add(offset);

                                            *dist_sq_dst = dist_sq.sqrt().0;
                                            *indices_dst = index;
                                        }
                                    }
                                    current_k -= 1;
                                }
                            });
                            // println!("queried in {} millis", timer.elapsed().as_millis());

                            let shape = (query.len(), num_k);
                            unsafe {
                                return Ok((
                                    Array2::from_shape_vec_unchecked(shape, dist_sq)
                                        .into_pyarray(py)
                                        .into(),
                                    Array2::from_shape_vec_unchecked(shape, indices)
                                        .into_pyarray(py)
                                        .into(),
                                ));
                            }
                        } else {
                            return Err(PyValueError::new_err(
                                "query array is not contiguous C order",
                            ));
                        }
                    }
                }

                Mode::CP32 => return Err(PyValueError::new_err("not implemented for cp32 yet")),
            }

            // if let Ok(data) = query.downcast::<PyArray2<f32>>(py) {
            //     return Ok(build_tree(unsafe { data.as_array_mut() }, None)?);
            // }

            // if let Ok(data) = data.downcast::<PyArray2<CP32>>(py) {
            //     return Ok(build_tree(data.as_array_mut(), None)?);
            // }

            Err(PyValueError::new_err("data/query type mismatch"))
        }

        pub fn print(&self) {
            match &self.mode {
                Mode::CP32 => {
                    // let cp32_slice = unsafe { self.mode.get_ref::<CPf32(start, size) };
                }
                Mode::F32 => {
                    let f32_slice = unsafe { Mode::get_ref::<[f32; 3]>(self.start, self.size) };
                    println!("{f32_slice:?}");
                }
                Mode::F64 => {
                    let f64_slice = unsafe { Mode::get_ref::<[f64; 3]>(self.start, self.size) };
                    println!("{f64_slice:?}");
                }
            }
        }

        pub fn num_threads(&self) {
            println!("using {} threads for query", rayon::current_num_threads());
        }
    }

    Ok(())
}

pub enum Mode {
    CP32,
    F32,
    F64,
}

impl Mode {
    /// TODO: safety...
    unsafe fn get_ref<T>(start: usize, size: usize) -> &'static [T] {
        core::slice::from_raw_parts(start as *const T, size)
    }
}

trait IntoMode {
    fn into_mode(self) -> Result<Mode, &'static str>;
}

impl IntoMode for &str {
    fn into_mode(self) -> Result<Mode, &'static str> {
        match self.to_lowercase().as_str() {
            "cp32" | "abacus" => Ok(Mode::CP32),
            "f32" | "float" | "single" => Ok(Mode::F32),
            "f64" | "double" => Ok(Mode::F64),
            _ => Err("only cp32/abacus, f32/float/single, f64/double are supported"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum KNN {
    Contiguous(usize),
    Sparse(Vec<usize>),
}

impl KNN {
    #[inline(always)]
    fn from_py(object: &PyAny) -> PyResult<KNN> {
        if let Ok(k) = <usize as FromPyObject>::extract(object) {
            if k > 0 {
                return Ok(KNN::Contiguous(k));
            }
        }

        if let Ok(ks) = <Vec<usize> as FromPyObject>::extract(object) {
            let non_empty = ks.len() > 0;
            let non_zero = ks.iter().all(|&k| k > 0);
            if non_empty & non_zero {
                return Ok(KNN::Sparse(ks));
            }
        }

        Err(PyValueError::new_err(
            "k needs to be a nonzero integer or nonempty list of nonzero integers",
        ))
    }

    #[inline(always)]
    fn num_k(&self) -> usize {
        match self {
            &KNN::Contiguous(k) => k,
            KNN::Sparse(s) => s.len(),
        }
    }

    #[inline(always)]
    fn max(&self) -> usize {
        match self {
            &KNN::Contiguous(k) => k,
            KNN::Sparse(s) => s
                .iter()
                .max()
                .copied()
                .expect("upon construction, nonemptiness is checked"),
        }
    }

    #[inline(always)]
    fn contained(&self, k: usize) -> bool {
        match self {
            // Technically, this should be k <= _k, but in the normal construction
            // and use of a `KNN` the input should always be k <= _k.
            KNN::Contiguous(_k) => true,

            KNN::Sparse(ks) => ks.contains(&k),
        }
    }
}
