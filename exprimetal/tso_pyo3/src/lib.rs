use ndarray::Array1;
use pyo3::prelude::*;
use tso_engine::action::ActionMotor;
use tso_engine::attractor::AttractorField;
use tso_engine::episodic::{ContextBuffer, EpisodicMemory};
use tso_engine::memory::AssociativeMemory;
use tso_engine::neurons::{DualLIFState, LIFState};
use tso_engine::working_memory::WorkingMemory;

fn vec_to_array(v: Vec<f64>) -> Array1<f64> {
    Array1::from_vec(v)
}

fn array_to_vec(a: &Array1<f64>) -> Vec<f64> {
    a.iter().copied().collect()
}

#[pyclass(name = "LIFState", module = "tso_pyo3")]
struct PyLIFState {
    inner: LIFState,
}

#[pymethods]
impl PyLIFState {
    #[new]
    fn new(dim: usize, alpha: f64) -> Self {
        PyLIFState {
            inner: LIFState::new(dim, alpha),
        }
    }

    fn step(&mut self, embedding: Vec<f64>, negate: bool) {
        self.inner.step(&vec_to_array(embedding), negate);
    }

    fn get_state(&self) -> Vec<f64> {
        array_to_vec(&self.inner.state)
    }
}

#[pyclass(name = "DualLIFState", module = "tso_pyo3")]
struct PyDualLIFState {
    inner: DualLIFState,
}

#[pymethods]
impl PyDualLIFState {
    #[new]
    fn new(dim: usize, alpha_slow: f64, alpha_fast: f64) -> Self {
        PyDualLIFState {
            inner: DualLIFState::new(dim, alpha_slow, alpha_fast),
        }
    }

    fn step(&mut self, embedding: Vec<f64>, negate: bool) {
        self.inner.step(&vec_to_array(embedding), negate);
    }

    fn alignment(&self, embedding: Vec<f64>, beta: f64) -> f64 {
        self.inner.alignment(&vec_to_array(embedding), beta)
    }

    fn get_slow_state(&self) -> Vec<f64> {
        array_to_vec(&self.inner.slow.state)
    }

    fn get_fast_state(&self) -> Vec<f64> {
        array_to_vec(&self.inner.fast.state)
    }
}

#[pyclass(name = "AttractorField", module = "tso_pyo3")]
struct PyAttractorField {
    inner: AttractorField,
}

#[pymethods]
impl PyAttractorField {
    #[new]
    fn new(dim: usize, n_classes: usize, k: usize, lr: f64) -> Self {
        PyAttractorField {
            inner: AttractorField::new(dim, n_classes, k, lr),
        }
    }

    fn predict(&self, state: Vec<f64>) -> usize {
        self.inner.predict(&vec_to_array(state))
    }

    fn predict_with_distance(&self, state: Vec<f64>) -> (usize, f64) {
        self.inner.predict_with_distance(&vec_to_array(state))
    }

    fn train_step(&mut self, state: Vec<f64>, true_label: usize) {
        self.inner.train_step(&vec_to_array(state), true_label);
    }

    fn add_class(&mut self, example: Vec<f64>) -> usize {
        self.inner.add_class(&vec_to_array(example))
    }

    fn add_prototype(&mut self, example: Vec<f64>, class: usize) {
        self.inner.add_prototype(&vec_to_array(example), class);
    }

    fn n_classes(&self) -> usize {
        self.inner.n_classes()
    }

    fn get_prototypes(&self) -> Vec<Vec<Vec<f64>>> {
        self.inner
            .prototypes
            .iter()
            .map(|class| class.iter().map(|p| array_to_vec(p)).collect())
            .collect()
    }
}

#[pyclass(name = "Graph", module = "tso_pyo3")]
struct PyGraph {
    inner: tso_engine::core::Graph,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> Self {
        PyGraph {
            inner: tso_engine::core::Graph::new(),
        }
    }

    fn add_node(&mut self, z: Vec<f64>) -> usize {
        self.inner.add_node(vec_to_array(z))
    }

    fn add_edge(&mut self, from: usize, to: usize, weight: i8) {
        self.inner.add_edge(from, to, weight);
    }

    fn phi(&self) -> f64 {
        self.inner.phi()
    }

    fn node_count(&self) -> usize {
        self.inner.nodes.len()
    }

    fn edge_count(&self) -> usize {
        self.inner.edges.len()
    }

    fn get_nodes(&self) -> Vec<Vec<f64>> {
        self.inner.nodes.iter().map(|n| array_to_vec(n)).collect()
    }

    fn get_edges(&self) -> Vec<(usize, usize, i8)> {
        self.inner
            .edges
            .iter()
            .map(|e| (e.from, e.to, e.weight))
            .collect()
    }

    fn edge_weight(&self, a: usize, b: usize) -> Option<i8> {
        self.inner.edge_weight(a, b)
    }

    fn add_transition(&mut self, from: Vec<f64>, to: Vec<f64>, reward: f64) -> (usize, usize) {
        self.inner.add_transition(&vec_to_array(from), &vec_to_array(to), reward)
    }
}

#[pyclass(name = "EpisodicMemory", module = "tso_pyo3")]
struct PyEpisodicMemory {
    inner: EpisodicMemory,
}

#[pymethods]
impl PyEpisodicMemory {
    #[new]
    fn new(max_episode_len: usize) -> Self {
        PyEpisodicMemory {
            inner: EpisodicMemory::new(max_episode_len),
        }
    }

    fn store(&mut self, sequence: Vec<usize>) {
        self.inner.store(&sequence);
    }

    fn recall(&self, context: Vec<usize>) -> Option<usize> {
        self.inner.recall(&context)
    }
}

#[pyclass(name = "ContextBuffer", module = "tso_pyo3")]
struct PyContextBuffer {
    inner: ContextBuffer,
}

#[pymethods]
impl PyContextBuffer {
    #[new]
    fn new(max_len: usize) -> Self {
        PyContextBuffer {
            inner: ContextBuffer::new(max_len),
        }
    }

    fn push(&mut self, word: usize) {
        self.inner.push(word);
    }

    fn as_slice(&self) -> Vec<usize> {
        self.inner.as_slice()
    }
}

#[pyclass(name = "AssociativeMemory", module = "tso_pyo3")]
struct PyAssociativeMemory {
    inner: AssociativeMemory,
}

#[pymethods]
impl PyAssociativeMemory {
    #[new]
    fn new() -> Self {
        PyAssociativeMemory {
            inner: AssociativeMemory::new(),
        }
    }

    fn store(&mut self, vector: Vec<f64>, data: usize) {
        self.inner.store(&vec_to_array(vector), data);
    }

    fn recall(&self, query: Vec<f64>) -> Option<usize> {
        self.inner.recall(&vec_to_array(query))
    }

    fn recall_with_sim(&self, query: Vec<f64>) -> Option<(usize, f64)> {
        self.inner.recall_with_sim(&vec_to_array(query))
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

#[pyclass(name = "WorkingMemory", module = "tso_pyo3")]
struct PyWorkingMemory {
    inner: WorkingMemory,
}

#[pymethods]
impl PyWorkingMemory {
    #[new]
    fn new(dim: usize, alpha_slow: f64, alpha_fast: f64) -> Self {
        PyWorkingMemory {
            inner: WorkingMemory::new(dim, alpha_slow, alpha_fast),
        }
    }

    fn observe(&mut self, objects: Vec<Vec<f64>>) -> Option<(usize, f64)> {
        let arrs: Vec<Array1<f64>> = objects.into_iter().map(vec_to_array).collect();
        self.inner.observe(&arrs)
    }

    fn recall(&self, query: Vec<f64>) -> Option<(usize, f64)> {
        self.inner.recall(&vec_to_array(query))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn store(&mut self, vector: Vec<f64>, data: usize) {
        self.inner.store(&vec_to_array(vector), data);
    }

    fn has_target(&self) -> bool {
        self.inner.has_target()
    }
}

#[pyclass(name = "ActionMotor", module = "tso_pyo3")]
struct PyActionMotor {
    inner: ActionMotor,
}

#[pymethods]
impl PyActionMotor {
    #[new]
    fn new(beta: f64) -> Self {
        PyActionMotor {
            inner: ActionMotor::new(beta),
        }
    }

    fn select(&self, context: &PyDualLIFState, actions: Vec<Vec<f64>>) -> (usize, f64) {
        let arrs: Vec<Array1<f64>> = actions.into_iter().map(vec_to_array).collect();
        self.inner.select(&context.inner, &arrs)
    }

    fn select_with_bonus(
        &self,
        context: &PyDualLIFState,
        actions: Vec<Vec<f64>>,
        bonuses: Vec<f64>,
    ) -> (usize, f64) {
        let arrs: Vec<Array1<f64>> = actions.into_iter().map(vec_to_array).collect();
        self.inner.select_with_bonus(&context.inner, &arrs, &bonuses)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLIFState>()?;
    m.add_class::<PyDualLIFState>()?;
    m.add_class::<PyAttractorField>()?;
    m.add_class::<PyGraph>()?;
    m.add_class::<PyEpisodicMemory>()?;
    m.add_class::<PyContextBuffer>()?;
    m.add_class::<PyAssociativeMemory>()?;
    m.add_class::<PyWorkingMemory>()?;
    m.add_class::<PyActionMotor>()?;
    Ok(())
}
