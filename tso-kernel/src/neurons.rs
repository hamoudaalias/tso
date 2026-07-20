use ndarray::Array1;

#[derive(Clone)]
pub struct LIFNeuron {
    pub tau_m: f64,
    pub v_rest: f64,
    pub v_th: f64,
    pub v_reset: f64,
    pub v: f64,
    pub refractory: i32,
    pub spiked: bool,
}

impl LIFNeuron {
    pub fn new() -> Self {
        Self {
            tau_m: 10.0,
            v_rest: -65.0,
            v_th: -55.0,
            v_reset: -70.0,
            v: -65.0,
            refractory: 0,
            spiked: false,
        }
    }

    pub fn step(&mut self, i_syn: f64, dt: f64) -> f64 {
        if self.refractory > 0 {
            self.refractory -= 1;
            self.spiked = false;
            return 0.0;
        }
        let dv = (dt / self.tau_m) * (-(self.v - self.v_rest) + i_syn);
        self.v += dv;
        if self.v >= self.v_th {
            self.v = self.v_reset;
            self.refractory = 2;
            self.spiked = true;
            return 1.0;
        }
        self.spiked = false;
        0.0
    }

    pub fn reset(&mut self) {
        self.v = self.v_rest;
        self.refractory = 0;
        self.spiked = false;
    }
}

impl Default for LIFNeuron {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct LIFCluster {
    pub n: usize,
    pub label: String,
    pub tau_m: f64,
    pub v_rest: f64,
    pub v_th: f64,
    pub v_reset: f64,
    pub v: Array1<f64>,
    pub spikes: Array1<f64>,
    pub rate: f64,
    pub tau_rate: f64,
}

impl LIFCluster {
    pub fn new(n_neurons: usize) -> Self {
        Self {
            n: n_neurons,
            label: String::new(),
            tau_m: 10.0,
            v_rest: -65.0,
            v_th: -55.0,
            v_reset: -70.0,
            v: Array1::from_elem(n_neurons, -65.0),
            spikes: Array1::zeros(n_neurons),
            rate: 0.0,
            tau_rate: 50.0,
        }
    }

    pub fn with_label(n_neurons: usize, label: &str) -> Self {
        let mut c = Self::new(n_neurons);
        c.label = label.to_string();
        c
    }

    pub fn step(&mut self, i_syn: &Array1<f64>, dt: f64) -> f64 {
        let dv = (dt / self.tau_m) * (-(&self.v - self.v_rest) + i_syn);
        self.v += &dv;
        self.spikes.fill(0.0);
        for i in 0..self.n {
            if self.v[i] >= self.v_th {
                self.spikes[i] = 1.0;
                self.v[i] = self.v_reset;
            }
        }
        let inst_rate = self.spikes.mean().unwrap_or(0.0);
        let a = 1.0 - (-dt / self.tau_rate).exp();
        self.rate += a * (inst_rate - self.rate);
        inst_rate
    }

    pub fn reset(&mut self) {
        self.v.fill(self.v_rest);
        self.spikes.fill(0.0);
        self.rate = 0.0;
    }
}
