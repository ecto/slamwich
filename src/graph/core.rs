//! The factor graph, its Levenberg-Marquardt solve, and marginalisation.
//!
//! Variables are plain `f64` blocks (a pose, a tag, a body parameter). A
//! factor reads the blocks it names and writes a whitened residual as tang
//! expressions; [`Graph::linearize`] records it on a fresh tape and takes
//! one reverse sweep per residual row for the Jacobian. A variable can be
//! *fixed*: online, body-card parameters are fixed inputs; overnight the same
//! graph is re-solved with them free. Nothing else changes between the two.

use std::collections::BTreeMap;
use std::sync::Arc;

use nalgebra::{DMatrix, DVector};
use tang_ad::{Tape, Var};

/// Handle to a variable block.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct Key(pub u64);

/// What a variable is, for wrapping and for readers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Kind {
    /// `[x, y, z, roll, pitch, yaw]`.
    Pose,
    /// A free vector (body parameters, biases).
    Vector,
    /// A caller-defined block (object state, a particle summary); the
    /// solver treats it as a plain vector.
    Other(u16),
}

/// A variable block.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Variable {
    pub kind: Kind,
    pub value: Vec<f64>,
    /// Held constant by the solve (body-card parameters online, gauge anchors).
    pub fixed: bool,
}

/// Robust loss on a factor's whitened residual norm.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Robust {
    None,
    /// Quadratic inside `k` (in σ), linear outside.
    Huber(f64),
}

/// A term in the cost: ½ ρ(‖r(x)‖²) with r already whitened.
pub trait Factor {
    /// The variable blocks this factor reads, in the order `residual` sees them.
    fn keys(&self) -> &[Key];
    /// Whitened residual of the concatenated blocks, as tape expressions.
    fn residual(&self, tape: &Arc<Tape>, x: &[Var]) -> Vec<Var>;
    /// Robust loss; outlier-prone factors (loop closures, tags) use Huber.
    fn robust(&self) -> Robust {
        Robust::None
    }
    /// Short name for diagnostics.
    fn name(&self) -> &'static str;
}

/// One factor linearised: residual and Jacobian columns per key, after the
/// robust weight.
pub struct Linear {
    pub keys: Vec<Key>,
    pub r: DVector<f64>,
    pub j: DMatrix<f64>,
    pub cost: f64,
}

/// Linearise one factor at `vars`. Inputs go on the tape first, so their
/// adjoints are entries `0..n` of each reverse sweep.
pub fn linearize(f: &dyn Factor, vars: &BTreeMap<Key, Variable>) -> Linear {
    let tape = Tape::new();
    let mut inputs = Vec::new();
    for k in f.keys() {
        for &v in &vars[k].value {
            inputs.push(tape.var(v));
        }
    }
    let n = inputs.len();
    let res = f.residual(&tape, &inputs);
    let m = res.len();
    let mut r = DVector::zeros(m);
    let mut j = DMatrix::zeros(m, n);
    for (row, ri) in res.iter().enumerate() {
        r[row] = ri.value();
        let g = ri.backward();
        for col in 0..n {
            j[(row, col)] = g[col];
        }
    }
    let sq = r.norm_squared();
    let (cost, w) = match f.robust() {
        Robust::None => (0.5 * sq, 1.0),
        Robust::Huber(k) => {
            let norm = sq.sqrt();
            if norm <= k {
                (0.5 * sq, 1.0)
            } else {
                (k * norm - 0.5 * k * k, k / norm)
            }
        }
    };
    if w != 1.0 {
        let s = w.sqrt();
        r *= s;
        j *= s;
    }
    Linear {
        keys: f.keys().to_vec(),
        r,
        j,
        cost,
    }
}

/// Solver settings.
#[derive(Clone, Debug)]
pub struct SolveConfig {
    pub max_iterations: usize,
    /// Stop when the relative cost decrease falls below this.
    pub rel_tol: f64,
    pub initial_lambda: f64,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            rel_tol: 1e-6,
            initial_lambda: 1e-4,
        }
    }
}

/// What a solve did.
#[derive(Clone, Debug, Default)]
pub struct SolveReport {
    pub iterations: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
}

/// A factor graph over variable blocks.
#[derive(Default)]
pub struct Graph {
    pub vars: BTreeMap<Key, Variable>,
    pub factors: Vec<Box<dyn Factor>>,
    next_key: u64,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a variable block and return its key.
    pub fn add(&mut self, kind: Kind, value: Vec<f64>, fixed: bool) -> Key {
        let k = Key(self.next_key);
        self.next_key += 1;
        self.vars.insert(k, Variable { kind, value, fixed });
        k
    }

    /// Insert a variable under a known key (map loading).
    pub fn insert(&mut self, key: Key, var: Variable) {
        self.next_key = self.next_key.max(key.0 + 1);
        self.vars.insert(key, var);
    }

    pub fn push(&mut self, f: impl Factor + 'static) {
        self.factors.push(Box::new(f));
    }

    pub fn value(&self, k: Key) -> &[f64] {
        &self.vars[&k].value
    }

    /// Column offsets of free variables in the solve vector.
    fn layout(&self) -> (BTreeMap<Key, usize>, usize) {
        let mut off = BTreeMap::new();
        let mut n = 0;
        for (k, v) in &self.vars {
            if !v.fixed {
                off.insert(*k, n);
                n += v.value.len();
            }
        }
        (off, n)
    }

    /// Normal equations H = Σ JᵀJ, g = Σ Jᵀr over free columns, and total cost.
    pub fn normal_equations(&self) -> (DMatrix<f64>, DVector<f64>, f64, BTreeMap<Key, usize>) {
        let (off, n) = self.layout();
        let mut h = DMatrix::zeros(n, n);
        let mut g = DVector::zeros(n);
        let mut cost = 0.0;
        for f in &self.factors {
            let lin = linearize(f.as_ref(), &self.vars);
            cost += lin.cost;
            // Map local columns to global free columns.
            let mut cols: Vec<(usize, usize, usize)> = Vec::new(); // (local, global, len)
            let mut local = 0;
            for k in &lin.keys {
                let len = self.vars[k].value.len();
                if let Some(&o) = off.get(k) {
                    cols.push((local, o, len));
                }
                local += len;
            }
            for &(la, ga, na) in &cols {
                let ja = lin.j.columns(la, na);
                let gr = ja.transpose() * &lin.r;
                g.rows_mut(ga, na).add_assign(&gr);
                for &(lb, gb, nb) in &cols {
                    let jb = lin.j.columns(lb, nb);
                    let blk = ja.transpose() * jb;
                    h.view_mut((ga, gb), (na, nb)).add_assign(&blk);
                }
            }
        }
        (h, g, cost, off)
    }

    pub fn cost(&self) -> f64 {
        self.factors
            .iter()
            .map(|f| linearize(f.as_ref(), &self.vars).cost)
            .sum()
    }

    fn apply(&mut self, off: &BTreeMap<Key, usize>, dx: &DVector<f64>) {
        for (k, &o) in off {
            let v = self.vars.get_mut(k).unwrap();
            for (i, x) in v.value.iter_mut().enumerate() {
                *x += dx[o + i];
            }
            if v.kind == Kind::Pose {
                for a in &mut v.value[3..6] {
                    *a = crate::graph::expr::wrap(*a);
                }
            }
        }
    }

    /// Levenberg-Marquardt on the whole graph.
    pub fn solve(&mut self, cfg: &SolveConfig) -> SolveReport {
        let mut lambda = cfg.initial_lambda;
        let (mut h, mut g, mut cost, mut off) = self.normal_equations();
        let mut report = SolveReport {
            initial_cost: cost,
            final_cost: cost,
            iterations: 0,
        };
        if off.is_empty() {
            return report;
        }
        for it in 0..cfg.max_iterations {
            report.iterations = it + 1;
            let mut accepted = false;
            for _ in 0..8 {
                let mut a = h.clone();
                for i in 0..a.nrows() {
                    a[(i, i)] += lambda * (1.0 + h[(i, i)]);
                }
                let Some(ch) = a.cholesky() else {
                    lambda *= 10.0;
                    continue;
                };
                let dx = -ch.solve(&g);
                let saved: Vec<(Key, Vec<f64>)> = off
                    .keys()
                    .map(|k| (*k, self.vars[k].value.clone()))
                    .collect();
                self.apply(&off, &dx);
                let new_cost = self.cost();
                if new_cost < cost {
                    lambda = (lambda * 0.3).max(1e-9);
                    let rel = (cost - new_cost) / cost.max(1e-300);
                    cost = new_cost;
                    accepted = true;
                    if rel < cfg.rel_tol {
                        report.final_cost = cost;
                        return report;
                    }
                    break;
                }
                for (k, v) in saved {
                    self.vars.get_mut(&k).unwrap().value = v;
                }
                lambda *= 10.0;
            }
            if !accepted {
                break;
            }
            (h, g, cost, off) = self.normal_equations();
        }
        report.final_cost = cost;
        report
    }

    /// Marginal covariance of a free variable at the current estimate
    /// (block of H⁻¹). `None` if the variable is fixed or H is singular.
    pub fn marginal(&self, k: Key) -> Option<DMatrix<f64>> {
        let (h, _, _, off) = self.normal_equations();
        let o = *off.get(&k)?;
        let n = self.vars[&k].value.len();
        let inv = h.cholesky()?.inverse();
        Some(inv.view((o, o), (n, n)).into_owned())
    }

    /// Marginalise variable `m` out: every factor touching it is linearised,
    /// the Schur complement onto its neighbours becomes one [`LinearPrior`],
    /// and `m` and those factors leave the graph. This is what keeps the
    /// window sliding without throwing information away.
    pub fn marginalize(&mut self, m: Key) {
        let (touching, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut self.factors)
            .into_iter()
            .partition(|f| f.keys().contains(&m));
        self.factors = rest;

        // Local layout: m first, then the other free neighbours.
        let mut order = vec![m];
        for f in &touching {
            for k in f.keys() {
                if !order.contains(k) && !self.vars[k].fixed {
                    order.push(*k);
                }
            }
        }
        let mut off = BTreeMap::new();
        let mut n = 0;
        for k in &order {
            off.insert(*k, n);
            n += self.vars[k].value.len();
        }
        let mut h = DMatrix::zeros(n, n);
        let mut g = DVector::zeros(n);
        for f in &touching {
            let lin = linearize(f.as_ref(), &self.vars);
            let mut cols = Vec::new();
            let mut local = 0;
            for k in &lin.keys {
                let len = self.vars[k].value.len();
                if let Some(&o) = off.get(k) {
                    cols.push((local, o, len));
                }
                local += len;
            }
            for &(la, ga, na) in &cols {
                let ja = lin.j.columns(la, na);
                g.rows_mut(ga, na).add_assign(&(ja.transpose() * &lin.r));
                for &(lb, gb, nb) in &cols {
                    let jb = lin.j.columns(lb, nb);
                    h.view_mut((ga, gb), (na, nb))
                        .add_assign(&(ja.transpose() * jb));
                }
            }
        }
        let nm = self.vars[&m].value.len();
        let nr = n - nm;
        self.vars.remove(&m);
        if nr == 0 {
            return;
        }
        let hmm = h.view((0, 0), (nm, nm)).into_owned();
        let hmr = h.view((0, nm), (nm, nr)).into_owned();
        let hrr = h.view((nm, nm), (nr, nr)).into_owned();
        let gm = g.rows(0, nm).into_owned();
        let gr = g.rows(nm, nr).into_owned();
        let hmm_inv = psd_inverse(&hmm);
        let hs = &hrr - hmr.transpose() * &hmm_inv * &hmr;
        let gs = &gr - hmr.transpose() * &hmm_inv * &gm;
        let keys: Vec<Key> = order[1..].to_vec();
        let x_lin: Vec<f64> = keys
            .iter()
            .flat_map(|k| self.vars[k].value.clone())
            .collect();
        let blocks = keys
            .iter()
            .map(|k| (self.vars[k].kind, self.vars[k].value.len()))
            .collect();
        self.push(crate::graph::factors::LinearPrior::new(
            keys, blocks, x_lin, &hs, &gs,
        ));
    }
}

/// Inverse of a symmetric PSD matrix with eigenvalues clamped away from zero.
pub fn psd_inverse(a: &DMatrix<f64>) -> DMatrix<f64> {
    let sym = (a + a.transpose()) * 0.5;
    let eig = sym.symmetric_eigen();
    let max = eig.eigenvalues.max().max(1e-12);
    let d = eig
        .eigenvalues
        .map(|l| if l > max * 1e-12 { 1.0 / l } else { 0.0 });
    &eig.eigenvectors * DMatrix::from_diagonal(&d) * eig.eigenvectors.transpose()
}

use std::ops::AddAssign;
