//! The factors vestibular needs, each a residual written in tang expressions.

use std::sync::Arc;

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use tang_ad::{Tape, Var};

use crate::graph::expr::{self, rot_err, sub_f, whiten};
use crate::graph::core::{Factor, Key, Kind, Robust};

/// Square-root information L (‖L e‖² = eᵀ Σ⁻¹ e) from a covariance.
/// Directions with variance ≥ `1e6` — slamwich's unobservable marker — get
/// zero information instead of a tiny-but-nonzero pull.
pub fn sqrt_info(cov: &DMatrix<f64>) -> DMatrix<f64> {
    let sym = (cov + cov.transpose()) * 0.5;
    let eig = sym.symmetric_eigen();
    let d = eig.eigenvalues.map(|v| {
        if v >= 1e6 || v <= 0.0 {
            0.0
        } else {
            1.0 / v.sqrt()
        }
    });
    DMatrix::from_diagonal(&d) * eig.eigenvectors.transpose()
}

/// Diagonal square-root information from standard deviations.
pub fn sigmas(s: &[f64]) -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_iterator(s.len(), s.iter().map(|v| 1.0 / v)))
}

/// A relative pose measurement in the body frame.
#[derive(Clone, Debug)]
pub struct Rel {
    pub r: Matrix3<f64>,
    pub t: Vector3<f64>,
}

impl Rel {
    pub fn identity() -> Self {
        Self {
            r: Matrix3::identity(),
            t: Vector3::zeros(),
        }
    }

    pub fn from_pose6(p: &[f64]) -> Self {
        Self {
            r: *nalgebra::Rotation3::from_euler_angles(p[3], p[4], p[5]).matrix(),
            t: Vector3::new(p[0], p[1], p[2]),
        }
    }
}

/// Unary prior: L (x − target), rotations wrapped.
pub struct Prior {
    keys: [Key; 1],
    kind: Kind,
    target: Vec<f64>,
    l: DMatrix<f64>,
    robust: Robust,
}

impl Prior {
    pub fn new(key: Key, kind: Kind, target: Vec<f64>, l: DMatrix<f64>) -> Self {
        Self {
            keys: [key],
            kind,
            target,
            l,
            robust: Robust::None,
        }
    }

    pub fn robust(mut self, r: Robust) -> Self {
        self.robust = r;
        self
    }
}

impl Factor for Prior {
    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn residual(&self, _tape: &Arc<Tape>, x: &[Var]) -> Vec<Var> {
        let e: Vec<Var> = x
            .iter()
            .zip(&self.target)
            .enumerate()
            .map(|(i, (xi, t))| {
                let mut d = sub_f(xi, *t);
                if self.kind == Kind::Pose && i >= 3 {
                    // A constant 2πk shift: same derivative, no wrap seam.
                    d = sub_f(&d, d.value() - expr::wrap(d.value()));
                }
                d
            })
            .collect();
        whiten(&self.l, &e)
    }

    fn robust(&self) -> Robust {
        self.robust
    }

    fn name(&self) -> &'static str {
        "prior"
    }
}

/// Relative pose between two pose blocks: scan matches, loop closures, tag
/// sightings (pose → tag block), zero-velocity (identity measurement).
pub struct Between {
    keys: [Key; 2],
    meas: Rel,
    l: DMatrix<f64>,
    robust: Robust,
    name: &'static str,
}

impl Between {
    /// `l` is 6×6 over `[t; rot]` in `i`'s frame.
    pub fn new(i: Key, j: Key, meas: Rel, l: DMatrix<f64>, name: &'static str) -> Self {
        Self {
            keys: [i, j],
            meas,
            l,
            robust: Robust::None,
            name,
        }
    }

    pub fn robust(mut self, r: Robust) -> Self {
        self.robust = r;
        self
    }
}

impl Factor for Between {
    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn residual(&self, tape: &Arc<Tape>, x: &[Var]) -> Vec<Var> {
        let (r_rel, t_rel) = expr::between(&x[0..6], &x[6..12]);
        let rm = expr::rot_const(tape, &self.meas.r);
        let e_r = rot_err(&expr::mtm(&rm, &r_rel));
        let e: Vec<Var> = (0..3)
            .map(|i| sub_f(&t_rel[i], self.meas.t[i]))
            .chain(e_r)
            .collect();
        whiten(&self.l, &e)
    }

    fn robust(&self) -> Robust {
        self.robust
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

/// Planar wheel/leg odometry between two poses: measured `[dx, dy, dyaw]`
/// in pose i's frame. Says nothing about z, roll or pitch; pair it with
/// [`GroundPlane`] or a 3D source for those.
pub struct Odometry {
    keys: [Key; 2],
    /// Measured `[dx, dy, dyaw]` in pose i's frame.
    pub delta: [f64; 3],
    l: DMatrix<f64>,
}

impl Odometry {
    /// `l` is 3×3 over `[dx, dy, dyaw]`.
    pub fn new(i: Key, j: Key, delta: [f64; 3], l: DMatrix<f64>) -> Self {
        Self { keys: [i, j], delta, l }
    }
}

impl Factor for Odometry {
    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn residual(&self, _tape: &Arc<Tape>, x: &[Var]) -> Vec<Var> {
        let (r_rel, t_rel) = expr::between(&x[0..6], &x[6..12]);
        let (s, c) = self.delta[2].sin_cos();
        // sin(yaw_rel − yaw_meas) from the planar block of R_rel.
        let e_yaw = &(&r_rel[1][0] * c) - &(&r_rel[0][0] * s);
        whiten(&self.l, &[sub_f(&t_rel[0], self.delta[0]), sub_f(&t_rel[1], self.delta[1]), e_yaw])
    }

    fn name(&self) -> &'static str {
        "odometry"
    }
}

/// Gyro pre-integration between two poses: measured ΔR.
pub struct Gyro {
    keys: [Key; 2],
    meas: Matrix3<f64>,
    l: DMatrix<f64>,
}

impl Gyro {
    /// `l` is 3×3 over the rotation error.
    pub fn new(i: Key, j: Key, meas: Matrix3<f64>, l: DMatrix<f64>) -> Self {
        Self {
            keys: [i, j],
            meas,
            l,
        }
    }
}

impl Factor for Gyro {
    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn residual(&self, tape: &Arc<Tape>, x: &[Var]) -> Vec<Var> {
        let (ri, _) = expr::pose(&x[0..6]);
        let (rj, _) = expr::pose(&x[6..12]);
        let rm = expr::rot_const(tape, &self.meas);
        let e = rot_err(&expr::mtm(&rm, &expr::mtm(&ri, &rj)));
        whiten(&self.l, &e)
    }

    fn name(&self) -> &'static str {
        "gyro"
    }
}

/// The body stands on a level floor at a known height: `[z − h, roll, pitch]`
/// read from the floor normal in the body frame (R's bottom row).
pub struct GroundPlane {
    keys: [Key; 1],
    pub height: f64,
    l: DMatrix<f64>,
}

impl GroundPlane {
    pub fn new(key: Key, height: f64, l: DMatrix<f64>) -> Self {
        Self {
            keys: [key],
            height,
            l,
        }
    }
}

impl Factor for GroundPlane {
    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn residual(&self, _tape: &Arc<Tape>, x: &[Var]) -> Vec<Var> {
        let (r, _) = expr::pose(x);
        whiten(
            &self.l,
            &[sub_f(&x[2], self.height), r[2][1].clone(), -r[2][0].clone()],
        )
    }

    fn name(&self) -> &'static str {
        "ground"
    }
}

/// What marginalisation leaves behind: ½ dxᵀ H dx + gᵀ dx about `x_lin`,
/// written as ‖Lᵀ dx + L⁻¹ g‖² / 2 with H = L Lᵀ.
pub struct LinearPrior {
    keys: Vec<Key>,
    blocks: Vec<(Kind, usize)>,
    x_lin: Vec<f64>,
    lt: DMatrix<f64>,
    offset: DVector<f64>,
}

impl LinearPrior {
    pub fn new(
        keys: Vec<Key>,
        blocks: Vec<(Kind, usize)>,
        x_lin: Vec<f64>,
        h: &DMatrix<f64>,
        g: &DVector<f64>,
    ) -> Self {
        // H can be rank-deficient (gauge directions); take a PSD square root.
        let sym = (h + h.transpose()) * 0.5;
        let eig = sym.symmetric_eigen();
        let max = eig.eigenvalues.max().max(1e-300);
        let mut lt = DMatrix::zeros(h.nrows(), h.ncols());
        let mut offset = DVector::zeros(h.nrows());
        for k in 0..h.nrows() {
            let lambda = eig.eigenvalues[k];
            if lambda <= max * 1e-12 {
                continue;
            }
            let v = eig.eigenvectors.column(k);
            let s = lambda.sqrt();
            lt.row_mut(k).copy_from(&(v.transpose() * s));
            offset[k] = v.dot(g) / s;
        }
        Self {
            keys,
            blocks,
            x_lin,
            lt,
            offset,
        }
    }
}

impl Factor for LinearPrior {
    fn keys(&self) -> &[Key] {
        &self.keys
    }

    fn residual(&self, _tape: &Arc<Tape>, x: &[Var]) -> Vec<Var> {
        let mut is_angle = Vec::with_capacity(x.len());
        for &(kind, len) in &self.blocks {
            is_angle.extend((0..len).map(|i| kind == Kind::Pose && i >= 3));
        }
        let dx: Vec<Var> = x
            .iter()
            .zip(&self.x_lin)
            .zip(&is_angle)
            .map(|((xi, l), &ang)| {
                let mut d = sub_f(xi, *l);
                if ang {
                    d = sub_f(&d, d.value() - expr::wrap(d.value()));
                }
                d
            })
            .collect();
        let r = whiten(&self.lt, &dx);
        r.into_iter()
            .zip(self.offset.iter())
            .map(|(ri, o)| ri + *o)
            .collect()
    }

    fn name(&self) -> &'static str {
        "marginal"
    }
}
