//! Pose algebra on tang tape variables.
//!
//! Every residual in this crate is built from these few operations on
//! [`tang_ad::Var`], so its Jacobian comes from the tape's reverse sweep and
//! nothing here is differentiated by hand. Poses are
//! `[x, y, z, roll, pitch, yaw]` with `R = Rz(yaw) Ry(pitch) Rx(roll)`, the
//! same convention as slamwich's GICP.

use std::sync::Arc;

use tang_ad::{Tape, Var};

/// A 3-vector of tape variables.
pub type V3 = [Var; 3];
/// A row-major 3×3 matrix of tape variables.
pub type M3 = [[Var; 3]; 3];

/// A constant on `tape`.
pub fn c(tape: &Arc<Tape>, v: f64) -> Var {
    Var::constant(tape, v)
}

/// `a - k` for a plain number `k`.
pub fn sub_f(a: &Var, k: f64) -> Var {
    a.clone() + (-k)
}

/// Rotation matrix from roll, pitch, yaw.
pub fn rot(roll: &Var, pitch: &Var, yaw: &Var) -> M3 {
    let (sr, cr) = (roll.sin(), roll.cos());
    let (sp, cp) = (pitch.sin(), pitch.cos());
    let (sy, cy) = (yaw.sin(), yaw.cos());
    [
        [
            &cy * &cp,
            &(&(&cy * &sp) * &sr) - &(&sy * &cr),
            &(&(&cy * &sp) * &cr) + &(&sy * &sr),
        ],
        [
            &sy * &cp,
            &(&(&sy * &sp) * &sr) + &(&cy * &cr),
            &(&(&sy * &sp) * &cr) - &(&cy * &sr),
        ],
        [-sp.clone(), &cp * &sr, &cp * &cr],
    ]
}

/// Constant rotation matrix (plain numbers lifted onto the tape).
pub fn rot_const(tape: &Arc<Tape>, r: &nalgebra::Matrix3<f64>) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| c(tape, r[(i, j)])))
}

/// Aᵀ B.
pub fn mtm(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            &(&(&a[0][i] * &b[0][j]) + &(&a[1][i] * &b[1][j])) + &(&a[2][i] * &b[2][j])
        })
    })
}

/// Aᵀ v.
pub fn mtv(a: &M3, v: &V3) -> V3 {
    std::array::from_fn(|i| &(&(&a[0][i] * &v[0]) + &(&a[1][i] * &v[1])) + &(&a[2][i] * &v[2]))
}

/// a − b.
pub fn vsub(a: &V3, b: &V3) -> V3 {
    std::array::from_fn(|i| &a[i] - &b[i])
}

/// Small-angle rotation error: ½ vee(E − Eᵀ), which is sin θ · axis.
/// Exact at the solution and smooth near it; built from products only.
pub fn rot_err(e: &M3) -> V3 {
    [
        (&e[2][1] - &e[1][2]) * 0.5,
        (&e[0][2] - &e[2][0]) * 0.5,
        (&e[1][0] - &e[0][1]) * 0.5,
    ]
}

/// The pose a 6-slice describes: (R, t).
pub fn pose(x: &[Var]) -> (M3, V3) {
    (
        rot(&x[3], &x[4], &x[5]),
        [x[0].clone(), x[1].clone(), x[2].clone()],
    )
}

/// Relative pose of `j` in `i`'s frame: (R_iᵀ R_j, R_iᵀ (t_j − t_i)).
pub fn between(xi: &[Var], xj: &[Var]) -> (M3, V3) {
    let (ri, ti) = pose(xi);
    let (rj, tj) = pose(xj);
    (mtm(&ri, &rj), mtv(&ri, &vsub(&tj, &ti)))
}

/// Whitened residual: L e for an upper-triangular-or-full square `L`.
pub fn whiten(l: &nalgebra::DMatrix<f64>, e: &[Var]) -> Vec<Var> {
    (0..l.nrows())
        .map(|r| {
            let mut acc: Option<Var> = None;
            for (k, ek) in e.iter().enumerate() {
                let w = l[(r, k)];
                if w == 0.0 {
                    continue;
                }
                let term = ek * w;
                acc = Some(match acc {
                    Some(a) => a + term,
                    None => term,
                });
            }
            acc.unwrap_or_else(|| ek_zero(e))
        })
        .collect()
}

fn ek_zero(e: &[Var]) -> Var {
    e[0].clone() * 0.0
}

/// Wrap an angle into (−π, π].
pub fn wrap(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    a - t * ((a + std::f64::consts::PI) / t).floor()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_matches_slamwich_convention() {
        let tape = Tape::new();
        let (r, p, y) = (tape.var(0.1), tape.var(-0.2), tape.var(0.7));
        let m = rot(&r, &p, &y);
        let reference = nalgebra::Rotation3::from_euler_angles(0.1, -0.2, 0.7);
        for i in 0..3 {
            for j in 0..3 {
                assert!((m[i][j].value() - reference[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn rot_err_is_axis_angle_for_small_angles() {
        let tape = Tape::new();
        let z = tape.var(0.0);
        let a = tape.var(0.01);
        let e = rot_err(&rot(&z, &z, &a));
        assert!((e[2].value() - 0.01f64.sin()).abs() < 1e-12);
        assert!(e[0].value().abs() < 1e-15 && e[1].value().abs() < 1e-15);
    }
}
