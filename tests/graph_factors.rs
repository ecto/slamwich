#![cfg(feature = "tang")]
//! Tape Jacobians against finite differences, marginalisation against the
//! full solve.

use std::collections::BTreeMap;

use slamwich::graph::{Variable, linearize};
use slamwich::graph::{
    Between, Factor, Graph, GroundPlane, Gyro, Key, Kind, Odometry, Prior, Rel, SolveConfig, sigmas,
};
use nalgebra::Rotation3;

fn vars(blocks: &[(Kind, Vec<f64>)]) -> BTreeMap<Key, Variable> {
    blocks
        .iter()
        .enumerate()
        .map(|(i, (kind, v))| {
            (
                Key(i as u64),
                Variable {
                    kind: *kind,
                    value: v.clone(),
                    fixed: false,
                },
            )
        })
        .collect()
}

/// Central differences on the whitened residual, column by column.
fn check(f: &dyn Factor, v: &BTreeMap<Key, Variable>) {
    let lin = linearize(f, v);
    let mut col = 0;
    for k in f.keys() {
        for i in 0..v[k].value.len() {
            let h = 1e-6;
            let mut plus = v.clone();
            plus.get_mut(k).unwrap().value[i] += h;
            let mut minus = v.clone();
            minus.get_mut(k).unwrap().value[i] -= h;
            let fd = (linearize(f, &plus).r - linearize(f, &minus).r) / (2.0 * h);
            let err = (fd - lin.j.column(col)).amax();
            assert!(
                err < 1e-6,
                "{}: column {col} tape vs FD differs by {err:e}",
                f.name()
            );
            col += 1;
        }
    }
}

const PI: [f64; 6] = [0.3, -0.2, 0.5, 0.05, -0.04, 0.9];
const PJ: [f64; 6] = [0.7, 0.1, 0.48, -0.03, 0.02, 1.3];

#[test]
fn tape_jacobians_match_finite_differences() {
    let poses = vars(&[
        (Kind::Pose, PI.to_vec()),
        (Kind::Pose, PJ.to_vec()),
        (Kind::Vector, vec![0.03, -0.01, 1.2]),
    ]);
    let meas = Rel::from_pose6(&[0.4, 0.2, -0.01, 0.01, 0.02, 0.35]);
    check(
        &Between::new(
            Key(0),
            Key(1),
            meas,
            sigmas(&[0.1, 0.2, 0.3, 0.01, 0.02, 0.03]),
            "between",
        ),
        &poses,
    );
    check(
        &Odometry::new(
            Key(0),
            Key(1),
            [0.4, 0.15, 0.3],
            sigmas(&[0.05, 0.05, 0.02]),
        ),
        &poses,
    );
    check(
        &Gyro::new(
            Key(0),
            Key(1),
            *Rotation3::from_euler_angles(0.0, 0.01, 0.4).matrix(),
            sigmas(&[0.01; 3]),
        ),
        &poses,
    );
    check(
        &GroundPlane::new(Key(0), 0.5, sigmas(&[0.02, 0.03, 0.03])),
        &poses,
    );
    check(
        &Prior::new(
            Key(0),
            Kind::Pose,
            vec![0.0, 0.0, 0.5, 0.0, 0.0, -3.0],
            sigmas(&[0.1; 6]),
        ),
        &poses,
    );
}

/// A chain of five poses with odometry and a prior: solving it whole, and
/// solving it after marginalising the first two, agree on the rest.
#[test]
fn marginalisation_preserves_the_solution() {
    let build = || {
        let mut g = Graph::new();
        let mut keys = vec![g.add(Kind::Pose, vec![0.0; 6], false)];
        g.push(Prior::new(
            keys[0],
            Kind::Pose,
            vec![0.0; 6],
            sigmas(&[0.01; 6]),
        ));
        for i in 1..5 {
            let k = g.add(
                Kind::Pose,
                vec![i as f64 * 0.9, 0.1, 0.0, 0.0, 0.0, 0.05 * i as f64],
                false,
            );
            g.push(GroundPlane::new(k, 0.0, sigmas(&[0.01; 3])));
            g.push(Odometry::new(
                keys[i - 1],
                k,
                [1.0, 0.0, 0.1],
                sigmas(&[0.05, 0.05, 0.02]),
            ));
            keys.push(k);
        }
        // A loop-closure-like constraint from the first to the last pose.
        let d = Rel::from_pose6(&[3.5, 1.0, 0.0, 0.0, 0.0, 0.42]);
        g.push(Between::new(
            keys[0],
            keys[4],
            d,
            sigmas(&[0.1, 0.1, 0.1, 0.05, 0.05, 0.05]),
            "loop",
        ));
        (g, keys)
    };
    let cfg = SolveConfig {
        max_iterations: 50,
        rel_tol: 1e-14,
        ..Default::default()
    };
    let (mut full, keys) = build();
    full.solve(&cfg);
    let (mut part, _) = build();
    part.solve(&cfg);
    part.marginalize(keys[0]);
    part.marginalize(keys[1]);
    // Perturb and re-solve from elsewhere: the marginal prior must hold the answer.
    for k in &keys[2..] {
        part.vars.get_mut(k).unwrap().value[0] += 0.2;
    }
    part.solve(&cfg);
    for k in &keys[2..] {
        let (a, b) = (full.value(*k), part.value(*k));
        let err = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(
            err < 1e-3,
            "pose {k:?} differs by {err} after marginalisation"
        );
    }
}
