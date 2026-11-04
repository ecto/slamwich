#![cfg(feature = "tang")]
//! Gate: the tang smoother against slamwich's own EKF + pose graph on
//! synthetic trajectories with known truth, both fed the same clouds and the
//! same corrupted leg odometry.
//!
//! `cargo test -p ipse-slam --release --test gate_vs_ekf -- --nocapture`
//! prints the table; the asserts are the gate.

use slamwich::graph::{Between, Odometry, Rel, Smoother, SmootherConfig, sigmas, sqrt_info};
use nalgebra::{DMatrix, Matrix6};
use slamwich::{
    CorrelativeScanMatcher, LidarSource, Point3D, PointCloud, Pose, ScanMatchConfig, SlamConfig,
    SlamProcessor, Transform2D,
};

const HEIGHT: f64 = 0.5;
const ODOM_HZ: f64 = 10.0;

struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-12), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// A 6 × 5 m room, 2 m walls, plus two boxes so it is not a pure rectangle.
fn room() -> Vec<[f64; 3]> {
    let mut rng = Rng(7);
    let mut pts = Vec::new();
    for _ in 0..6000 {
        pts.push([rng.uniform() * 6.0, rng.uniform() * 5.0, 0.0]);
    }
    for _ in 0..2500 {
        let (a, z) = (rng.uniform(), rng.uniform() * 2.0);
        pts.push([a * 6.0, 0.0, z]);
        pts.push([a * 6.0, 5.0, z]);
        pts.push([0.0, a * 5.0, z]);
        pts.push([6.0, a * 5.0, z]);
    }
    for (cx, cy) in [(4.5, 1.0), (1.2, 4.0)] {
        for _ in 0..600 {
            let (a, z) = (rng.uniform() * 0.6 - 0.3, rng.uniform() * 0.6);
            pts.push([cx + a, cy - 0.3, z]);
            pts.push([cx + a, cy + 0.3, z]);
            pts.push([cx - 0.3, cy + a, z]);
            pts.push([cx + 0.3, cy + a, z]);
        }
    }
    pts
}

/// The room as the body sees it from planar pose (x, y, θ) at HEIGHT.
fn scan(world: &[[f64; 3]], p: [f64; 3], rng: &mut Rng) -> PointCloud {
    let (s, c) = p[2].sin_cos();
    let points = world
        .iter()
        .filter_map(|w| {
            let (dx, dy, dz) = (w[0] - p[0], w[1] - p[1], w[2] - HEIGHT);
            let (bx, by) = (c * dx + s * dy, -s * dx + c * dy);
            let r = (bx * bx + by * by + dz * dz).sqrt();
            (0.3..8.0).contains(&r).then(|| Point3D {
                x: (bx + rng.gauss() * 0.005) as f32,
                y: (by + rng.gauss() * 0.005) as f32,
                z: (dz + rng.gauss() * 0.005) as f32,
                ..Default::default()
            })
        })
        .collect();
    PointCloud::new(points)
}

/// Truth at ODOM_HZ plus whether the door commanded motion.
struct Trajectory {
    name: &'static str,
    truth: Vec<[f64; 3]>,
    commanded: Vec<bool>,
}

fn square() -> Trajectory {
    let mut truth: Vec<[f64; 3]> = vec![[1.5, 1.5, 0.0]];
    let mut commanded = vec![false];
    let dt = 1.0 / ODOM_HZ;
    for _ in 0..4 {
        for _ in 0..(2.0 / 0.2 / dt) as usize {
            let p = *truth.last().unwrap();
            truth.push([
                p[0] + 0.2 * dt * p[2].cos(),
                p[1] + 0.2 * dt * p[2].sin(),
                p[2],
            ]);
            commanded.push(true);
        }
        for _ in 0..(std::f64::consts::FRAC_PI_2 / 0.3 / dt).round() as usize {
            let p = *truth.last().unwrap();
            truth.push([
                p[0],
                p[1],
                p[2] + std::f64::consts::FRAC_PI_2
                    / (std::f64::consts::FRAC_PI_2 / 0.3 / dt).round(),
            ]);
            commanded.push(true);
        }
    }
    Trajectory {
        name: "square 2 m",
        truth,
        commanded,
    }
}

/// Slot E: stands still for 150 s, marching in place.
fn march_in_place() -> Trajectory {
    let n = (150.0 * ODOM_HZ) as usize;
    Trajectory {
        name: "march in place 150 s",
        truth: vec![[3.0, 2.5, 0.3]; n],
        commanded: vec![false; n],
    }
}

/// The leg odometer as measured on the K1: +5 % stride, +5 % yaw, and
/// 3.7 cm/s of phantom forward motion whenever the legs are stepping
/// (always, on this gait). Returns cumulative odometer poses.
fn odometer(t: &Trajectory, rng: &mut Rng) -> Vec<[f64; 3]> {
    let dt = 1.0 / ODOM_HZ;
    let mut odo = vec![t.truth[0]];
    for w in t.truth.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (s, c) = a[2].sin_cos();
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let fwd = (c * dx + s * dy) * 1.05 + 0.037 * dt + rng.gauss() * 0.002;
        let lat = (-s * dx + c * dy) * 1.05 + rng.gauss() * 0.002;
        let dyaw = (b[2] - a[2]) * 1.05 + rng.gauss() * 0.002;
        let o = *odo.last().unwrap();
        let (so, co) = o[2].sin_cos();
        odo.push([
            o[0] + co * fwd - so * lat,
            o[1] + so * fwd + co * lat,
            o[2] + dyaw,
        ]);
    }
    odo
}

fn rel2(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let (s, c) = a[2].sin_cos();
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    [
        c * dx + s * dy,
        -s * dx + c * dy,
        slamwich::graph::expr::wrap(b[2] - a[2]),
    ]
}

fn rel6(a: &[f64; 6], b: &[f64; 6]) -> [f64; 3] {
    rel2([a[0], a[1], a[5]], [b[0], b[1], b[5]])
}

fn compose(a: [f64; 6], d: [f64; 3]) -> [f64; 6] {
    let (s, c) = a[5].sin_cos();
    [
        a[0] + c * d[0] - s * d[1],
        a[1] + s * d[0] + c * d[1],
        a[2],
        a[3],
        a[4],
        a[5] + d[2],
    ]
}

struct Errors {
    rms: f64,
    max: f64,
    last: f64,
}

fn errors(est: &[[f64; 2]], truth: &[[f64; 2]]) -> Errors {
    let d: Vec<f64> = est
        .iter()
        .zip(truth)
        .map(|(e, t)| ((e[0] - t[0]).powi(2) + (e[1] - t[1]).powi(2)).sqrt())
        .collect();
    Errors {
        rms: (d.iter().map(|x| x * x).sum::<f64>() / d.len() as f64).sqrt(),
        max: d.iter().cloned().fold(0.0, f64::max),
        last: *d.last().unwrap(),
    }
}

/// slamwich's own pipeline: odometry at 10 Hz into the EKF, a scan per second.
fn run_ekf(world: &[[f64; 3]], t: &Trajectory, odo: &[[f64; 3]]) -> Vec<[f64; 2]> {
    let mut slam = SlamProcessor::new(SlamConfig::default());
    let mut rng = Rng(99);
    let o0 = odo[0];
    let mut out = Vec::new();
    for (i, o) in odo.iter().enumerate() {
        // slamwich starts at its own origin; feed odometry relative to the start.
        let r = rel2(o0, *o);
        slam.update_odometry(&Pose {
            x: r[0],
            y: r[1],
            theta: r[2],
        });
        if i % ODOM_HZ as usize == 0 {
            let cloud = scan(world, t.truth[i], &mut rng);
            slam.process_scan(&cloud);
            let p = slam.pose();
            // Back into the world frame of the truth.
            let (s, c) = t.truth[0][2].sin_cos();
            out.push([
                t.truth[0][0] + c * p.x - s * p.y,
                t.truth[0][1] + s * p.x + c * p.y,
            ]);
        }
    }
    out
}

/// The tang smoother: odometry + GICP scan factors + ground plane, window 20.
fn run_smoother(world: &[[f64; 3]], t: &Trajectory, odo: &[[f64; 3]]) -> (Vec<[f64; 2]>, f64) {
    let mut solve_s = 0.0;
    let start = t.truth[0];
    let cfg = SmootherConfig {
        ground_height: Some(HEIGHT),
        ..Default::default()
    };
    let mut sm = Smoother::new(cfg, [start[0], start[1], HEIGHT, 0.0, 0.0, start[2]]);
    let matcher = CorrelativeScanMatcher::new(ScanMatchConfig {
        resolution: 0.15,
        ..Default::default()
    });
    let mut rng = Rng(99);
    let step = ODOM_HZ as usize;
    // Clouds of live keyframes: each new one is matched to the newest, the
    // oldest live, and one in between, as slamwich matches to its keyframe.
    let mut clouds: Vec<(slamwich::graph::Key, [f64; 6], PointCloud)> = vec![(
        sm.head(),
        sm.pose(sm.head()),
        scan(world, t.truth[0], &mut rng),
    )];
    let mut prev_i = 0;
    let mut out = vec![[start[0], start[1]]];
    for i in (step..t.truth.len()).step_by(step) {
        let d = rel2(odo[prev_i], odo[i]);
        let dt = (i - prev_i) as f64 / ODOM_HZ;
        let moving = t.commanded[prev_i..i].iter().any(|&c| c);
        let head = sm.head();
        let guess = compose(sm.pose(head), d);
        let k = sm.add_keyframe(guess);
        // Leg odometry: its errors are biases, not white noise, so it is
        // loose; looser still when the door commanded nothing.
        let s_lin = 0.15 * d[0].hypot(d[1]) + 0.02 * dt + if moving { 0.0 } else { 0.2 * dt };
        let s_yaw = 0.1 * d[2].abs() + 0.01 * dt;
        sm.graph.push(Odometry::new(head, k, d, sigmas(&[s_lin, s_lin, s_yaw])));
        let cloud = scan(world, t.truth[i], &mut rng);
        let n = clouds.len();
        let mut targets = vec![n - 1, 0, n / 2];
        targets.sort_unstable();
        targets.dedup();
        for ti in targets {
            let (rk, rp, rc) = &clouds[ti];
            let g = rel6(rp, &guess);
            let Ok(m) = matcher.match_sources(
                &LidarSource::new(rc),
                &LidarSource::new(&cloud),
                Transform2D::new(g[0], g[1], g[2]),
            ) else {
                continue;
            };
            if m.score > 0.5 {
                let cov = m.covariance6 + Matrix6::identity() * 1e-5;
                let cov = DMatrix::from_iterator(6, 6, cov.iter().cloned());
                sm.graph.push(Between::new(
                    *rk,
                    k,
                    Rel::from_pose6(m.pose6.as_slice()),
                    sqrt_info(&cov),
                    "scan",
                ));
            }
        }
        let ts = std::time::Instant::now();
        sm.optimize();
        solve_s += ts.elapsed().as_secs_f64();
        let live: Vec<_> = sm.poses().collect();
        clouds.retain(|(key, _, _)| live.contains(key));
        for c in &mut clouds {
            c.1 = sm.pose(c.0);
        }
        clouds.push((k, sm.pose(k), cloud));
        let p = sm.pose(k);
        out.push([p[0], p[1]]);
        prev_i = i;
    }
    let n = out.len() as f64;
    (out, solve_s * 1e3 / n)
}

#[test]
fn smoother_beats_ekf_on_synthetic_truth() {
    let world = room();
    println!("\n| trajectory | estimator | RMS m | max m | final m |");
    println!("|---|---|---|---|---|");
    for t in [square(), march_in_place()] {
        let odo = odometer(&t, &mut Rng(3));
        let truth: Vec<[f64; 2]> = t
            .truth
            .iter()
            .step_by(ODOM_HZ as usize)
            .map(|p| [p[0], p[1]])
            .collect();
        let odo_xy: Vec<[f64; 2]> = odo
            .iter()
            .step_by(ODOM_HZ as usize)
            .map(|p| [p[0], p[1]])
            .collect();
        let ekf = run_ekf(&world, &t, &odo);
        let tm = std::time::Instant::now();
        let (sm, solve_ms) = run_smoother(&world, &t, &odo);
        let per_kf = tm.elapsed().as_secs_f64() * 1e3 / sm.len() as f64;
        let (eo, ee, es) = (
            errors(&odo_xy, &truth),
            errors(&ekf, &truth[..ekf.len()]),
            errors(&sm, &truth[..sm.len()]),
        );
        for (name, e) in [
            ("leg odometer", &eo),
            ("slamwich EKF+graph", &ee),
            ("tang smoother", &es),
        ] {
            println!(
                "| {} | {} | {:.3} | {:.3} | {:.3} |",
                t.name, name, e.rms, e.max, e.last
            );
        }
        println!(
            "| {} | smoother ms/keyframe on a Mac: all {:.1}, solve+marginalise {:.1} | | | |",
            t.name, per_kf, solve_ms
        );
        assert!(
            es.rms < ee.rms,
            "{}: smoother RMS {:.3} not below EKF {:.3}",
            t.name,
            es.rms,
            ee.rms
        );
        assert!(
            es.max < 0.3,
            "{}: smoother max error {:.3} m",
            t.name,
            es.max
        );
    }
}
