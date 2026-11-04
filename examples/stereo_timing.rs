//! Times one stereo-cloud match: `cargo run --release --example stereo_timing`.
use slamwich::{CorrelativeScanMatcher, Point3D, ScanMatchConfig, StereoSource, Transform2D};
use std::time::Instant;

fn room(dx: f32) -> Vec<Point3D> {
    // A 160x120 depth image's worth of points on a floor and two walls.
    let mut pts = Vec::new();
    for v in 0..120 {
        for u in 0..160 {
            let (a, b) = (u as f32 / 160.0 * 3.0 - 1.5, v as f32 / 120.0 * 2.0);
            let p = match (u + v) % 3 {
                0 => (a + 2.0 + dx, b - 1.0, 0.0),
                1 => (3.0 + dx, a, b),
                _ => (a + 2.0 + dx, 1.5, b),
            };
            pts.push(Point3D { x: p.0, y: p.1, z: p.2, ..Default::default() });
        }
    }
    pts
}

fn main() {
    let (r, c) = (room(0.0), room(-0.03));
    let src = |p| StereoSource {
        points: p,
        focal_px: 230.0,
        baseline_m: 0.06,
        disparity_sigma_px: 0.5,
        pixel_sigma_px: 0.5,
        half_fov_h: 1.0,
        half_fov_v: 0.8,
        min_range: 0.3,
        max_range: 4.0,
    };
    let m = CorrelativeScanMatcher::new(ScanMatchConfig { resolution: 0.1, ..Default::default() });
    let n = 20;
    let t = Instant::now();
    for _ in 0..n {
        let res = m.match_sources(&src(&r), &src(&c), Transform2D::identity()).unwrap();
        std::hint::black_box(res);
    }
    println!("{} points: {:.2} ms per match", r.len(), t.elapsed().as_secs_f64() * 1e3 / n as f64);
}
