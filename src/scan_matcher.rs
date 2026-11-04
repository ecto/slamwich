//! 3D Generalized ICP (GICP) scan matcher.
//!
//! Replaces the previous 2D correlative scan matcher with a full 3D
//! point-to-plane ICP variant that uses local covariance matrices.
//! The result is projected back to 2D (x, y, yaw) for the pose graph.

use std::collections::HashMap;
use std::num::NonZero;
use std::sync::Arc;

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use nalgebra::{Matrix3, Matrix3x6, Matrix6, Vector3, Vector6};
use tracing::debug;

use crate::point_source::{LidarSource, PointSource};
use crate::transforms::Transform2D;
#[cfg(test)]
use crate::types::Point3D;
use crate::types::PointCloud;
use crate::SlamError;

/// Configuration for GICP scan matching.
#[derive(Debug, Clone)]
pub struct ScanMatchConfig {
    /// Voxel size for downsampling (meters)
    pub resolution: f64,
    /// Linear search range — unused in GICP, kept for config compat
    pub linear_range: f64,
    /// Angular search range — unused in GICP, kept for config compat
    pub angular_range: f64,
    /// Maximum GICP iterations
    pub max_iterations: usize,
    /// Convergence threshold (pose delta norm)
    pub convergence_threshold: f64,
    /// Maximum correspondence distance for outlier rejection (meters)
    pub max_correspondence_dist: f64,
    /// Minimum fraction of points with valid correspondences
    pub min_overlap: f64,
    /// A Hessian direction is unobservable when its eigenvalue, after
    /// scaling rotations by the cloud radius, is below this fraction of the
    /// largest. Updates are applied only in observable directions.
    pub degeneracy_ratio: f64,
}

impl Default for ScanMatchConfig {
    fn default() -> Self {
        Self {
            resolution: 0.2,
            linear_range: 0.5,
            angular_range: 0.26,
            max_iterations: 30,
            convergence_threshold: 1e-4,
            max_correspondence_dist: 1.0,
            min_overlap: 0.3,
            degeneracy_ratio: 1e-2,
        }
    }
}

/// Which pose directions the scene constrains.
///
/// Directions are in the matcher's 6-vector `[x, y, z, roll, pitch, yaw]`,
/// with rotations scaled by `length_scale` so metres and radians compare.
#[derive(Debug, Clone, Default)]
pub struct Degeneracy {
    /// Eigenvalues of the scaled Hessian, ascending.
    pub eigenvalues: Vector6<f64>,
    /// Unit eigenvectors (columns) matching `eigenvalues`.
    pub eigenvectors: Matrix6<f64>,
    /// How many of the leading (smallest) eigen-directions are unobservable.
    pub unobservable: usize,
    /// Cloud radius used to scale the rotational block (meters).
    pub length_scale: f64,
}

impl Degeneracy {
    /// True when every direction is constrained.
    pub fn is_well_conditioned(&self) -> bool {
        self.unobservable == 0
    }

    /// Unobservable directions in unscaled pose coordinates (unit norm).
    pub fn unobservable_directions(&self) -> impl Iterator<Item = Vector6<f64>> + '_ {
        let d_inv = scale_inv(self.length_scale);
        (0..self.unobservable).map(move |k| {
            let v = d_inv * self.eigenvectors.column(k);
            v / v.norm()
        })
    }
}

/// Result of scan matching.
#[derive(Debug, Clone)]
pub struct ScanMatchResult {
    /// Estimated relative transform (projected to 2D: x, y, yaw)
    pub transform: Transform2D,
    /// Match quality score (0-1, overlap ratio)
    pub score: f64,
    /// Estimated covariance of the match (3x3: x, y, yaw)
    pub covariance: Matrix3<f64>,
    /// Full estimate `[x, y, z, roll, pitch, yaw]`.
    pub pose6: Vector6<f64>,
    /// Covariance of `pose6`. Unobservable directions carry
    /// [`UNOBSERVABLE_VARIANCE`] rather than the inverse of a near-zero
    /// eigenvalue, so downstream fusion ignores them instead of trusting them.
    pub covariance6: Matrix6<f64>,
    /// Observability of the final solve.
    pub degeneracy: Degeneracy,
}

/// Variance assigned to directions the scene does not constrain (m², rad²).
pub const UNOBSERVABLE_VARIANCE: f64 = 1e6;

/// 3D GICP scan matcher.
pub struct CorrelativeScanMatcher {
    config: ScanMatchConfig,
}

/// A point source after voxelisation: centroids plus averaged sensor noise.
struct Prepared {
    points: Vec<Vector3<f64>>,
    covs: Vec<Matrix3<f64>>,
}

impl CorrelativeScanMatcher {
    /// Create a new scan matcher.
    pub fn new(config: ScanMatchConfig) -> Self {
        Self { config }
    }

    /// Match current scan to reference scan.
    pub fn match_scans(
        &self,
        reference: &Arc<PointCloud>,
        current: &Arc<PointCloud>,
        initial_guess: Transform2D,
    ) -> Result<ScanMatchResult, SlamError> {
        self.match_sources(
            &LidarSource::new(reference),
            &LidarSource::new(current),
            initial_guess,
        )
    }

    /// Match any two point sources. `current` is expressed in the frame of
    /// `reference` by the returned transform.
    pub fn match_sources<R: PointSource + ?Sized, C: PointSource + ?Sized>(
        &self,
        reference: &R,
        current: &C,
        initial_guess: Transform2D,
    ) -> Result<ScanMatchResult, SlamError> {
        let guess_t = initial_guess.translation();
        let guess = Vector6::new(
            guess_t.x,
            guess_t.y,
            0.0,
            0.0,
            0.0,
            initial_guess.rotation(),
        );
        self.match_sources_6d(reference, current, guess)
    }

    /// [`Self::match_sources`] with a full 6-DOF initial guess.
    pub fn match_sources_6d<R: PointSource + ?Sized, C: PointSource + ?Sized>(
        &self,
        reference: &R,
        current: &C,
        initial_guess: Vector6<f64>,
    ) -> Result<ScanMatchResult, SlamError> {
        let reference = prepare(reference, self.config.resolution);
        let current = prepare(current, self.config.resolution);

        if reference.points.len() < 10 || current.points.len() < 10 {
            return Err(SlamError::ScanMatchFailed(
                "Too few points after downsampling".into(),
            ));
        }

        // Surface covariances are fixed for the whole solve: computed once,
        // so the iteration loop below never allocates.
        let mut ref_covs = estimate_covariances(&reference.points, 10);
        for (c, s) in ref_covs.iter_mut().zip(&reference.covs) {
            *c += s;
        }
        let mut cur_covs = estimate_covariances(&current.points, 10);
        for (c, s) in cur_covs.iter_mut().zip(&current.covs) {
            *c += s;
        }
        let ref_tree = build_kdtree(&reference.points);
        let length_scale = rms_radius(&current.points);

        let mut pose = initial_guess;
        let max_dist_sq =
            (self.config.max_correspondence_dist * self.config.max_correspondence_dist) as f32;
        let n_cur = current.points.len() as f64;

        for iteration in 0..self.config.max_iterations {
            let (h, b, valid_count) = accumulate(
                &pose,
                &reference.points,
                &ref_covs,
                &ref_tree,
                &current.points,
                &cur_covs,
                max_dist_sq,
            );

            let overlap = valid_count as f64 / n_cur;
            if overlap < self.config.min_overlap {
                return Err(SlamError::ScanMatchFailed(format!(
                    "Insufficient overlap: {:.2}",
                    overlap
                )));
            }

            let deg = analyse(&h, length_scale, self.config.degeneracy_ratio);
            let dx = observable_step(&b, &deg);

            pose += dx;

            let delta_norm = dx.norm();
            if delta_norm < self.config.convergence_threshold {
                debug!(
                    iterations = iteration + 1,
                    overlap = overlap,
                    unobservable = deg.unobservable,
                    "GICP converged"
                );
                break;
            }
        }

        // Final overlap, observability and covariance
        let (h_final, _, valid_count) = accumulate(
            &pose,
            &reference.points,
            &ref_covs,
            &ref_tree,
            &current.points,
            &cur_covs,
            max_dist_sq,
        );
        let score = valid_count as f64 / n_cur;
        let degeneracy = analyse(&h_final, length_scale, self.config.degeneracy_ratio);
        let cov_6d = observable_covariance(&degeneracy);

        let idx = [0, 1, 5];
        let mut cov_3d = Matrix3::zeros();
        for (r, &ri) in idx.iter().enumerate() {
            for (c, &ci) in idx.iter().enumerate() {
                cov_3d[(r, c)] = cov_6d[(ri, ci)];
            }
        }

        let transform = Transform2D::new(pose[0], pose[1], pose[5]);

        Ok(ScanMatchResult {
            transform,
            score,
            covariance: cov_3d,
            pose6: pose,
            covariance6: cov_6d,
            degeneracy,
        })
    }
}

/// One Gauss-Newton linearisation: H = Σ JᵀC⁻¹J, b = Σ JᵀC⁻¹r.
/// Allocation-free: kd lookups, fixed-size matrices only.
fn accumulate(
    pose: &Vector6<f64>,
    ref_points: &[Vector3<f64>],
    ref_covs: &[Matrix3<f64>],
    ref_tree: &ImmutableKdTree<f32, 3>,
    cur_points: &[Vector3<f64>],
    cur_covs: &[Matrix3<f64>],
    max_dist_sq: f32,
) -> (Matrix6<f64>, Vector6<f64>, usize) {
    let rot = rotation_from_euler(pose[3], pose[4], pose[5]);
    let trans = Vector3::new(pose[0], pose[1], pose[2]);
    let d_rot = rotation_derivatives(pose, &rot);

    let mut h = Matrix6::zeros();
    let mut b = Vector6::zeros();
    let mut valid_count = 0usize;

    for (j, cur_pt) in cur_points.iter().enumerate() {
        let transformed = rot * cur_pt + trans;
        let query = [
            transformed.x as f32,
            transformed.y as f32,
            transformed.z as f32,
        ];
        let nn = ref_tree.nearest_one::<SquaredEuclidean>(&query);
        if nn.distance > max_dist_sq {
            continue;
        }
        let i = nn.item as usize;
        valid_count += 1;

        let c_combined = ref_covs[i] + rot * cur_covs[j] * rot.transpose();
        let c_inv = (c_combined + Matrix3::identity() * 1e-6)
            .try_inverse()
            .unwrap_or(Matrix3::identity());
        let residual = ref_points[i] - transformed;

        let mut jac = Matrix3x6::zeros();
        jac[(0, 0)] = -1.0;
        jac[(1, 1)] = -1.0;
        jac[(2, 2)] = -1.0;
        for (axis, dr) in d_rot.iter().enumerate() {
            jac.set_column(3 + axis, &(-(dr * cur_pt)));
        }

        let jt_cinv = jac.transpose() * c_inv;
        h += jt_cinv * jac;
        b += jt_cinv * residual;
    }
    (h, b, valid_count)
}

/// ∂R/∂(roll, pitch, yaw) at `pose`, by the same forward difference the
/// per-point Jacobian always used — hoisted out of the point loop.
fn rotation_derivatives(pose: &Vector6<f64>, rot: &Matrix3<f64>) -> [Matrix3<f64>; 3] {
    let h = 1e-6;
    let mut out = [Matrix3::zeros(); 3];
    for (axis, d) in out.iter_mut().enumerate() {
        let mut p = *pose;
        p[3 + axis] += h;
        *d = (rotation_from_euler(p[3], p[4], p[5]) - rot) / h;
    }
    out
}

/// D⁻¹ for D = diag(1, 1, 1, L, L, L), which turns radians into metres at
/// the cloud radius L.
fn scale_inv(length: f64) -> Matrix6<f64> {
    let l = 1.0 / length;
    Matrix6::from_diagonal(&Vector6::new(1.0, 1.0, 1.0, l, l, l))
}

/// Eigen-analysis of D⁻¹ H D⁻¹ (Zhang, Kaess & Singh 2016).
fn analyse(h: &Matrix6<f64>, length_scale: f64, ratio: f64) -> Degeneracy {
    let d_inv = scale_inv(length_scale);
    let hs = d_inv * h * d_inv;
    let hs = (hs + hs.transpose()) * 0.5;
    let eig = hs.symmetric_eigen();
    let mut order = [0usize, 1, 2, 3, 4, 5];
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    let mut eigenvalues = Vector6::zeros();
    let mut eigenvectors = Matrix6::zeros();
    for (k, &o) in order.iter().enumerate() {
        eigenvalues[k] = eig.eigenvalues[o];
        eigenvectors.set_column(k, &eig.eigenvectors.column(o));
    }
    let max = eigenvalues[5].max(f64::MIN_POSITIVE);
    let unobservable = eigenvalues.iter().take_while(|&&l| l < ratio * max).count();
    Degeneracy {
        eigenvalues,
        eigenvectors,
        unobservable,
        length_scale,
    }
}

/// Gauss-Newton step solved in the eigenbasis, with unobservable
/// components zeroed: the scene moves the pose only where it can see.
fn observable_step(b: &Vector6<f64>, deg: &Degeneracy) -> Vector6<f64> {
    let d_inv = scale_inv(deg.length_scale);
    // H dx = -b  ⇔  Hs (D dx) = -D⁻¹ b
    let bs = d_inv * b;
    let mut ys = Vector6::zeros();
    for k in deg.unobservable..6 {
        let v = deg.eigenvectors.column(k);
        let lambda = deg.eigenvalues[k] + 1e-6;
        ys -= v * (v.dot(&bs) / lambda);
    }
    d_inv * ys
}

/// H⁻¹ in observable directions, [`UNOBSERVABLE_VARIANCE`] in the rest.
fn observable_covariance(deg: &Degeneracy) -> Matrix6<f64> {
    let d_inv = scale_inv(deg.length_scale);
    let mut cs = Matrix6::zeros();
    for k in 0..6 {
        let v = deg.eigenvectors.column(k);
        let var = if k < deg.unobservable {
            UNOBSERVABLE_VARIANCE
        } else {
            1.0 / (deg.eigenvalues[k] + 1e-6)
        };
        cs += v * v.transpose() * var;
    }
    d_inv * cs * d_inv
}

fn rms_radius(points: &[Vector3<f64>]) -> f64 {
    let n = points.len().max(1) as f64;
    let c = points.iter().fold(Vector3::zeros(), |a, p| a + p) / n;
    let r2 = points.iter().map(|p| (p - c).norm_squared()).sum::<f64>() / n;
    r2.sqrt().max(1e-3)
}

/// Voxelise a source: range/FOV gate, centroid per voxel, mean sensor noise
/// per voxel divided by the count (averaging n points shrinks noise by n).
/// Output is sorted by voxel key, so results do not depend on hash order.
fn prepare<S: PointSource + ?Sized>(source: &S, voxel_size: f64) -> Prepared {
    let inv = 1.0 / voxel_size;
    let (min_r, max_r) = source.range_limits();
    let (min_sq, max_sq) = (min_r * min_r, max_r * max_r);
    let mut grid: HashMap<(i32, i32, i32), (Vector3<f64>, Matrix3<f64>, usize)> = HashMap::new();

    for i in 0..source.len() {
        let p = source.point(i);
        let range_sq = p.norm_squared();
        if !range_sq.is_finite() || range_sq < min_sq || range_sq > max_sq || !source.in_fov(&p) {
            continue;
        }
        let key = (
            (p.x * inv).floor() as i32,
            (p.y * inv).floor() as i32,
            (p.z * inv).floor() as i32,
        );
        let entry = grid
            .entry(key)
            .or_insert((Vector3::zeros(), Matrix3::zeros(), 0));
        entry.0 += p;
        entry.1 += source.covariance(i);
        entry.2 += 1;
    }

    let mut cells: Vec<_> = grid.into_iter().collect();
    cells.sort_unstable_by_key(|(k, _)| *k);
    let mut points = Vec::with_capacity(cells.len());
    let mut covs = Vec::with_capacity(cells.len());
    for (_, (sum, cov, count)) in cells {
        let n = count as f64;
        points.push(sum / n);
        covs.push(cov / (n * n));
    }
    Prepared { points, covs }
}

/// Voxel downsample a point cloud.
#[cfg(test)]
fn voxel_downsample(points: &[Point3D], voxel_size: f64) -> Vec<Vector3<f64>> {
    let cloud = PointCloud::new(points.to_vec());
    prepare(&LidarSource::new(&cloud), voxel_size).points
}

/// Estimate local covariance matrices for each point.
fn estimate_covariances(points: &[Vector3<f64>], k: usize) -> Vec<Matrix3<f64>> {
    if points.len() < k {
        return points.iter().map(|_| Matrix3::identity() * 0.01).collect();
    }

    let tree = build_kdtree(points);
    let k_query = k.min(points.len());

    points
        .iter()
        .map(|pt| {
            let query = [pt.x as f32, pt.y as f32, pt.z as f32];
            let neighbors =
                tree.nearest_n::<SquaredEuclidean>(&query, NonZero::new(k_query).unwrap());

            if neighbors.len() < 3 {
                return Matrix3::identity() * 0.01;
            }

            let mut centroid = Vector3::zeros();
            for nn in &neighbors {
                let idx = nn.item as usize;
                centroid += &points[idx];
            }
            centroid /= neighbors.len() as f64;

            let mut cov = Matrix3::zeros();
            for nn in &neighbors {
                let idx = nn.item as usize;
                let diff = &points[idx] - centroid;
                cov += diff * diff.transpose();
            }
            cov /= neighbors.len() as f64;

            // Symmetric eigendecomposition, not SVD: for a planar patch the
            // smallest singular value is ~0 and SVD is free to flip the sign
            // of that column of V against U, so U S Vᵀ came back with a
            // negative variance along the normal and GICP's Hessian lost
            // definiteness.
            let eig = cov.symmetric_eigen();
            let max = eig.eigenvalues.max();
            let eps = max * 1e-3;
            let s = eig.eigenvalues.map(|l| l.max(eps));
            eig.eigenvectors * Matrix3::from_diagonal(&s) * eig.eigenvectors.transpose()
        })
        .collect()
}

/// Build a KD-tree from 3D points.
fn build_kdtree(points: &[Vector3<f64>]) -> ImmutableKdTree<f32, 3> {
    let entries: Vec<[f32; 3]> = points
        .iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32])
        .collect();

    ImmutableKdTree::new_from_slice(&entries)
}

/// Compute rotation matrix from Euler angles (roll, pitch, yaw).
fn rotation_from_euler(roll: f64, pitch: f64, yaw: f64) -> Matrix3<f64> {
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sy, cy) = yaw.sin_cos();

    Matrix3::new(
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_box_scan(offset_x: f32, offset_y: f32, offset_z: f32) -> PointCloud {
        let mut points = Vec::new();

        for ix in -20..=20 {
            for iy in -20..=20 {
                points.push(Point3D {
                    x: ix as f32 * 0.25 + offset_x,
                    y: iy as f32 * 0.25 + offset_y,
                    z: 0.0 + offset_z,
                    reflectivity: 128,
                    tag: 0,
                });
            }
        }

        for iy in -20..=20 {
            for iz in 0..=8 {
                for &x in &[5.0, -5.0] {
                    points.push(Point3D {
                        x: x + offset_x,
                        y: iy as f32 * 0.25 + offset_y,
                        z: iz as f32 * 0.25 + offset_z,
                        reflectivity: 128,
                        tag: 0,
                    });
                }
            }
        }

        for ix in -20..=20 {
            for iz in 0..=8 {
                for &y in &[5.0, -5.0] {
                    points.push(Point3D {
                        x: ix as f32 * 0.25 + offset_x,
                        y: y + offset_y,
                        z: iz as f32 * 0.25 + offset_z,
                        reflectivity: 128,
                        tag: 0,
                    });
                }
            }
        }

        PointCloud::new(points)
    }

    #[test]
    fn test_voxel_downsample() {
        let scan = make_box_scan(0.0, 0.0, 0.0);
        let downsampled = voxel_downsample(&scan.points, 0.5);
        assert!(downsampled.len() < scan.points.len());
        assert!(downsampled.len() > 10);
    }

    #[test]
    fn test_rotation_from_euler_identity() {
        let r = rotation_from_euler(0.0, 0.0, 0.0);
        let identity: Matrix3<f64> = Matrix3::identity();
        for i in 0..3usize {
            for j in 0..3usize {
                assert!((r[(i, j)] - identity[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_identity_match() {
        let config = ScanMatchConfig::default();
        let matcher = CorrelativeScanMatcher::new(config);

        let scan = Arc::new(make_box_scan(0.0, 0.0, 0.0));
        let result = matcher
            .match_scans(&scan, &scan, Transform2D::identity())
            .unwrap();

        assert!(result.transform.translation().norm() < 0.1);
        assert!(result.transform.rotation().abs() < 0.1);
        assert!(result.score > 0.8);
    }

    fn grid(points: &mut Vec<Point3D>, f: impl Fn(f32, f32) -> (f32, f32, f32), n: i32) {
        for a in -n..=n {
            for b in -n..=n {
                let (x, y, z) = f(a as f32 * 0.1, b as f32 * 0.1);
                points.push(Point3D {
                    x,
                    y,
                    z,
                    reflectivity: 0,
                    tag: 0,
                });
            }
        }
    }

    /// Floor plus one wall facing +x: nothing pins motion along the wall.
    fn floor_and_wall(dy: f32) -> PointCloud {
        let mut pts = Vec::new();
        grid(&mut pts, |a, b| (a + 1.5, b + dy, 0.0), 15);
        grid(&mut pts, |a, b| (3.0, a + dy, b + 1.5), 15);
        PointCloud::new(pts)
    }

    #[test]
    fn floor_and_single_wall_reports_along_wall_unobservable() {
        let matcher = CorrelativeScanMatcher::new(ScanMatchConfig {
            resolution: 0.1,
            ..Default::default()
        });
        let reference = floor_and_wall(0.0);
        // Current is shifted along the wall: the true answer is unknowable.
        let current = floor_and_wall(0.3);
        let res = matcher
            .match_sources(
                &LidarSource::new(&reference),
                &LidarSource::new(&current),
                Transform2D::new(0.05, 0.0, 0.0),
            )
            .unwrap();
        let deg = &res.degeneracy;
        assert_eq!(deg.unobservable, 1, "eigenvalues {:?}", deg.eigenvalues);
        let dir = deg.unobservable_directions().next().unwrap();
        assert!(
            dir[1].abs() > 0.99,
            "unobservable direction {dir:?} is not along the wall"
        );
        // Observable directions converge; the unobservable one is left at
        // the guess instead of wandering on noise.
        assert!(res.pose6[0].abs() < 1e-3, "x {}", res.pose6[0]);
        assert!(res.pose6[1].abs() < 1e-3, "y moved to {}", res.pose6[1]);
        assert!(res.covariance6[(1, 1)] > 1e3);
        assert!(res.covariance6[(0, 0)] < res.covariance6[(1, 1)] * 1e-3);
    }

    #[test]
    fn box_is_fully_observable() {
        let matcher = CorrelativeScanMatcher::new(ScanMatchConfig::default());
        let scan = Arc::new(make_box_scan(0.0, 0.0, 0.0));
        let res = matcher
            .match_scans(&scan, &scan, Transform2D::identity())
            .unwrap();
        assert!(
            res.degeneracy.is_well_conditioned(),
            "{:?}",
            res.degeneracy.eigenvalues
        );
    }

    /// The pre-PointSource matcher on this scene (two runs of the old code;
    /// it was hash-order nondeterministic at ~1e-5 m): the shift is
    /// recovered to < 2e-5 m and the score is 1.0.
    #[test]
    fn lidar_matches_legacy_results() {
        let matcher = CorrelativeScanMatcher::new(ScanMatchConfig::default());
        let reference = Arc::new(legacy_box(0.0, 0.0));
        for (dx, dy) in [(0.3f32, -0.2f32), (-0.15, 0.25)] {
            let current = Arc::new(legacy_box(dx, dy));
            let res = matcher
                .match_scans(&reference, &current, Transform2D::identity())
                .unwrap();
            let t = res.transform.translation();
            assert!((t.x + dx as f64).abs() < 2e-4, "x {} vs {}", t.x, -dx);
            assert!((t.y + dy as f64).abs() < 2e-4, "y {} vs {}", t.y, -dy);
            assert!(res.transform.rotation().abs() < 1e-4);
            assert_eq!(res.score, 1.0);
            assert!(res.degeneracy.is_well_conditioned());
        }
    }

    #[test]
    fn match_is_deterministic() {
        let matcher = CorrelativeScanMatcher::new(ScanMatchConfig::default());
        let reference = Arc::new(legacy_box(0.0, 0.0));
        let current = Arc::new(legacy_box(-0.15, 0.25));
        let a = matcher
            .match_scans(&reference, &current, Transform2D::identity())
            .unwrap();
        let b = matcher
            .match_scans(&reference, &current, Transform2D::identity())
            .unwrap();
        assert_eq!(a.pose6, b.pose6);
    }

    fn legacy_box(ox: f32, oy: f32) -> PointCloud {
        let mut points = Vec::new();
        let p = |x: f32, y: f32, z: f32| Point3D {
            x: x + ox,
            y: y + oy,
            z,
            reflectivity: 0,
            tag: 0,
        };
        for ix in -20..=20 {
            for iy in -20..=20 {
                points.push(p(ix as f32 * 0.25, iy as f32 * 0.25, 0.0));
            }
        }
        for i in -20..=20 {
            for iz in 0..=8 {
                for s in [5.0f32, -5.0] {
                    points.push(p(s, i as f32 * 0.25, iz as f32 * 0.25));
                    points.push(p(i as f32 * 0.25 * 0.9, s * 0.8, iz as f32 * 0.25));
                }
            }
        }
        PointCloud::new(points)
    }

    #[test]
    fn stereo_source_matches_through_the_same_solver() {
        use crate::point_source::StereoSource;
        // Floor, a wall ahead and a wall to the left: a corner, all observable.
        // Scattered, not gridded: a regular grid aliases in-plane and pins
        // the solve to whole grid steps, which no real surface does.
        let corner = |dx: f32| {
            let mut seed = 0x2545_f491_u32;
            let mut rnd = move || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as f32 / u32::MAX as f32 * 3.0 - 1.5
            };
            let mut pts = Vec::new();
            let mut push = |x: f32, y: f32, z: f32| {
                pts.push(Point3D {
                    x,
                    y,
                    z,
                    reflectivity: 0,
                    tag: 0,
                })
            };
            for _ in 0..3000 {
                let (a, b) = (rnd(), rnd());
                push(a + 1.0 + dx, b, 0.0);
                let (a, b) = (rnd(), rnd());
                push(2.0 + dx, a, b + 1.5);
                let (a, b) = (rnd(), rnd());
                push(a + 1.0 + dx, 1.5, b + 1.5);
            }
            pts
        };
        let (rp, cp) = (corner(0.0), corner(-0.04));
        let stereo = |pts| StereoSource {
            points: pts,
            focal_px: 400.0,
            baseline_m: 0.06,
            disparity_sigma_px: 0.3,
            pixel_sigma_px: 0.5,
            half_fov_h: 1.2,
            half_fov_v: 1.2,
            min_range: 0.2,
            max_range: 6.0,
        };
        let matcher = CorrelativeScanMatcher::new(ScanMatchConfig {
            resolution: 0.1,
            ..Default::default()
        });
        let res = matcher
            .match_sources(&stereo(&rp), &stereo(&cp), Transform2D::identity())
            .unwrap();
        assert!((res.pose6[0] - 0.04).abs() < 5e-3, "x {}", res.pose6[0]);
        assert!(res.degeneracy.is_well_conditioned());
        // Ray-aligned noise shows up as looser x than a LiDAR would give.
        assert!(res.covariance6[(0, 0)] > 0.0);
    }
}
