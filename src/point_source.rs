//! Range sensors as point sources.
//!
//! GICP only needs points in the sensor frame, how noisy each one is, and
//! where the sensor can see. A LiDAR, a stereo pair and an RGB-D camera all
//! answer those three questions differently; [`PointSource`] is the shape the
//! scan matcher consumes, so it never needs to know which one it got.

use nalgebra::{Matrix3, Vector3};

use crate::types::{Point3D, PointCloud};

/// A sensor's points in its own frame, with per-point measurement noise.
///
/// Frames follow the cloud they come from: x forward, y left, z up for
/// LiDAR; implementors with optical frames (z forward) convert before
/// handing points out, so the matcher sees one convention.
pub trait PointSource {
    /// Number of raw points.
    fn len(&self) -> usize;

    /// True when there are no points.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The i-th point in the sensor frame (meters).
    fn point(&self, i: usize) -> Vector3<f64>;

    /// Measurement covariance of the i-th point (m²), sensor frame.
    ///
    /// This is the sensor's own noise, added to the local surface covariance
    /// GICP estimates from neighbours. Zero means "trust the geometry".
    fn covariance(&self, i: usize) -> Matrix3<f64>;

    /// Ranges outside `[min, max]` are dropped before matching (meters).
    fn range_limits(&self) -> (f64, f64);

    /// Whether a sensor-frame point lies inside the field of view.
    ///
    /// The matcher drops points that fail this, which keeps returns from
    /// lens edges or behind a stereo pair out of the solve.
    fn in_fov(&self, _p: &Vector3<f64>) -> bool {
        true
    }
}

/// A spinning or solid-state LiDAR cloud.
///
/// Range noise is small and isotropic next to surface covariance, so the
/// sensor term is zero by default: matching a [`PointCloud`] through this
/// wrapper is the same solve slamwich has always run.
#[derive(Debug, Clone, Copy)]
pub struct LidarSource<'a> {
    /// Points as returned by the sensor.
    pub cloud: &'a PointCloud,
    /// Isotropic range noise (m). Zero keeps legacy behaviour.
    pub sigma: f64,
    /// Minimum usable range (m).
    pub min_range: f64,
    /// Maximum usable range (m).
    pub max_range: f64,
}

impl<'a> LidarSource<'a> {
    /// Wrap a cloud with the limits slamwich has always applied (0.1–50 m).
    pub fn new(cloud: &'a PointCloud) -> Self {
        Self {
            cloud,
            sigma: 0.0,
            min_range: 0.1,
            max_range: 50.0,
        }
    }
}

impl PointSource for LidarSource<'_> {
    fn len(&self) -> usize {
        self.cloud.points.len()
    }

    fn point(&self, i: usize) -> Vector3<f64> {
        let p = &self.cloud.points[i];
        Vector3::new(p.x as f64, p.y as f64, p.z as f64)
    }

    fn covariance(&self, _i: usize) -> Matrix3<f64> {
        Matrix3::identity() * (self.sigma * self.sigma)
    }

    fn range_limits(&self) -> (f64, f64) {
        (self.min_range, self.max_range)
    }
}

/// Points triangulated by a calibrated stereo pair (or depth from one).
///
/// Disparity noise σ_d (pixels) maps to depth noise along the ray as
/// σ_z = z² σ_d / (f b) and to lateral noise as σ_⊥ = z σ_px / f, so far
/// points are long thin ellipsoids pointing back at the camera. Points are
/// in body convention (x forward, y left, z up) relative to the left eye.
#[derive(Debug, Clone)]
pub struct StereoSource<'a> {
    /// Points, x forward along the optical axis.
    pub points: &'a [Point3D],
    /// Focal length (pixels).
    pub focal_px: f64,
    /// Baseline (meters).
    pub baseline_m: f64,
    /// Disparity noise (pixels).
    pub disparity_sigma_px: f64,
    /// Pixel-position noise for the lateral axes (pixels).
    pub pixel_sigma_px: f64,
    /// Horizontal half field of view (radians).
    pub half_fov_h: f64,
    /// Vertical half field of view (radians).
    pub half_fov_v: f64,
    /// Minimum usable depth (m).
    pub min_range: f64,
    /// Maximum usable depth (m); past this σ_z dwarfs the voxel.
    pub max_range: f64,
}

impl StereoSource<'_> {
    /// Depth standard deviation at depth `z` (meters).
    pub fn sigma_z(&self, z: f64) -> f64 {
        z * z * self.disparity_sigma_px / (self.focal_px * self.baseline_m)
    }
}

impl PointSource for StereoSource<'_> {
    fn len(&self) -> usize {
        self.points.len()
    }

    fn point(&self, i: usize) -> Vector3<f64> {
        let p = &self.points[i];
        Vector3::new(p.x as f64, p.y as f64, p.z as f64)
    }

    fn covariance(&self, i: usize) -> Matrix3<f64> {
        let p = self.point(i);
        let r = p.norm();
        if r < 1e-9 {
            return Matrix3::zeros();
        }
        let ray = p / r;
        let z = p.x.max(1e-3);
        let s_ray = self.sigma_z(z);
        let s_lat = z * self.pixel_sigma_px / self.focal_px;
        // σ_lat² I + (σ_ray² − σ_lat²) r̂ r̂ᵀ: long along the ray, thin across.
        Matrix3::identity() * (s_lat * s_lat)
            + ray * ray.transpose() * (s_ray * s_ray - s_lat * s_lat)
    }

    fn range_limits(&self) -> (f64, f64) {
        (self.min_range, self.max_range)
    }

    fn in_fov(&self, p: &Vector3<f64>) -> bool {
        p.x > 0.0
            && (p.y / p.x).atan().abs() <= self.half_fov_h
            && (p.z / p.x).atan().abs() <= self.half_fov_v
    }
}

/// A depth camera that measures range directly (time-of-flight or
/// structured light): depth noise grows as `a + b z²`, lateral as pixels.
#[derive(Debug, Clone)]
pub struct RgbdSource<'a> {
    /// Points, x forward along the optical axis.
    pub points: &'a [Point3D],
    /// Focal length (pixels).
    pub focal_px: f64,
    /// Constant depth noise term (m).
    pub sigma_z0: f64,
    /// Quadratic depth noise coefficient (1/m).
    pub sigma_z2: f64,
    /// Horizontal half field of view (radians).
    pub half_fov_h: f64,
    /// Vertical half field of view (radians).
    pub half_fov_v: f64,
    /// Minimum usable depth (m).
    pub min_range: f64,
    /// Maximum usable depth (m).
    pub max_range: f64,
}

impl PointSource for RgbdSource<'_> {
    fn len(&self) -> usize {
        self.points.len()
    }

    fn point(&self, i: usize) -> Vector3<f64> {
        let p = &self.points[i];
        Vector3::new(p.x as f64, p.y as f64, p.z as f64)
    }

    fn covariance(&self, i: usize) -> Matrix3<f64> {
        let p = self.point(i);
        let z = p.x.max(1e-3);
        let s_z = self.sigma_z0 + self.sigma_z2 * z * z;
        let s_lat = z / self.focal_px;
        let mut c = Matrix3::identity() * (s_lat * s_lat);
        c[(0, 0)] = s_z * s_z;
        c
    }

    fn range_limits(&self) -> (f64, f64) {
        (self.min_range, self.max_range)
    }

    fn in_fov(&self, p: &Vector3<f64>) -> bool {
        p.x > 0.0
            && (p.y / p.x).atan().abs() <= self.half_fov_h
            && (p.z / p.x).atan().abs() <= self.half_fov_v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo(points: &[Point3D]) -> StereoSource<'_> {
        StereoSource {
            points,
            focal_px: 500.0,
            baseline_m: 0.06,
            disparity_sigma_px: 0.5,
            pixel_sigma_px: 0.5,
            half_fov_h: 0.8,
            half_fov_v: 0.6,
            min_range: 0.2,
            max_range: 6.0,
        }
    }

    #[test]
    fn stereo_depth_noise_is_quadratic_and_along_the_ray() {
        let pts = [
            Point3D {
                x: 1.0,
                ..Default::default()
            },
            Point3D {
                x: 2.0,
                ..Default::default()
            },
        ];
        let s = stereo(&pts);
        let c1 = s.covariance(0);
        let c2 = s.covariance(1);
        // σz(1 m) = 0.5 / 30 m; doubling depth quadruples σz.
        assert!((c1[(0, 0)].sqrt() - 0.5 / 30.0).abs() < 1e-12);
        assert!((c2[(0, 0)].sqrt() / c1[(0, 0)].sqrt() - 4.0).abs() < 1e-9);
        // Lateral noise grows linearly and is far smaller than depth noise.
        assert!((c2[(1, 1)].sqrt() / c1[(1, 1)].sqrt() - 2.0).abs() < 1e-9);
        assert!(c1[(1, 1)] < c1[(0, 0)] / 10.0);
    }

    #[test]
    fn stereo_fov_rejects_behind_and_wide() {
        let s = stereo(&[]);
        assert!(s.in_fov(&Vector3::new(1.0, 0.0, 0.0)));
        assert!(!s.in_fov(&Vector3::new(-1.0, 0.0, 0.0)));
        assert!(!s.in_fov(&Vector3::new(1.0, 2.0, 0.0)));
    }

    #[test]
    fn lidar_default_adds_no_sensor_noise() {
        let cloud = PointCloud::new(vec![Point3D {
            x: 3.0,
            ..Default::default()
        }]);
        let s = LidarSource::new(&cloud);
        assert_eq!(s.covariance(0), Matrix3::zeros());
        assert_eq!(s.range_limits(), (0.1, 50.0));
    }
}
