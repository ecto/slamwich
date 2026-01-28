//! IMU pre-integration between LiDAR scans.
//!
//! Accumulates gyro-Z samples at ~200Hz between ~10Hz LiDAR scans,
//! producing a pre-integrated yaw delta with variance estimate.

use crate::transforms::Transform2D;

/// Configuration for IMU pre-integration.
#[derive(Debug, Clone)]
pub struct ImuPreintegrationConfig {
    /// Gyro noise density (rad/s/sqrt(Hz)). BMI088 typical: 0.014
    pub gyro_noise_density: f64,
    /// Gyro bias random walk (rad/s²/sqrt(Hz)). BMI088 typical: 0.0003
    pub gyro_bias_rw: f64,
    /// Enable IMU pre-integration
    pub enabled: bool,
}

impl Default for ImuPreintegrationConfig {
    fn default() -> Self {
        Self {
            gyro_noise_density: 0.014,
            gyro_bias_rw: 0.0003,
            enabled: true,
        }
    }
}

/// Result of consuming accumulated IMU samples between scans.
#[derive(Debug, Clone, Copy)]
pub struct PreintegratedDelta {
    /// Accumulated yaw rotation (radians)
    pub delta_yaw: f64,
    /// Variance of the accumulated yaw (rad²)
    pub yaw_variance: f64,
    /// Number of samples integrated
    pub sample_count: u32,
}

/// Accumulates gyro-Z samples between LiDAR scans.
pub struct ImuPreintegrator {
    config: ImuPreintegrationConfig,
    delta_yaw: f64,
    yaw_variance: f64,
    sample_count: u32,
    gyro_bias: f64,
    stationary_threshold: f64,
}

impl ImuPreintegrator {
    /// Create a new pre-integrator.
    pub fn new(config: ImuPreintegrationConfig) -> Self {
        Self {
            config,
            delta_yaw: 0.0,
            yaw_variance: 0.0,
            sample_count: 0,
            gyro_bias: 0.0,
            stationary_threshold: 0.01,
        }
    }

    /// Integrate a single gyro-Z sample.
    pub fn integrate(&mut self, gyro_z_body: f64, dt: f64) {
        if !self.config.enabled || dt <= 0.0 || !dt.is_finite() {
            return;
        }

        let rate = gyro_z_body - self.gyro_bias;

        if rate.abs() < self.stationary_threshold {
            self.gyro_bias += (gyro_z_body - self.gyro_bias) * 0.001;
            self.sample_count += 1;
            return;
        }

        self.delta_yaw += rate * dt;

        let nd = self.config.gyro_noise_density;
        let brw = self.config.gyro_bias_rw;
        self.yaw_variance += nd * nd * dt + brw * brw * dt;

        self.sample_count += 1;
    }

    /// Consume the accumulated pre-integrated delta, resetting for the next interval.
    pub fn consume(&mut self) -> Option<PreintegratedDelta> {
        if !self.config.enabled || self.sample_count == 0 {
            return None;
        }

        let delta = PreintegratedDelta {
            delta_yaw: self.delta_yaw,
            yaw_variance: self.yaw_variance,
            sample_count: self.sample_count,
        };

        self.delta_yaw = 0.0;
        self.yaw_variance = 0.0;
        self.sample_count = 0;

        Some(delta)
    }

    /// Check if pre-integration is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

/// Compute a hybrid initial guess combining odometry translation with IMU yaw.
pub fn compute_initial_guess(
    odom_delta: &Transform2D,
    imu_delta: Option<&PreintegratedDelta>,
) -> Transform2D {
    match imu_delta {
        Some(delta) if delta.sample_count > 0 => Transform2D::new(
            odom_delta.translation().x,
            odom_delta.translation().y,
            delta.delta_yaw,
        ),
        _ => *odom_delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_accumulation() {
        let config = ImuPreintegrationConfig::default();
        let mut pre = ImuPreintegrator::new(config);

        let dt = 0.005;
        let rate = 1.0_f64;
        for _ in 0..20 {
            pre.integrate(rate, dt);
        }

        let delta = pre.consume().unwrap();
        assert!(
            (delta.delta_yaw - 0.1).abs() < 0.01,
            "Expected ~0.1 rad, got {}",
            delta.delta_yaw
        );
        assert!(delta.yaw_variance > 0.0);
        assert_eq!(delta.sample_count, 20);
    }

    #[test]
    fn test_stationary_skip() {
        let config = ImuPreintegrationConfig::default();
        let mut pre = ImuPreintegrator::new(config);

        let dt = 0.005;
        for _ in 0..20 {
            pre.integrate(0.005, dt);
        }

        let delta = pre.consume().unwrap();
        assert!(
            delta.delta_yaw.abs() < 0.001,
            "Expected ~0 rad, got {}",
            delta.delta_yaw
        );
    }

    #[test]
    fn test_consume_resets() {
        let config = ImuPreintegrationConfig::default();
        let mut pre = ImuPreintegrator::new(config);

        pre.integrate(1.0, 0.005);
        let first = pre.consume().unwrap();
        assert!(first.delta_yaw.abs() > 0.0);

        assert!(pre.consume().is_none());
    }

    #[test]
    fn test_disabled() {
        let config = ImuPreintegrationConfig {
            enabled: false,
            ..Default::default()
        };
        let mut pre = ImuPreintegrator::new(config);

        pre.integrate(1.0, 0.005);
        assert!(pre.consume().is_none());
    }

    #[test]
    fn test_compute_initial_guess_with_imu() {
        let odom = Transform2D::new(0.5, 0.1, 0.3);
        let imu = PreintegratedDelta {
            delta_yaw: 0.5,
            yaw_variance: 0.001,
            sample_count: 20,
        };

        let guess = compute_initial_guess(&odom, Some(&imu));
        assert!((guess.translation().x - 0.5).abs() < 1e-10);
        assert!((guess.translation().y - 0.1).abs() < 1e-10);
        assert!((guess.rotation() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_initial_guess_fallback() {
        let odom = Transform2D::new(0.5, 0.1, 0.3);
        let guess = compute_initial_guess(&odom, None);
        assert!((guess.rotation() - 0.3).abs() < 1e-10);
    }
}
