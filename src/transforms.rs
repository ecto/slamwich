//! 2D rigid body transforms for SLAM.

use nalgebra::{Isometry2, Vector2};
use std::f64::consts::PI;

use crate::types::Pose;

/// A 2D rigid body transform (translation + rotation).
#[derive(Debug, Clone, Copy)]
pub struct Transform2D {
    inner: Isometry2<f64>,
}

impl Transform2D {
    /// Create identity transform.
    pub fn identity() -> Self {
        Self {
            inner: Isometry2::identity(),
        }
    }

    /// Create transform from translation and rotation angle.
    pub fn new(x: f64, y: f64, theta: f64) -> Self {
        Self {
            inner: Isometry2::new(Vector2::new(x, y), theta),
        }
    }

    /// Create from a Pose struct.
    pub fn from_pose(pose: &Pose) -> Self {
        Self::new(pose.x, pose.y, pose.theta)
    }

    /// Convert to a Pose struct.
    pub fn to_pose(&self) -> Pose {
        Pose {
            x: self.inner.translation.x,
            y: self.inner.translation.y,
            theta: self.inner.rotation.angle(),
        }
    }

    /// Get the underlying nalgebra Isometry2.
    pub fn as_isometry(&self) -> &Isometry2<f64> {
        &self.inner
    }

    /// Create from nalgebra Isometry2.
    pub fn from_isometry(iso: Isometry2<f64>) -> Self {
        Self { inner: iso }
    }

    /// Get translation component.
    pub fn translation(&self) -> Vector2<f64> {
        self.inner.translation.vector
    }

    /// Get rotation angle in radians.
    pub fn rotation(&self) -> f64 {
        self.inner.rotation.angle()
    }

    /// Compute inverse transform.
    pub fn inverse(&self) -> Self {
        Self {
            inner: self.inner.inverse(),
        }
    }

    /// Compose transforms: self * other.
    /// If self is A->B and other is B->C, result is A->C.
    pub fn compose(&self, other: &Transform2D) -> Transform2D {
        Transform2D {
            inner: self.inner * other.inner,
        }
    }

    /// Transform a point from child frame to parent frame.
    pub fn transform_point(&self, point: Vector2<f64>) -> Vector2<f64> {
        self.inner
            .transform_point(&nalgebra::Point2::from(point))
            .coords
    }

    /// Transform a pose from child frame to parent frame.
    pub fn transform_pose(&self, pose: &Pose) -> Pose {
        let child_iso = Isometry2::new(Vector2::new(pose.x, pose.y), pose.theta);
        let result = self.inner * child_iso;
        Pose {
            x: result.translation.x,
            y: result.translation.y,
            theta: normalize_angle(result.rotation.angle()),
        }
    }

    /// Compute relative transform: from self to other.
    /// If self is A and other is B (both in same frame), returns A->B transform.
    pub fn relative_to(&self, other: &Transform2D) -> Transform2D {
        Transform2D {
            inner: self.inner.inverse() * other.inner,
        }
    }
}

impl Default for Transform2D {
    fn default() -> Self {
        Self::identity()
    }
}

impl std::ops::Mul for Transform2D {
    type Output = Transform2D;

    fn mul(self, rhs: Transform2D) -> Transform2D {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<&Transform2D> for Transform2D {
    type Output = Transform2D;

    fn mul(self, rhs: &Transform2D) -> Transform2D {
        self.compose(rhs)
    }
}

impl std::ops::Mul<Transform2D> for &Transform2D {
    type Output = Transform2D;

    fn mul(self, rhs: Transform2D) -> Transform2D {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<&Transform2D> for &Transform2D {
    type Output = Transform2D;

    fn mul(self, rhs: &Transform2D) -> Transform2D {
        self.compose(rhs)
    }
}

/// Normalize angle to [-PI, PI).
pub fn normalize_angle(angle: f64) -> f64 {
    let mut a = angle % (2.0 * PI);
    if a >= PI {
        a -= 2.0 * PI;
    } else if a < -PI {
        a += 2.0 * PI;
    }
    a
}

/// Compute shortest angular difference from a to b.
pub fn angle_diff(a: f64, b: f64) -> f64 {
    normalize_angle(b - a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_transform_identity() {
        let tf = Transform2D::identity();
        assert_relative_eq!(tf.translation().x, 0.0);
        assert_relative_eq!(tf.translation().y, 0.0);
        assert_relative_eq!(tf.rotation(), 0.0);
    }

    #[test]
    fn test_transform_from_pose() {
        let pose = Pose {
            x: 1.0,
            y: 2.0,
            theta: PI / 4.0,
        };
        let tf = Transform2D::from_pose(&pose);
        let back = tf.to_pose();
        assert_relative_eq!(back.x, pose.x, epsilon = 1e-10);
        assert_relative_eq!(back.y, pose.y, epsilon = 1e-10);
        assert_relative_eq!(back.theta, pose.theta, epsilon = 1e-10);
    }

    #[test]
    fn test_transform_inverse() {
        let tf = Transform2D::new(1.0, 2.0, PI / 2.0);
        let inv = tf.inverse();
        let composed = tf.compose(&inv);
        assert_relative_eq!(composed.translation().x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(composed.translation().y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(composed.rotation(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_angle() {
        assert_relative_eq!(normalize_angle(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(normalize_angle(PI).abs(), PI, epsilon = 1e-10);
        assert_relative_eq!(normalize_angle(-PI), -PI, epsilon = 1e-10);
        assert_relative_eq!(normalize_angle(2.0 * PI), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_angle_diff() {
        assert_relative_eq!(angle_diff(0.0, PI / 2.0), PI / 2.0, epsilon = 1e-10);
        assert_relative_eq!(angle_diff(PI / 2.0, 0.0), -PI / 2.0, epsilon = 1e-10);
    }
}
