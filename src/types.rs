//! Core types for point clouds and poses.

use serde::{Deserialize, Serialize};

/// 2D pose: position and heading.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Pose {
    /// X position in meters
    pub x: f64,
    /// Y position in meters
    pub y: f64,
    /// Heading in radians (positive = counter-clockwise)
    pub theta: f64,
}

/// A single 3D point from a LiDAR scan.
#[derive(Debug, Clone, Copy, Default)]
pub struct Point3D {
    /// X coordinate in meters (forward)
    pub x: f32,
    /// Y coordinate in meters (left)
    pub y: f32,
    /// Z coordinate in meters (up)
    pub z: f32,
    /// Reflectivity (0-255)
    pub reflectivity: u8,
    /// Tag/classification
    pub tag: u8,
}

/// A point cloud frame from a LiDAR sensor.
#[derive(Debug, Clone, Default)]
pub struct PointCloud {
    /// Point cloud data
    pub points: Vec<Point3D>,
    /// Frame sequence number
    pub frame_id: u32,
}

impl PointCloud {
    /// Create a new point cloud with the given points.
    pub fn new(points: Vec<Point3D>) -> Self {
        Self {
            points,
            frame_id: 0,
        }
    }

    /// Create an empty point cloud.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Number of points in the cloud.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the cloud is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}
