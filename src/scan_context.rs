//! Scan Context descriptors for efficient loop closure detection.
//!
//! Computes a compact 2D descriptor (rings × sectors) from an unorganized
//! 3D point cloud by binning points into polar cells and recording max height.
//! Rotation-invariant matching is achieved via column-shift cosine distance.
//!
//! Reference: Kim & Kim, "Scan Context: Egocentric Spatial Descriptor for
//! Place Recognition within 3D Point Cloud Map" (2018).

use nalgebra::Vector3;

/// Configuration for Scan Context descriptors.
#[derive(Debug, Clone)]
pub struct ScanContextConfig {
    /// Number of radial rings
    pub num_rings: usize,
    /// Number of azimuthal sectors
    pub num_sectors: usize,
    /// Maximum range to consider (meters)
    pub max_range: f64,
    /// Number of top candidates to return from database query
    pub top_k: usize,
    /// Descriptor distance threshold for candidate acceptance
    pub descriptor_threshold: f32,
    /// Number of ring-key pre-filter candidates before full cosine distance
    pub ring_key_prefilter: usize,
}

impl Default for ScanContextConfig {
    fn default() -> Self {
        Self {
            num_rings: 20,
            num_sectors: 60,
            max_range: 40.0,
            top_k: 5,
            descriptor_threshold: 0.3,
            ring_key_prefilter: 20,
        }
    }
}

/// A Scan Context descriptor computed from a point cloud.
#[derive(Debug, Clone)]
pub struct ScanContextDescriptor {
    /// Max-height matrix (rings × sectors), stored row-major
    pub matrix: Vec<f32>,
    /// Ring key: mean value per ring (used for fast pre-filtering)
    pub ring_key: Vec<f32>,
    /// Sector key: mean value per sector
    pub sector_key: Vec<f32>,
    /// Number of rings
    pub(crate) num_rings: usize,
    /// Number of sectors
    pub(crate) num_sectors: usize,
}

impl ScanContextDescriptor {
    /// Compute a Scan Context descriptor from a 3D point cloud.
    pub fn from_points(points: &[Vector3<f64>], config: &ScanContextConfig) -> Self {
        let nr = config.num_rings;
        let ns = config.num_sectors;
        let max_range = config.max_range;
        let ring_gap = max_range / nr as f64;
        let sector_gap = std::f64::consts::TAU / ns as f64;

        let mut matrix = vec![f32::NEG_INFINITY; nr * ns];

        for p in points {
            let range = (p.x * p.x + p.y * p.y).sqrt();
            if range < 0.1 || range > max_range {
                continue;
            }

            let azimuth = p.y.atan2(p.x);
            let azimuth = if azimuth < 0.0 {
                azimuth + std::f64::consts::TAU
            } else {
                azimuth
            };

            let ring_idx = ((range / ring_gap) as usize).min(nr - 1);
            let sector_idx = ((azimuth / sector_gap) as usize).min(ns - 1);

            let cell = ring_idx * ns + sector_idx;
            let z = p.z as f32;
            if z > matrix[cell] {
                matrix[cell] = z;
            }
        }

        for v in &mut matrix {
            if v.is_infinite() {
                *v = 0.0;
            }
        }

        let ring_key: Vec<f32> = (0..nr)
            .map(|r| {
                let row = &matrix[r * ns..(r + 1) * ns];
                let sum: f32 = row.iter().sum();
                sum / ns as f32
            })
            .collect();

        let sector_key: Vec<f32> = (0..ns)
            .map(|s| {
                let sum: f32 = (0..nr).map(|r| matrix[r * ns + s]).sum();
                sum / nr as f32
            })
            .collect();

        Self {
            matrix,
            ring_key,
            sector_key,
            num_rings: nr,
            num_sectors: ns,
        }
    }

    /// Compute the column-shift cosine distance to another descriptor.
    pub fn distance(&self, other: &ScanContextDescriptor) -> (f32, usize) {
        assert_eq!(self.num_sectors, other.num_sectors);
        assert_eq!(self.num_rings, other.num_rings);

        let ns = self.num_sectors;
        let nr = self.num_rings;

        let mut best_dist = f32::MAX;
        let mut best_shift = 0;

        for shift in 0..ns {
            let mut dot = 0.0_f64;
            let mut norm_a = 0.0_f64;
            let mut norm_b = 0.0_f64;

            for r in 0..nr {
                for s in 0..ns {
                    let a = self.matrix[r * ns + s] as f64;
                    let b_s = (s + shift) % ns;
                    let b = other.matrix[r * ns + b_s] as f64;

                    dot += a * b;
                    norm_a += a * a;
                    norm_b += b * b;
                }
            }

            let denom = (norm_a * norm_b).sqrt();
            let cosine = if denom > 1e-12 { dot / denom } else { 0.0 };
            let dist = (1.0 - cosine) as f32;

            if dist < best_dist {
                best_dist = dist;
                best_shift = shift;
            }
        }

        (best_dist, best_shift)
    }

    /// Compute L2 distance between ring keys (fast pre-filter).
    pub fn ring_key_distance(&self, other: &ScanContextDescriptor) -> f32 {
        assert_eq!(self.ring_key.len(), other.ring_key.len());
        self.ring_key
            .iter()
            .zip(other.ring_key.iter())
            .map(|(a, b)| {
                let d = a - b;
                d * d
            })
            .sum::<f32>()
            .sqrt()
    }
}

/// Database of Scan Context descriptors for fast candidate retrieval.
pub struct ScanContextDatabase {
    config: ScanContextConfig,
    entries: Vec<(usize, ScanContextDescriptor)>,
}

impl ScanContextDatabase {
    /// Create a new empty database.
    pub fn new(config: ScanContextConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
        }
    }

    /// Insert a descriptor for a keyframe.
    pub fn insert(&mut self, keyframe_id: usize, descriptor: ScanContextDescriptor) {
        self.entries.push((keyframe_id, descriptor));
    }

    /// Find the top-K candidate keyframe IDs for a query descriptor.
    pub fn find_candidates(&self, query: &ScanContextDescriptor) -> Vec<ScanContextCandidate> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        let prefilter_n = self.config.ring_key_prefilter.min(self.entries.len());
        let mut ring_dists: Vec<(usize, f32)> = self
            .entries
            .iter()
            .map(|(id, desc)| (*id, query.ring_key_distance(desc)))
            .collect();
        ring_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut candidates: Vec<ScanContextCandidate> = ring_dists
            .iter()
            .take(prefilter_n)
            .filter_map(|(id, _)| {
                let entry = self.entries.iter().find(|(eid, _)| eid == id)?;
                let (dist, shift) = query.distance(&entry.1);
                if dist <= self.config.descriptor_threshold {
                    Some(ScanContextCandidate {
                        keyframe_id: *id,
                        distance: dist,
                        best_shift: shift,
                    })
                } else {
                    None
                }
            })
            .collect();

        candidates.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(self.config.top_k);
        candidates
    }

    /// Number of entries in the database.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if database is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries (for serialization).
    pub fn entries(&self) -> &[(usize, ScanContextDescriptor)] {
        &self.entries
    }

    /// Get the config.
    pub fn config(&self) -> &ScanContextConfig {
        &self.config
    }
}

/// A candidate loop closure match from the scan context database.
#[derive(Debug, Clone)]
pub struct ScanContextCandidate {
    /// Keyframe ID of the candidate
    pub keyframe_id: usize,
    /// Scan context distance (lower = better, range [0, 2])
    pub distance: f32,
    /// Best column shift (corresponds to azimuthal rotation)
    pub best_shift: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_cloud(num_points: usize, z_offset: f64) -> Vec<Vector3<f64>> {
        let mut points = Vec::with_capacity(num_points);
        let angle_step = std::f64::consts::TAU / num_points as f64;
        for i in 0..num_points {
            let angle = i as f64 * angle_step;
            let range = 5.0 + 2.0 * (angle * 3.0).sin();
            points.push(Vector3::new(
                range * angle.cos(),
                range * angle.sin(),
                1.0 + z_offset + 0.5 * (angle * 2.0).cos(),
            ));
        }
        points
    }

    #[test]
    fn test_descriptor_from_points() {
        let config = ScanContextConfig::default();
        let cloud = make_synthetic_cloud(1000, 0.0);
        let desc = ScanContextDescriptor::from_points(&cloud, &config);

        assert_eq!(desc.ring_key.len(), config.num_rings);
        assert_eq!(desc.sector_key.len(), config.num_sectors);
        assert_eq!(desc.matrix.len(), config.num_rings * config.num_sectors);
    }

    #[test]
    fn test_self_distance_is_zero() {
        let config = ScanContextConfig::default();
        let cloud = make_synthetic_cloud(1000, 0.0);
        let desc = ScanContextDescriptor::from_points(&cloud, &config);

        let (dist, shift) = desc.distance(&desc);
        assert!(dist < 1e-5, "Self-distance should be ~0, got {}", dist);
        assert_eq!(shift, 0);
    }

    #[test]
    fn test_empty_cloud() {
        let config = ScanContextConfig::default();
        let desc = ScanContextDescriptor::from_points(&[], &config);
        assert!(desc.ring_key.iter().all(|v| *v == 0.0));
        assert!(desc.sector_key.iter().all(|v| *v == 0.0));
    }
}
