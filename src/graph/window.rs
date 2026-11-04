//! The sliding window: keyframes in, oldest marginalised out.

use std::collections::VecDeque;

use nalgebra::DMatrix;

use crate::graph::factors::{GroundPlane, Prior, sigmas};
use crate::graph::core::{Graph, Key, Kind, SolveConfig, SolveReport};

#[derive(Clone, Debug)]
pub struct SmootherConfig {
    /// Pose keyframes kept live; older ones are marginalised.
    pub window: usize,
    pub solve: SolveConfig,
    /// Sensor/body height above the floor for the ground-plane prior, if any.
    pub ground_height: Option<f64>,
    /// σ for `[z, roll, pitch]` of the ground-plane prior.
    pub ground_sigmas: [f64; 3],
    /// σ of the first pose's prior (fixes the gauge).
    pub origin_sigmas: [f64; 6],
}

impl Default for SmootherConfig {
    fn default() -> Self {
        Self {
            window: 20,
            solve: SolveConfig::default(),
            ground_height: None,
            ground_sigmas: [0.02, 0.03, 0.03],
            origin_sigmas: [1e-3; 6],
        }
    }
}

/// A sliding-window smoother over poses plus persistent blocks
/// (landmarks, sensor or body parameters) that are never marginalised.
pub struct Smoother {
    pub cfg: SmootherConfig,
    pub graph: Graph,
    poses: VecDeque<Key>,
}

impl Smoother {
    /// Start at `origin`.
    pub fn new(cfg: SmootherConfig, origin: [f64; 6]) -> Self {
        let mut graph = Graph::new();
        let p0 = graph.add(Kind::Pose, origin.to_vec(), false);
        graph.push(Prior::new(
            p0,
            Kind::Pose,
            origin.to_vec(),
            sigmas(&cfg.origin_sigmas),
        ));
        let mut s = Self {
            cfg,
            graph,
            poses: VecDeque::new(),
        };
        s.poses.push_back(p0);
        s.ground(p0);
        s
    }

    fn ground(&mut self, k: Key) {
        if let Some(h) = self.cfg.ground_height {
            let l = sigmas(&self.cfg.ground_sigmas);
            self.graph.push(GroundPlane::new(k, h, l));
        }
    }

    /// The newest keyframe.
    pub fn head(&self) -> Key {
        *self.poses.back().unwrap()
    }

    /// Live keyframes, oldest first.
    pub fn poses(&self) -> impl Iterator<Item = Key> + '_ {
        self.poses.iter().copied()
    }

    /// Add a keyframe at `guess`; the caller links it with factors.
    pub fn add_keyframe(&mut self, guess: [f64; 6]) -> Key {
        let k = self.graph.add(Kind::Pose, guess.to_vec(), false);
        self.poses.push_back(k);
        self.ground(k);
        k
    }

    /// Add a persistent block (a tag, a bias) that the window never drops.
    pub fn add_landmark(&mut self, guess: [f64; 6]) -> Key {
        self.graph.add(Kind::Pose, guess.to_vec(), false)
    }

    /// Solve, then marginalise keyframes beyond the window.
    pub fn optimize(&mut self) -> SolveReport {
        let report = self.graph.solve(&self.cfg.solve);
        while self.poses.len() > self.cfg.window {
            let old = self.poses.pop_front().unwrap();
            self.graph.marginalize(old);
        }
        report
    }

    /// Pose of a block.
    pub fn pose(&self, k: Key) -> [f64; 6] {
        let v = self.graph.value(k);
        std::array::from_fn(|i| v[i])
    }

    /// Marginal 6×6 covariance of a block at the current estimate.
    pub fn covariance(&self, k: Key) -> Option<DMatrix<f64>> {
        self.graph.marginal(k)
    }
}
