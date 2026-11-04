//! A sliding-window factor-graph smoother whose residuals are tang
//! expressions (feature `tang`).
//!
//! The alternative to [`crate::SlamProcessor`]'s EKF + separate pose graph:
//! one graph holds odometry, IMU, scan matches (with their degeneracy-aware
//! covariance), landmarks, a ground plane and loop closures, solved together
//! by Levenberg-Marquardt with Huber, marginalising old keyframes into a
//! linear prior. Every residual is written on a [`tang_ad`] tape and its
//! Jacobian is the tape's reverse sweep, one per residual row; nothing is
//! differentiated by hand. Variables can be held fixed, so sensor or body
//! parameters can sit in the same graph as fixed inputs online and be
//! freed for an offline re-solve.

pub mod core;
pub mod expr;
pub mod factors;
pub mod window;

pub use self::core::{
    linearize, Factor, Graph, Key, Kind, Linear, Robust, SolveConfig, SolveReport, Variable,
};
pub use factors::{sigmas, sqrt_info, Between, GroundPlane, Gyro, LinearPrior, Odometry, Prior, Rel};
pub use window::{Smoother, SmootherConfig};
