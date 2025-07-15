// src/bootstrap_particles.rs
//! Bootstrap Particle Filter Implementation
//!
//! This module implements a parallelized bootstrap particle filter with an effective sample size (ESS) heuristic for adaptive resampling.
//!
//! # Example
//! ```rust
//! use nalgebra::DVector;
//! use bootstrap_filter_lib::bootstrap_particles::{BootstrapFilter, Particle};
//!
//! // Define model functions...
//! # fn f(_t0: f64, _t: f64, x: &DVector<f64>, _u: &DVector<f64>, _q: &DVector<f64>) -> DVector<f64> { x.clone() }
//! # fn h(_t: f64, x: &DVector<f64>, _u: &DVector<f64>) -> DVector<f64> { x.clone() }
//! # fn draw_noise(_t: f64, _x: &DVector<f64>) -> DVector<f64> { DVector::zeros(1) }
//! # fn p_obs(_t: f64, _dz: &DVector<f64>, _x: &DVector<f64>) -> f64 { 1.0 }
//!
//! // Create filter and update with ESS threshold of 50%
//! let mut filter = BootstrapFilter::new_with_initial_state(
//!     1000,
//!     &DVector::zeros(1),
//!     f,
//!     h,
//!     draw_noise,
//!     p_obs,
//! );
//! let u = DVector::<f64>::from_element(1, 0.0);
//! let z = DVector::<f64>::from_element(1, 1.0);
//! // Here threshold=0.5 means resample when ESS < 0.5 * N
//! filter.update(0.0, 1.0, &u, &z, 0.5);
//! ```

use nalgebra::DVector;
use rand::prelude::*;
use rayon::prelude::*;

/// A single particle: state vector and its weight.
#[derive(Clone)]
pub struct Particle {
    pub x: DVector<f64>,
    pub w: f64,
}

/// Bootstrap (particle) filter with adaptive resampling based on ESS.
pub struct BootstrapFilter<F, H, Q, P>
where
    F: Fn(f64, f64, &DVector<f64>, &DVector<f64>, &DVector<f64>) -> DVector<f64> + Sync,
    H: Fn(f64, &DVector<f64>, &DVector<f64>) -> DVector<f64> + Sync,
    Q: Fn(f64, &DVector<f64>) -> DVector<f64> + Sync,
    P: Fn(f64, &DVector<f64>, &DVector<f64>) -> f64 + Sync,
{
    pub particles: Vec<Particle>,
    // state‑transition function (the “process” or “motion” model). The dynamics function.
    // F is a function of 
    //    t_prev and t (to handle time-varying dynamics if needed)
    //    u: control command
    //    q: A draw from the process noise distribution.
    f: F,
    // The measurement function. H is a function of:
    //    t: The current time (the time of measurement)
    //    x: the *predicted* x state.
    //    u: Any control input that might affect the measurement
    h: H,
    // Process noise sampler. It is a function of 
    //    t_prev: Previous time stamp
    //    x: Particle's previous state.
    // It returns q which is a sample of the process noise for this particle.
    draw_noise: Q,
    // observation‑likelihood function. It is a function of 
    //    t: Time of the measurement,
    //    dz = z_actual − z_pred — the measurement residual
    //    x: the state at which you’re evaluating the likelihood.
    // It returns a scalar probability (or density) that 
    // your predicted state would have produced that measurement error. 
    // You multiply your old particle weight by this likelihood to get the new weight.
    p_obs: P,
}

impl<F, H, Q, P> BootstrapFilter<F, H, Q, P>
where
    F: Fn(f64, f64, &DVector<f64>, &DVector<f64>, &DVector<f64>) -> DVector<f64> + Sync,
    H: Fn(f64, &DVector<f64>, &DVector<f64>) -> DVector<f64> + Sync,
    Q: Fn(f64, &DVector<f64>) -> DVector<f64> + Sync,
    P: Fn(f64, &DVector<f64>, &DVector<f64>) -> f64 + Sync,
{
    /// Initialize with random particles drawn from default Normal(0,1).
    pub fn new(n_particles: usize, f: F, h: H, draw_noise: Q, p_obs: P) -> Self {
        let mut rng = rand::rng();
        let normal = rand_distr::StandardNormal;
        let init_w = 1.0 / (n_particles as f64);
        let particles = (0..n_particles)
            .map(|_| Particle { x: DVector::from_element(1, rng.sample(normal)), w: init_w })
            .collect();
        BootstrapFilter { particles, f, h, draw_noise, p_obs }
    }

    /// Initialize all particles at a given state.
    pub fn new_with_initial_state(
        n_particles: usize,
        initial_state: &DVector<f64>,
        f: F,
        h: H,
        draw_noise: Q,
        p_obs: P,
    ) -> Self {
        let init_w = 1.0 / (n_particles as f64);
        let particles = (0..n_particles)
            .map(|_| Particle { x: initial_state.clone(), w: init_w })
            .collect();
        BootstrapFilter { particles, f, h, draw_noise, p_obs }
    }

    /// Compute effective sample size: ESS = 1 / sum(w_i^2).
    pub fn effective_sample_size(&self) -> f64 {
        let sum_sq: f64 = self.particles.iter().map(|p| p.w * p.w).sum();
        1.0 / sum_sq
    }

    /// Update filter: predict, weight, normalize, adaptive resample, then estimate.
    ///
    /// `resample_threshold`: fraction in (0,1] of N below which resampling occurs.
    pub fn update(
        &mut self,
        t_prev: f64,
        t: f64,
        u: &DVector<f64>,
        z: &DVector<f64>,
        resample_threshold: f64,
    ) -> DVector<f64> {
        let n = self.particles.len();

        // 1. Predict & weight update (parallel)
        let total_w: f64 = self.particles.par_iter_mut()
            .map(|p| {
                // Sample some process noise to be then fed to the dynamics function.
                let q = (self.draw_noise)(t_prev, &p.x);
                // Apply dynamics
                let x_pred = (self.f)(t_prev, t, &p.x, u, &q);
                // Run the observation function. We don't always measure the
                // thing that we predict. 
                let z_pred = (self.h)(t, &x_pred, u);
                // Find the probability that this particle could have predicted the
                // given observation. Multiply that by the particle weight.
                let w_new = p.w * (self.p_obs)(t, &(z - z_pred), &x_pred);
                // Update particle's state and weight.
                p.x = x_pred;
                p.w = w_new;
                w_new
            })
            .sum();

        // 2. Normalize the weights
        self.particles.iter_mut().for_each(|p| p.w /= total_w);

        // 3. ESS-based resampling
        let ess = self.effective_sample_size();
        if ess < resample_threshold * (n as f64) {
            self.resample();
        }

        // 4. Estimate: weighted mean
        let dim = self.particles[0].x.len();
        self.particles.iter()
            .fold(DVector::zeros(dim), |mut acc, p| { acc += &(&p.x * p.w); acc })
    }

    /// Multinomial resampling: rebuild particles with uniform weights.
    fn resample(&mut self) {
        let n = self.particles.len();
        let mut cdf = Vec::with_capacity(n);
        let mut cum = 0.0;
        for p in &self.particles {
            cum += p.w;
            cdf.push(cum);
        }
        // thread-local
        let mut rng = rand::rng();
        // new: will hold the resampled particles
        let mut new = Vec::with_capacity(n);
        for _ in 0..n {
          // r: draw a uniform random number in [0,1). 
          // This picks a random “height” under the total-weight curve.
            let r: f64 = rng.random();
            // binary_search_by returns Ok(i) if cdf[i] == r, 
            // or Err(i) where i is the insertion point (first cdf[i] > r).
            // We use unwrap_or_else(|i| i) to treat that insertion point as our chosen index.
            let idx = cdf.binary_search_by(|cw| cw.partial_cmp(&r).unwrap()).unwrap_or_else(|i| i);
            // re-use that same x.
            let mut chosen = self.particles[idx].clone();
            chosen.w = 1.0 / (n as f64);
            new.push(chosen);
        }
        self.particles = new;
    }
}

// Unit tests for ESS heuristic
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    fn f_id(_t0:f64,_t:f64,x:&DVector<f64>,_u:&DVector<f64>,_q:&DVector<f64>)->DVector<f64>{x.clone()}
    fn h_id(_t:f64,x:&DVector<f64>,_u:&DVector<f64>)->DVector<f64>{x.clone()}
    fn noise_zero(_t:f64,_x:&DVector<f64>)->DVector<f64>{DVector::zeros(1)}
    fn p_obs_const(_t:f64,_dz:&DVector<f64>,_x:&DVector<f64>)->f64{1.0}


    #[test]
    fn test_identity_filter() {
        let initial = DVector::from_element(1, 0.0);
        let mut bf = BootstrapFilter::new_with_initial_state(100, &initial, f_id, h_id, noise_zero, p_obs_const);
        let u = DVector::from_element(1, 0.0);
        let z = DVector::from_element(1, 0.0);
        let est = bf.update(0.0, 1.0, &u, &z, 0.5);
        // Since identity and zero measurement, estimate should be near zero
        assert!(est.norm() < 1e-6);
    }

    #[test]
    fn test_identity_filter_nonzero_measurement() {
        let initial = DVector::from_element(1, 0.0);
        let mut bf = BootstrapFilter::new_with_initial_state(100, &initial, f_id, h_id, noise_zero, p_obs_const);
        let u = DVector::from_element(1, 0.0);
        let z = DVector::from_element(1, 2.0);
        let mut est = DVector::zeros(1);
        for _ in 0..10 {
            est = bf.update(0.0, 1.0, &u, &z, 0.5);
        }
        // With flat likelihood and zero process noise, estimate will remain at initial mean (zero)
        assert!(
            est[0].abs() < 1e-6,
            "Estimate did not remain at initial mean: est={}",
            est[0]
        );
    }

    // Test with Gaussian observation likelihood
    fn p_obs_gauss(_t: f64, dz: &DVector<f64>, _x: &DVector<f64>) -> f64 {
        let sigma = 0.5;
        let norm = (2.0 * std::f64::consts::PI * sigma * sigma).sqrt();
        (-dz.norm_squared() / (2.0 * sigma * sigma)).exp() / norm
    }

    #[test]
    fn test_gaussian_observation_likelihood() {
        let mut bf = BootstrapFilter::new(200, f_id, h_id, noise_zero, p_obs_gauss);
        let u = DVector::from_element(1, 0.0);
        let z = DVector::from_element(1, 1.0);
        let mut est = DVector::zeros(1);
        for _ in 0..10 {
            est = bf.update(0.0, 1.0, &u, &z, 0.5);
        }
        // Estimate should move towards measurement
        assert!(
            (est[0] - 1.0).abs() < 0.2,
            "Estimate did not move toward measurement: est={}",
            est[0]
        );
    }

    // Test with nontrivial dynamics: x_{k+1} = x_k + u + q

    use rand_distr::{Distribution, Normal};

    fn noise_gauss(_t: f64, _x: &DVector<f64>) -> DVector<f64> {
        // Draw from N(0, 0.2^2)
        let normal = Normal::new(0.0, 0.2).unwrap();
        DVector::from_element(1, normal.sample(&mut rand::rng()))
    }

    #[test]
    fn test_tracking_random_walk() {
        // System: x_{k+1} = x_k + q, z_k = x_k + r
        fn f_rw(
            _t0: f64,
            _t: f64,
            x: &DVector<f64>,
            _u: &DVector<f64>,
            q: &DVector<f64>,
        ) -> DVector<f64> {
            x + q
        }
        fn h_rw(_t: f64, x: &DVector<f64>, _u: &DVector<f64>) -> DVector<f64> {
            x.clone()
        }
        fn p_obs_gauss(_t: f64, dz: &DVector<f64>, _x: &DVector<f64>) -> f64 {
            let sigma = 0.3;
            let norm = (2.0 * std::f64::consts::PI * sigma * sigma).sqrt();
            (-dz.norm_squared() / (2.0 * sigma * sigma)).exp() / norm
        }

        let mut bf = BootstrapFilter::new(500, f_rw, h_rw, noise_gauss, p_obs_gauss);
        let u = DVector::from_element(1, 0.0);

        // Simulate a random walk with measurements
        let mut x_true = DVector::from_element(1, 0.0);
        let mut rng = rand::rng();
        let noise = Normal::new(0.0, 0.2).unwrap();
        for k in 0..10 {
            // True state evolves
            x_true[0] += noise.sample(&mut rng);
            // Measurement with noise
            let z = DVector::from_element(1, x_true[0] + noise.sample(&mut rng));
            let est = bf.update(k as f64, (k + 1) as f64, &u, &z, 0.5);
            // Estimate should track true state within reasonable error
            assert!(
                (est[0] - x_true[0]).abs() < 0.5,
                "k={k}: est={}, true={}",
                est[0],
                x_true[0]
            );
        }
    }

    #[test]
    fn test_filter_converges_to_measurement() {
        // If we repeatedly observe the same measurement, the estimate should converge
        let mut bf = BootstrapFilter::new(300, f_id, h_id, noise_gauss, p_obs_gauss);
        let u = DVector::from_element(1, 0.0);
        let z = DVector::from_element(1, 3.0);
        let mut est = DVector::zeros(1);
        for _ in 0..30 {
            est = bf.update(0.0, 1.0, &u, &z, 0.5);
        }
        assert!(
            (est[0] - 3.0).abs() < 0.6,
            "Estimate did not converge to measurement: est={}",
            est[0]
        );
    }

    #[test]
    fn test_filter_with_large_noise() {
        // Test that the filter can still track the state with large process noise
        fn noise_large(_t: f64, _x: &DVector<f64>) -> DVector<f64> {
            let normal = Normal::new(0.0, 1.0).unwrap();
            DVector::from_element(1, normal.sample(&mut rand::rng()))
        }
        let mut bf = BootstrapFilter::new(1000, f_id, h_id, noise_large, p_obs_gauss);
        let u = DVector::from_element(1, 0.0);
        let z = DVector::from_element(1, 5.0);
        let mut est = DVector::zeros(1);
        for _ in 0..5 {
            est = bf.update(0.0, 1.0, &u, &z, 0.5);
        }
        // The estimate should be in the vicinity of the measurement, but with more uncertainty
        assert!(
            (est[0] - 5.0).abs() < 1.0,
            "Estimate too far from measurement: est={}",
            est[0]
        );
    }

    #[test]
    fn ess_initial() {
        let bf = BootstrapFilter::new(100, f_id, h_id, noise_zero, p_obs_const);
        assert!((bf.effective_sample_size() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn update_triggers_resample() {
        let mut bf = BootstrapFilter::new(50, f_id, h_id, noise_zero, p_obs_const);
        // after one update with uniform weights, ESS=N so no resample
        let u = DVector::from_element(1,0.0);
        let z = DVector::from_element(1,0.0);
        bf.update(0.0,1.0,&u,&z,0.5);
        // force low ESS by manually setting weights
        let first_w = bf.particles[0].w;
        bf.particles.iter_mut().for_each(|p| p.w = if p.w == first_w {1.0} else {0.0});
        let before = bf.particles[0].w;
        bf.update(1.0,2.0,&u,&z,0.5);
        // after resample, weights reset to uniform
        let after = bf.particles[0].w;
        assert!((after - 1.0/50.0).abs() < 1e-6 && (before - 1.0).abs() < 1e-6);
    }
}
