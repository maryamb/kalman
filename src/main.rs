//! Example main for running the particle filter and visualizing with the visualization module.

mod bootstrap_particles;
mod bootstrap_particles_visualization;

use bootstrap_particles_visualization::{ParticlePoint, render_trajectories_plotters};
use bootstrap_particles::{BootstrapFilter, Particle};
use nalgebra::DVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example: 10 time steps, 50 particles
    let n_particles = 50;
    let n_steps = 60;
    let mut filter = BootstrapFilter::new_with_initial_state(
        n_particles,
        &DVector::from_element(1, 0.0),
        |t0, t, x, u, q| x + q, // f: random walk
        |t, x, u| x.clone(),    // h: identity
        |_t, _x| DVector::from_element(1, 0.1 * rand::random::<f64>()), // noise
        |_t, dz, _x| (-dz.norm_squared() / 0.1).exp(), // p_obs: Gaussian
    );
    let u = DVector::from_element(1, 0.0);
    let mut particles_over_time: Vec<Vec<ParticlePoint>> = Vec::new();
    for t in 0..n_steps {
        let z = DVector::from_element(1, t as f64 * 0.5 + 0.1 * rand::random::<f64>()); // Simulated measurement
        filter.update(t as f64, (t+1) as f64, &u, &z, 0.5);
        // Record all particles for visualization
        let snapshot: Vec<ParticlePoint> = filter.particles.iter().enumerate().map(|(id, p)|
            ParticlePoint { t, x: p.x[0], w: p.w, id }
        ).collect();
        particles_over_time.push(snapshot);
    }
    render_trajectories_plotters("particles.png", &particles_over_time)?;
    println!("Wrote particles.png");
    Ok(())
}
