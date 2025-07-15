//! Visualization module offering both Plotly (interactive) and Plotters (static) APIs,
//! including trajectory plots and time‐series animations of particles colored by weight.

use nalgebra::DVector;

#[derive(serde::Serialize)]
pub struct ParticlePoint {
    pub t: usize,
    pub x: f64,
    pub w: f64,
    pub id: usize,
}



///============== Plotters Backend ==============

/// Static trajectory plot: each particle overlaid with color opacity ~ weight.
pub fn render_trajectories_plotters(
    filename: &str,
    particles_over_time: &[Vec<ParticlePoint>],
) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;
    let root = BitMapBackend::new(filename, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Particle Trajectories", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0..particles_over_time.len(),
            particles_over_time.iter().flat_map(|v| v.iter().map(|p| p.x)).fold(f64::INFINITY, f64::min)
            ..particles_over_time.iter().flat_map(|v| v.iter().map(|p| p.x)).fold(f64::NEG_INFINITY, f64::max)
        )?;
    chart.configure_mesh().draw()?;
    for id in 0..particles_over_time[0].len() {
        let series: Vec<(usize, f64)> = particles_over_time.iter().map(|v| (v[0].t, v[id].x)).collect();
        chart.draw_series(LineSeries::new(
            series,
            &RGBColor(0, 0, 255).mix((particles_over_time[0][id].w * 255.0)),
        ))?;
    }
    Ok(())
}