//! The per-pass parameter blocks, derived from one [`Configuration`] (the original's
//! `CORTAOParameters.h`).

use bytemuck::{Pod, Zeroable};

/// The tweakable settings every pass's parameters are derived from.
///
/// Matrices are column-major, indexed `[column][row]` (glam's `to_cols_array_2d`), and view
/// space is right-handed with the camera looking down -Z (OpenGL / glam `*_rh` conventions).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Configuration {
    /// The frame's size in pixels, 1 to 65535 each.
    pub width: u32,
    pub height: u32,
    /// The number of mips of the packed data buffer, 1 to 8.
    pub mip_count: u32,
    /// Re-randomise every frame when a temporal filter follows; keep it constant otherwise.
    pub random_seed: u32,

    /// True when texture coordinates run down the screen while NDC Y runs up (WebGPU, Vulkan,
    /// Direct3D, Metal all do with a standard projection matrix).
    pub flip_y: bool,
    /// A perspective projection (true) or an orthographic one (false). Must match the
    /// `CORTAO_PERSPECTIVE_PROJECTION` override of every pass.
    pub is_perspective: bool,
    /// The projection the depth buffer was drawn with: no rotation or translation, an
    /// off-centre projection allowed.
    pub projection_matrix: [[f32; 4]; 4],
    /// Its exact inverse.
    pub inverse_projection_matrix: [[f32; 4]; 4],
    /// The near and far planes the projection was built from. With a reversed depth buffer,
    /// `near_plane` is the larger. For perspective, one may be `f32::INFINITY`.
    pub near_plane: f32,
    pub far_plane: f32,

    /// The largest occlusion radius, in view-space units.
    pub max_radius: f32,
    /// The largest radius on screen, as a share of the frame's height (0.1 to 0.25 suggested).
    pub max_screenspace_radius_height_ratio: f32,
    /// Mip bias for sampling the packed data buffer: lower is steadier and slower.
    pub mip_bias: f32,
    /// How far occluders grow to make up for surfaces seen edge-on (1 is no compensation).
    pub max_slope_compensation: f32,
    /// Front/back-facing occluder threshold, 0 to 1.
    pub backface_threshold: f32,
    /// Extra slope compensation for back-facing occluders, 0 to 1.
    pub backface_boost: f32,

    /// The filter pass's depth sensitivity (0 turns the depth check off).
    pub filter_depth_sensitivity: f32,
    /// The filter pass's normal sensitivity (0 turns the normal check off).
    pub filter_normal_sensitivity: f32,
}

impl Configuration {
    /// The original's defaults for the tweakable values, for a frame of `width` x `height`.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            mip_count: 8,
            random_seed: 0,
            flip_y: true,
            is_perspective: true,
            projection_matrix: IDENTITY,
            inverse_projection_matrix: IDENTITY,
            near_plane: 0.1,
            far_plane: 1000.0,
            max_radius: 1.0,
            max_screenspace_radius_height_ratio: 0.2,
            mip_bias: -4.0,
            max_slope_compensation: 4.0,
            backface_threshold: 0.5,
            backface_boost: 0.5,
            filter_depth_sensitivity: 512.0,
            filter_normal_sensitivity: 16.0,
        }
    }
}

const IDENTITY: [[f32; 4]; 4] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

fn packed_resolution(c: &Configuration) -> u32 {
    c.width | (c.height << 16)
}

/// The prepass's uniform block (`CORTAOPrepassParameters`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct PrepassParameters {
    pub packed_resolution: u32,
    pub depth_scale: f32,
    pub depth_offset: f32,
    pub random_seed: u32,
    pub mip_count: u32,
    pub _padding: [u32; 3],
}

impl PrepassParameters {
    pub fn new(c: &Configuration) -> Self {
        let (near, far) = (c.near_plane, c.far_plane);
        let (depth_scale, depth_offset) = if c.is_perspective {
            if near.is_infinite() {
                // infinite near plane (reverse depth)
                (1.0 / far, 0.0)
            } else if far.is_infinite() {
                // infinite far plane
                (-1.0 / near, 1.0 / near)
            } else {
                // finite near and far planes (reverse depth is supported)
                ((near - far) / (near * far), 1.0 / near)
            }
        } else {
            (far - near, near)
        };
        Self {
            packed_resolution: packed_resolution(c),
            depth_scale,
            depth_offset,
            random_seed: c.random_seed,
            mip_count: c.mip_count,
            _padding: [0; 3],
        }
    }

    /// These parameters for a dispatch that writes at most `levels` mips: the first prepass
    /// dispatch when a pass can bind only `levels` storage textures.
    pub fn first_dispatch(self, levels: u32) -> Self {
        Self { mip_count: self.mip_count.min(levels), ..self }
    }
}

/// The main pass's uniform block (`CORTAOMainPassParameters`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct MainPassParameters {
    pub packed_resolution: u32,
    pub random_seed: u32,
    pub normal_bias: f32,
    pub radius: f32,
    pub unproject_params: [f32; 4],
    pub projection_params: [f32; 4],
    pub max_apparent_circle_radius_coefficient: f32,
    pub mip_bias: f32,
    pub inverse_max_slope_compensation: f32,
    pub backface_threshold: f32,
    pub backface_boost: f32,
    pub _padding: [u32; 3],
}

impl MainPassParameters {
    pub fn new(c: &Configuration) -> Self {
        let p = &c.projection_matrix;
        let i = &c.inverse_projection_matrix;
        let (proj_x, proj_y, proj_offset_x, proj_offset_y) = (p[0][0], p[1][1], p[2][0], p[2][1]);
        let (inv_x, inv_y, inv_offset_x, inv_offset_y) = (i[0][0], i[1][1], i[3][0], i[3][1]);

        let (unproject_params, projection_params) = if c.flip_y {
            (
                // The original has `invY - invOffsetY` here, which unprojects an off-centre lens (a jitter, a
                // shifted window) on the wrong side; `+` makes it the inverse of the projection below.
                [2.0 * inv_x, -2.0 * inv_y, -inv_x + inv_offset_x, inv_y + inv_offset_y],
                [0.5 * proj_x, -0.5 * proj_y, -0.5 + 0.5 * proj_offset_x, -0.5 - 0.5 * proj_offset_y],
            )
        } else {
            (
                [2.0 * inv_x, 2.0 * inv_y, -inv_x + inv_offset_x, -inv_y + inv_offset_y],
                [0.5 * proj_x, 0.5 * proj_y, -0.5 + 0.5 * proj_offset_x, -0.5 + 0.5 * proj_offset_y],
            )
        };

        let max_apparent_circle_radius = inv_y * c.max_screenspace_radius_height_ratio * 2.0;
        Self {
            packed_resolution: packed_resolution(c),
            random_seed: c.random_seed,
            normal_bias: inv_x / c.width as f32 * 2.0f32.sqrt(),
            radius: c.max_radius,
            unproject_params,
            projection_params,
            max_apparent_circle_radius_coefficient: max_apparent_circle_radius
                / (max_apparent_circle_radius * max_apparent_circle_radius + 1.0).sqrt(),
            mip_bias: (proj_y * c.height as f32 * 0.5).log2() + c.mip_bias,
            inverse_max_slope_compensation: 1.0 / c.max_slope_compensation,
            backface_threshold: c.backface_threshold,
            backface_boost: c.backface_boost,
            _padding: [0; 3],
        }
    }
}

/// The filter pass's uniform block (`CORTAOFilterPassParameters`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct FilterPassParameters {
    pub packed_resolution: u32,
    pub depth_sensitivity: f32,
    pub normal_sensitivity: f32,
    pub _padding: u32,
}

impl FilterPassParameters {
    pub fn new(c: &Configuration) -> Self {
        Self {
            packed_resolution: packed_resolution(c),
            depth_sensitivity: c.filter_depth_sensitivity,
            normal_sensitivity: c.filter_normal_sensitivity,
            _padding: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_match_the_wgsl_struct_sizes() {
        assert_eq!(size_of::<PrepassParameters>(), 32);
        assert_eq!(size_of::<MainPassParameters>(), 80);
        assert_eq!(size_of::<FilterPassParameters>(), 16);
    }

    #[test]
    fn reversed_infinite_perspective_linearises_to_near_over_depth() {
        let c = Configuration { near_plane: f32::INFINITY, far_plane: 0.1, ..Configuration::new(4, 4) };
        let p = PrepassParameters::new(&c);
        // linear = 1 / (depth * scale + offset)
        let linear = 1.0 / (0.05 * p.depth_scale + p.depth_offset);
        assert!((linear - 2.0).abs() < 1e-5);
        // The sky (depth 0) is told apart by a positive scale.
        assert!(p.depth_scale > 0.0);
    }

    /// A view-space point carried to texture coordinates the way the main pass projects samples,
    /// and back the way it unprojects them, lands on the ray it started on.
    fn round_trip(flip_y: bool) -> [f32; 2] {
        let (a, b, c, d) = (1.3f32, 0.05f32, 1.7f32, -0.04f32);
        // Columns: an off-centre, reversed, infinite perspective lens.
        let projection = [[a, 0.0, 0.0, 0.0], [0.0, c, 0.0, 0.0], [b, d, 0.0, -1.0], [0.0, 0.0, 0.1, 0.0]];
        let inverse = [[1.0 / a, 0.0, 0.0, 0.0], [0.0, 1.0 / c, 0.0, 0.0], [0.0, 0.0, 0.0, 10.0], [b / a, d / c, -1.0, 0.0]];
        let c0 = Configuration { flip_y, projection_matrix: projection, inverse_projection_matrix: inverse, ..Configuration::new(64, 64) };
        let p = MainPassParameters::new(&c0);
        let (x, y, z) = (0.3f32, -0.2f32, -2.0f32);
        // Main pass: projectionMatrix * (x, y, z) with columns (px, 0, 0), (0, py, 0), (pz, pw, -1).
        let (u, v, w) = (p.projection_params[0] * x + p.projection_params[2] * z, p.projection_params[1] * y + p.projection_params[3] * z, -z);
        let (u, v) = (u / w, v / w);
        // Main pass: direction = uv * unproject.xy + unproject.zw, at depth 1 along -z.
        let back = [u * p.unproject_params[0] + p.unproject_params[2], v * p.unproject_params[1] + p.unproject_params[3]];
        [back[0] - x / -z, back[1] - y / -z]
    }

    #[test]
    fn an_off_centre_lens_round_trips() {
        for flip_y in [false, true] {
            let error = round_trip(flip_y);
            assert!(error[0].abs() < 1e-5 && error[1].abs() < 1e-5, "flip_y {flip_y}: off by {error:?}");
        }
    }

    #[test]
    fn standard_perspective_linearises_both_planes() {
        let c = Configuration { near_plane: 0.5, far_plane: 100.0, ..Configuration::new(4, 4) };
        let p = PrepassParameters::new(&c);
        let at = |depth: f32| 1.0 / (depth * p.depth_scale + p.depth_offset);
        assert!((at(0.0) - 0.5).abs() < 1e-5);
        assert!((at(1.0) - 100.0).abs() < 1e-2);
    }
}
