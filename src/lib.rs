//! CORTAO (Circular Occluder Ray-Traced Ambient Occlusion) for WGSL and WebGPU.
//!
//! A port of the [CORTAO reference implementation](https://github.com/theagentd/CORTAO) by
//! Daniel Isheden. The shaders are WGSL text to compose with your own input and output
//! callbacks, exactly as the original's headers are included into your own HLSL or GLSL:
//!
//! ```
//! let main_pass = cortao_wgsl::compose(&[
//!     cortao_wgsl::BASE,
//!     cortao_wgsl::MAIN_PASS,
//!     cortao_wgsl::reference::MAIN_PASS_COMMON,
//!     cortao_wgsl::reference::MAIN_PASS_OCCLUSION,
//! ]);
//! assert!(main_pass.contains("fn cortao_mainPass"));
//! ```
//!
//! The original's compile-time defines are WGSL `override` constants; set them per pipeline
//! (`PipelineCompilationOptions::constants` in wgpu). [`Configuration`] derives the three
//! passes' uniform blocks, and [`tables`] generates the two lookup tables.

mod params;
pub mod tables;

pub use params::{Configuration, FilterPassParameters, MainPassParameters, PrepassParameters};

/// Shared definitions every pass needs.
pub const BASE: &str = include_str!("../shaders/cortao_base.wgsl");
/// The prepass: linear depth and a packed normal, into every mip of the packed data buffer.
pub const PREPASS: &str = include_str!("../shaders/cortao_prepass.wgsl");
/// The main pass: occlusion and bent normals.
pub const MAIN_PASS: &str = include_str!("../shaders/cortao_main_pass.wgsl");
/// The optional filter pass, ahead of a temporal filter.
pub const FILTER_PASS: &str = include_str!("../shaders/cortao_filter_pass.wgsl");

/// Ready-made bindings, callbacks and entry points (`main`) for each pass, the WGSL
/// counterparts of the original's reference shaders.
pub mod reference {
    /// Levels 0 to 3 of the packed data buffer. With [`super::BASE`] and [`super::PREPASS`].
    pub const PREPASS: &str = include_str!("../shaders/reference/prepass.wgsl");
    /// Levels 4 to 7, from level 3. With [`super::BASE`] and [`super::PREPASS`].
    pub const PREPASS_CONTINUE: &str = include_str!("../shaders/reference/prepass_continue.wgsl");
    /// The main pass's inputs and entry point; add one of the `MAIN_PASS_*` outputs.
    pub const MAIN_PASS_COMMON: &str = include_str!("../shaders/reference/main_pass_common.wgsl");
    /// Occlusion alone, `r32float`.
    pub const MAIN_PASS_OCCLUSION: &str = include_str!("../shaders/reference/main_pass_occlusion.wgsl");
    /// Occlusion and view-space bent normal, `rgba8snorm`.
    pub const MAIN_PASS_BENT_NORMAL: &str = include_str!("../shaders/reference/main_pass_bent_normal.wgsl");
    /// Occlusion and bent normal packed into `r32uint`, for the filter pass.
    pub const MAIN_PASS_PACKED_BENT_NORMAL: &str = include_str!("../shaders/reference/main_pass_packed_bent_normal.wgsl");
    /// The filter pass's inputs and entry point; add one of the `FILTER_PASS_*` files.
    pub const FILTER_PASS_COMMON: &str = include_str!("../shaders/reference/filter_pass_common.wgsl");
    /// Filters [`MAIN_PASS_OCCLUSION`]'s output into `r32float`.
    pub const FILTER_PASS_OCCLUSION: &str = include_str!("../shaders/reference/filter_pass_occlusion.wgsl");
    /// Filters [`MAIN_PASS_PACKED_BENT_NORMAL`]'s output into `rgba8snorm`.
    pub const FILTER_PASS_BENT_NORMAL: &str = include_str!("../shaders/reference/filter_pass_bent_normal.wgsl");
}

/// Joins WGSL sources into one module. WGSL declarations may appear in any order, so the
/// order of `parts` does not matter.
pub fn compose(parts: &[&str]) -> String {
    parts.join("\n")
}
