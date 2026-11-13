//! Runs the original CORTAO GLSL (its headers unchanged, compiled with glslc, passed to Vulkan as
//! SPIR-V) and this WGSL port on the same inputs, and compares what they write.
//!
//! Needs `glslc` and a Vulkan GPU, so it is ignored by default:
//! `cargo test --test against_original -- --ignored --nocapture`.

mod support;

use cortao_wgsl::{Configuration, FilterPassParameters, MainPassParameters, PrepassParameters, tables};
use support::gpu::{Gpu, Slot, level_view, view};
use support::scene::Scene;
use wgpu::TextureFormat as F;

const SIZE: (u32, u32) = (333, 207);
const MIPS: u32 = 8;

fn configuration(scene: &Scene) -> Configuration {
    Configuration {
        mip_count: MIPS,
        random_seed: 0x9E37_79B9,
        flip_y: true,
        is_perspective: scene.perspective,
        projection_matrix: scene.projection.to_cols_array_2d(),
        inverse_projection_matrix: scene.projection.inverse().to_cols_array_2d(),
        near_plane: scene.near,
        far_plane: scene.far,
        max_radius: 2.0,
        ..Configuration::new(scene.size.0, scene.size.1)
    }
}

/// One device for the whole run, used by one test at a time: creating a second device after
/// dropping the first, or several at once, has crashed the Vulkan driver.
fn gpu() -> (std::sync::MutexGuard<'static, ()>, &'static Gpu) {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    static GPU: std::sync::OnceLock<Gpu> = std::sync::OnceLock::new();
    let turn = TURN.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    (turn, GPU.get_or_init(|| Gpu::new().expect("a Vulkan adapter with SPIR-V passthrough and border clamping")))
}

fn groups(size: (u32, u32), per: u32) -> (u32, u32) {
    (size.0.div_ceil(per), size.1.div_ceil(per))
}

struct Inputs {
    depth: wgpu::Texture,
    normals: wgpu::Texture,
}

fn inputs(gpu: &Gpu, scene: &Scene) -> Inputs {
    Inputs {
        depth: gpu.texture_with(scene.size, F::R32Float, bytemuck::cast_slice(&scene.depth)),
        normals: gpu.texture_with(scene.size, F::Rgba32Float, bytemuck::cast_slice(&scene.normals)),
    }
}

fn prepass_original(gpu: &Gpu, scene: &Scene, inputs: &Inputs, c: &Configuration) -> wgpu::Texture {
    let projection = if scene.perspective { "CORTAO_PERSPECTIVE_PROJECTION" } else { "CORTAO_ORTHOGRAPHIC_PROJECTION" };
    let module = gpu.glsl("prepass", &[(projection, 1)]);
    let mut slots = vec![Slot::Uniform, Slot::FloatTexture, Slot::FloatTexture];
    slots.extend([Slot::Storage(F::R32Uint); 8]);
    let (pipeline, layout) = gpu.spirv_pipeline(&module, &slots);
    let packed = gpu.texture(scene.size, MIPS, F::R32Uint);
    let params = gpu.uniform(&PrepassParameters::new(c));
    let mips: Vec<_> = (0..MIPS).map(|l| level_view(&packed, l)).collect();
    let (depth, normals) = (view(&inputs.depth), view(&inputs.normals));
    let mut resources = vec![
        (0, params.as_entire_binding()),
        (1, wgpu::BindingResource::TextureView(&depth)),
        (2, wgpu::BindingResource::TextureView(&normals)),
    ];
    resources.extend(mips.iter().enumerate().map(|(i, v)| (3 + i as u32, wgpu::BindingResource::TextureView(v))));
    gpu.dispatch(&pipeline, &layout, &resources, groups(scene.size, 8));
    packed
}

fn prepass_port(gpu: &Gpu, scene: &Scene, inputs: &Inputs, c: &Configuration) -> wgpu::Texture {
    let constants = [("CORTAO_PERSPECTIVE_PROJECTION", f64::from(u8::from(scene.perspective)))];
    let packed = gpu.texture(scene.size, MIPS, F::R32Uint);
    let full = PrepassParameters::new(c);
    let mips: Vec<_> = (0..MIPS).map(|l| level_view(&packed, l)).collect();

    let source = cortao_wgsl::compose(&[cortao_wgsl::BASE, cortao_wgsl::PREPASS, cortao_wgsl::reference::PREPASS]);
    let (pipeline, layout) = gpu.wgsl_pipeline(&source, &constants);
    let params = gpu.uniform(&full.first_dispatch(4));
    let (depth, normals) = (view(&inputs.depth), view(&inputs.normals));
    let mut resources = vec![
        (0, params.as_entire_binding()),
        (1, wgpu::BindingResource::TextureView(&depth)),
        (2, wgpu::BindingResource::TextureView(&normals)),
    ];
    resources.extend(mips[..4].iter().enumerate().map(|(i, v)| (3 + i as u32, wgpu::BindingResource::TextureView(v))));
    gpu.dispatch(&pipeline, &layout, &resources, groups(scene.size, 8));

    let source = cortao_wgsl::compose(&[cortao_wgsl::BASE, cortao_wgsl::PREPASS, cortao_wgsl::reference::PREPASS_CONTINUE]);
    let (pipeline, layout) = gpu.wgsl_pipeline(&source, &constants);
    let params = gpu.uniform(&full);
    let mut resources = vec![(0, params.as_entire_binding()), (1, wgpu::BindingResource::TextureView(&mips[3]))];
    resources.extend(mips[4..].iter().enumerate().map(|(i, v)| (3 + i as u32, wgpu::BindingResource::TextureView(v))));
    let level3 = ((scene.size.0 >> 3).max(1), (scene.size.1 >> 3).max(1));
    gpu.dispatch(&pipeline, &layout, &resources, groups(level3, 8));
    packed
}

#[derive(Clone, Copy, Debug)]
struct MainVariant {
    samples: u32,
    rays: u32,
    occluder_table: bool,
    bent_normal_table: bool,
}

struct MainOutput {
    /// (occlusion, bent normal) per pixel.
    result: Vec<[f32; 4]>,
    ray_bits: Vec<u32>,
}

/// The main pass's test output for the port: every value at full precision, as the GLSL
/// wrapper writes them.
const PORT_MAIN_OUTPUT: &str = r#"
@group(0) @binding(5) var outputBuffer: texture_storage_2d<rgba32float, write>;
@group(0) @binding(6) var rayBitsBuffer: texture_storage_2d<r32uint, write>;
fn cortao_writeResult(pixelCoords: vec2<i32>, occlusion: f32, bentNormal: vec3<f32>, tangentSpaceTransform: mat3x3<f32>, rayBits: u32) {
    textureStore(outputBuffer, pixelCoords, vec4<f32>(occlusion, bentNormal));
    textureStore(rayBitsBuffer, pixelCoords, vec4<u32>(rayBits, 0u, 0u, 0u));
}
"#;

fn main_pass(gpu: &Gpu, scene: &Scene, inputs: &Inputs, packed: &wgpu::Texture, c: &Configuration, v: MainVariant, original: bool) -> MainOutput {
    let table_format = match tables::occluder_bytes_per_texel(v.rays) {
        1 => F::R8Uint,
        2 => F::R16Uint,
        _ => F::R32Uint,
    };
    let (d, r) = (tables::OCCLUDER_DIRECTION_RESOLUTION, tables::OCCLUDER_RADIUS_RESOLUTION);
    let table = gpu.texture3_with((d, d, r), table_format, &tables::occluder_lookup_table_bytes(d, r, v.rays));
    let bent = if v.rays == 8 || v.rays == 16 { tables::bent_normal_lookup_table_words(v.rays) } else { vec![0] };
    let bent = gpu.storage_buffer(&bent);
    let output = gpu.texture(scene.size, 1, F::Rgba32Float);
    let ray_bits = gpu.texture(scene.size, 1, F::R32Uint);
    let params = gpu.uniform(&MainPassParameters::new(c));
    let (packed_view, normals, table_view, output_view, bits_view) =
        (view(packed), view(&inputs.normals), view(&table), view(&output), view(&ray_bits));
    let border = gpu.sampler(wgpu::AddressMode::ClampToBorder);
    let clamp = gpu.sampler(wgpu::AddressMode::ClampToEdge);
    let mut resources = vec![
        (0, params.as_entire_binding()),
        (1, wgpu::BindingResource::TextureView(&packed_view)),
        (2, wgpu::BindingResource::TextureView(&normals)),
        (3, wgpu::BindingResource::TextureView(&table_view)),
        (4, bent.as_entire_binding()),
        (5, wgpu::BindingResource::TextureView(&output_view)),
        (6, wgpu::BindingResource::TextureView(&bits_view)),
    ];
    let (pipeline, layout) = if original {
        let projection = if scene.perspective { "CORTAO_PERSPECTIVE_PROJECTION" } else { "CORTAO_ORTHOGRAPHIC_PROJECTION" };
        let module = gpu.glsl(
            "main",
            &[
                (projection, 1),
                ("CORTAO_SAMPLE_COUNT", v.samples),
                ("CORTAO_RAY_COUNT", v.rays),
                ("CORTAO_OCCLUDER_LOOKUP_TABLE", u32::from(v.occluder_table)),
                ("CORTAO_BENT_NORMAL_LOOKUP_TABLE", u32::from(v.bent_normal_table)),
            ],
        );
        resources.push((7, wgpu::BindingResource::Sampler(&border)));
        resources.push((8, wgpu::BindingResource::Sampler(&clamp)));
        gpu.spirv_pipeline(
            &module,
            &[
                Slot::Uniform,
                Slot::UintTexture,
                Slot::FloatTexture,
                Slot::UintTexture3,
                Slot::ReadOnlyBuffer,
                Slot::Storage(F::Rgba32Float),
                Slot::Storage(F::R32Uint),
                Slot::Sampler,
                Slot::Sampler,
            ],
        )
    } else {
        let source = cortao_wgsl::compose(&[
            cortao_wgsl::BASE,
            cortao_wgsl::MAIN_PASS,
            cortao_wgsl::reference::MAIN_PASS_COMMON,
            PORT_MAIN_OUTPUT,
        ]);
        gpu.wgsl_pipeline(
            &source,
            &[
                ("CORTAO_PERSPECTIVE_PROJECTION", f64::from(u8::from(scene.perspective))),
                ("CORTAO_SAMPLE_COUNT", f64::from(v.samples)),
                ("CORTAO_RAY_COUNT", f64::from(v.rays)),
                ("CORTAO_OCCLUDER_LOOKUP_TABLE", f64::from(u8::from(v.occluder_table))),
                ("CORTAO_BENT_NORMAL_LOOKUP_TABLE", f64::from(u8::from(v.bent_normal_table))),
            ],
        )
    };
    gpu.dispatch(&pipeline, &layout, &resources, groups(scene.size, 16));
    let result = gpu.read(&output, 0);
    MainOutput { result: bytemuck::cast_slice(&result).to_vec(), ray_bits: gpu.read(&ray_bits, 0) }
}

#[derive(Clone, Copy, Debug)]
struct FilterVariant {
    bent_normal: bool,
    large: bool,
    symmetry: bool,
}

/// The filter pass's test input and output for the port, matching the GLSL wrapper's: the main
/// pass's output as r32uint (packed snorm, or the occlusion's bits), the result as rgba32float.
const PORT_FILTER_IO: &str = r#"
@group(0) @binding(2) var ssaoBuffer: texture_2d<u32>;
@group(0) @binding(3) var outputBuffer: texture_storage_2d<rgba32float, write>;
fn loadSSAO(texel: vec2<i32>, size: vec2<u32>) -> vec4<f32> {
    if (!cortao_inBounds(texel, size)) {
        return vec4<f32>(0.0);
    }
    let value = textureLoad(ssaoBuffer, texel, 0).r;
    if (CORTAO_FILTER_BENT_NORMAL) {
        return unpack4x8snorm(value);
    }
    return vec4<f32>(bitcast<f32>(value), 0.0, 0.0, 0.0);
}
fn cortao_gatherSSAO(texCoords: vec2<f32>, offset: vec2<i32>) -> mat4x4<f32> {
    let size = textureDimensions(ssaoBuffer, 0);
    let t = cortao_gatherTexels(texCoords, size, offset);
    return mat4x4<f32>(loadSSAO(t.x, size), loadSSAO(t.y, size), loadSSAO(t.z, size), loadSSAO(t.w, size));
}
fn cortao_writeFilterOutput(pixelCoords: vec2<i32>, ssao: vec4<f32>) {
    if (CORTAO_FILTER_BENT_NORMAL) {
        textureStore(outputBuffer, pixelCoords, ssao);
    } else {
        textureStore(outputBuffer, pixelCoords, vec4<f32>(ssao.x, 0.0, 0.0, 0.0));
    }
}
"#;

fn pack_snorm(v: [f32; 4]) -> u32 {
    v.iter().enumerate().fold(0, |word, (i, c)| word | (u32::from((c.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8) << (8 * i)))
}

fn filter_pass(gpu: &Gpu, scene: &Scene, packed: &wgpu::Texture, main: &MainOutput, c: &Configuration, v: FilterVariant, original: bool) -> Vec<[f32; 4]> {
    let ssao: Vec<u32> = main.result.iter().map(|r| if v.bent_normal { pack_snorm(*r) } else { r[0].to_bits() }).collect();
    let ssao = gpu.texture_with(scene.size, F::R32Uint, bytemuck::cast_slice(&ssao));
    let output = gpu.texture(scene.size, 1, F::Rgba32Float);
    let params = gpu.uniform(&FilterPassParameters::new(c));
    let level0 = level_view(packed, 0);
    let (ssao_view, output_view) = (view(&ssao), view(&output));
    let border = gpu.sampler(wgpu::AddressMode::ClampToBorder);
    let mut resources = vec![
        (0, params.as_entire_binding()),
        (1, wgpu::BindingResource::TextureView(&level0)),
        (2, wgpu::BindingResource::TextureView(&ssao_view)),
        (3, wgpu::BindingResource::TextureView(&output_view)),
    ];
    let (pipeline, layout) = if original {
        let module = gpu.glsl(
            "filter",
            &[
                ("CORTAO_FILTER_BENT_NORMAL", u32::from(v.bent_normal)),
                ("CORTAO_LARGE_FILTER", u32::from(v.large)),
                ("CORTAO_FILTER_SYMMETRY", u32::from(v.symmetry)),
            ],
        );
        resources.push((4, wgpu::BindingResource::Sampler(&border)));
        gpu.spirv_pipeline(&module, &[Slot::Uniform, Slot::UintTexture, Slot::UintTexture, Slot::Storage(F::Rgba32Float), Slot::Sampler])
    } else {
        let source = cortao_wgsl::compose(&[
            cortao_wgsl::BASE,
            cortao_wgsl::FILTER_PASS,
            cortao_wgsl::reference::FILTER_PASS_COMMON,
            PORT_FILTER_IO,
        ]);
        gpu.wgsl_pipeline(
            &source,
            &[
                ("CORTAO_FILTER_BENT_NORMAL", f64::from(u8::from(v.bent_normal))),
                ("CORTAO_LARGE_FILTER", f64::from(u8::from(v.large))),
                ("CORTAO_FILTER_SYMMETRY", f64::from(u8::from(v.symmetry))),
            ],
        )
    };
    gpu.dispatch(&pipeline, &layout, &resources, groups((scene.size.0 + 1, scene.size.1 + 1), 16));
    bytemuck::cast_slice(&gpu.read(&output, 0)).to_vec()
}

/// Largest and mean absolute difference over the first `components` components of `a` and `b`.
fn difference(a: &[[f32; 4]], b: &[[f32; 4]], components: usize) -> (f32, f32) {
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (x, y) in a.iter().zip(b) {
        for k in 0..components {
            // A pixel whose every ray is occluded has no bent normal: NaN in both.
            if x[k].is_nan() && y[k].is_nan() {
                count += 1;
                continue;
            }
            let d = (x[k] - y[k]).abs();
            max = max.max(d);
            sum += f64::from(d);
            count += 1;
        }
    }
    (max, (sum / count as f64) as f32)
}

fn scenes() -> [Scene; 2] {
    [Scene::perspective(SIZE), Scene::orthographic(SIZE)]
}

#[test]
#[ignore = "needs glslc and a Vulkan GPU"]
fn prepass_writes_the_originals_packed_buffer() {
    let (_turn, gpu) = gpu();
    for scene in scenes() {
        let c = configuration(&scene);
        let inputs = inputs(gpu, &scene);
        let original = prepass_original(gpu, &scene, &inputs, &c);
        let port = prepass_port(gpu, &scene, &inputs, &c);
        let sky = scene.depth.iter().filter(|&&d| d == if scene.perspective { 0.0 } else { 1.0 }).count();
        assert!(sky > 0 && sky < scene.depth.len() / 2, "the scene has some sky and mostly surfaces");
        for level in 0..MIPS {
            let (a, b) = (gpu.read(&original, level), gpu.read(&port, level));
            let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            println!("perspective {} level {level}: {differing} of {} texels differ", scene.perspective, a.len());
            assert_eq!(differing, 0, "level {level} matches bit for bit");
        }
    }
}

#[test]
#[ignore = "needs glslc and a Vulkan GPU"]
fn main_pass_traces_the_originals_rays() {
    let (_turn, gpu) = gpu();
    let variants = [
        MainVariant { samples: 16, rays: 16, occluder_table: true, bent_normal_table: true },
        MainVariant { samples: 8, rays: 8, occluder_table: true, bent_normal_table: true },
        MainVariant { samples: 12, rays: 32, occluder_table: true, bent_normal_table: false },
        MainVariant { samples: 6, rays: 16, occluder_table: false, bent_normal_table: false },
        MainVariant { samples: 32, rays: 16, occluder_table: true, bent_normal_table: false },
    ];
    for scene in scenes() {
        let c = configuration(&scene);
        let inputs = inputs(gpu, &scene);
        // Both main passes read the original prepass's buffer, so only the main pass differs.
        let packed = prepass_original(gpu, &scene, &inputs, &c);
        for v in variants {
            let a = main_pass(gpu, &scene, &inputs, &packed, &c, v, true);
            let b = main_pass(gpu, &scene, &inputs, &packed, &c, v, false);
            let written = a.result.iter().filter(|r| r[0] != 0.0 || r[1] != 0.0).count();
            let same_rays = a.ray_bits.iter().zip(&b.ray_bits).filter(|(x, y)| x == y).count();
            let share = same_rays as f64 / a.ray_bits.len() as f64;
            let (occlusion_max, occlusion_mean) = difference(&a.result, &b.result, 1);
            let (bent_max, bent_mean) = difference(
                &a.result.iter().map(|r| [r[1], r[2], r[3], 0.0]).collect::<Vec<_>>(),
                &b.result.iter().map(|r| [r[1], r[2], r[3], 0.0]).collect::<Vec<_>>(),
                3,
            );
            let mean_occlusion = a.result.iter().map(|r| f64::from(r[0])).sum::<f64>() / written as f64;
            println!(
                "perspective {} {v:?}: {written} pixels, mean occlusion {mean_occlusion:.3}; ray masks equal {:.4}%; occlusion diff max {occlusion_max:.4} mean {occlusion_mean:.6}; bent normal diff max {bent_max:.4} mean {bent_mean:.6}",
                scene.perspective,
                share * 100.0
            );
            assert!(written > a.result.len() / 2, "the main pass ran");
            assert!(share >= 0.995, "ray masks agree on at least 99.5% of pixels");
            assert!(occlusion_mean <= 1e-3, "occlusion agrees on average");
            assert!(bent_mean <= 1e-3, "bent normals agree on average");
        }
    }
}

#[test]
#[ignore = "needs glslc and a Vulkan GPU"]
fn filter_pass_blurs_like_the_original() {
    let (_turn, gpu) = gpu();
    let variants = [
        FilterVariant { bent_normal: false, large: false, symmetry: false },
        FilterVariant { bent_normal: false, large: true, symmetry: true },
        FilterVariant { bent_normal: true, large: true, symmetry: true },
        FilterVariant { bent_normal: true, large: false, symmetry: false },
    ];
    let main_variant = MainVariant { samples: 16, rays: 16, occluder_table: true, bent_normal_table: true };
    for scene in scenes() {
        let c = configuration(&scene);
        let inputs = inputs(gpu, &scene);
        let packed = prepass_original(gpu, &scene, &inputs, &c);
        let main = main_pass(gpu, &scene, &inputs, &packed, &c, main_variant, true);
        for v in variants {
            let a = filter_pass(gpu, &scene, &packed, &main, &c, v, true);
            let b = filter_pass(gpu, &scene, &packed, &main, &c, v, false);
            let components = if v.bent_normal { 4 } else { 1 };
            let (max, mean) = difference(&a, &b, components);
            let written = a.iter().filter(|r| r[0] != 0.0).count();
            println!("perspective {} {v:?}: {written} pixels; diff max {max:.6} mean {mean:.8}", scene.perspective);
            assert!(written > a.len() / 2, "the filter pass ran");
            assert!(max <= 1e-3, "the filter agrees everywhere");
        }
    }
}
