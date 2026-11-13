# cortao-wgsl

A WGSL / WebGPU port of **CORTAO** (Circular Occluder Ray-Traced Ambient Occlusion), the
screen-space ambient occlusion technique by Daniel Isheden, released by Fenris Creations under
the MIT license. Read the [original repository](https://github.com/theagentd/CORTAO) for how
the technique works, its parameters, and its benchmarks. This port follows reference
implementation v1.0.1 (commit `9e8771f`).

The port matches the original. `tests/against_original.rs` compiles the original's GLSL
headers, unchanged, and runs them on the same GPU as the port, with the same inputs:

| Pass | Result |
|---|---|
| Prepass | Every mip of the packed buffer is bit-identical, for perspective and orthographic projections. |
| Main pass | Ray masks are identical on 99.99% of pixels. The rest are pixels where the original's hardware sampler rounds a coordinate on a texel edge differently from an exact `floor`. When the original reads texels the same way, 100% are identical. Covered: 8/16/32 rays, 6–32 samples, with and without either lookup table. |
| Filter pass | Bit-identical: occlusion and bent normal, 5- and 13-tap, with and without symmetry. |
| Lookup tables | Generated in code, and bit-identical to the original's shipped `.bin` and `.h` tables. |

Every reference module also compiles in Chrome's WebGPU (Tint) and in naga.

## Using it

Like the original, each pass is a library you compose with your own **callbacks**: the
functions that read your depth, normals and lookup tables, and write the results. WGSL
declarations may appear in any order, so composing is plain concatenation:

```rust
let source = cortao_wgsl::compose(&[
    cortao_wgsl::BASE,
    cortao_wgsl::MAIN_PASS,
    cortao_wgsl::reference::MAIN_PASS_COMMON,     // bindings, input callbacks, entry point
    cortao_wgsl::reference::MAIN_PASS_OCCLUSION,  // the output callback
]);
```

The files in `shaders/reference/` are ready-made callbacks and entry points (`main`), the WGSL
counterparts of the original's reference shaders. They expect a view-space normal buffer
(xyz, normalised, +Z toward the camera) and a depth buffer bound as an unfilterable-float
texture. Copy them and adapt them when your inputs differ.

- **Compile-time switches are `override` constants.** Set them per pipeline (wgpu:
  `PipelineCompilationOptions::constants`; WebGPU: `compute.constants`):
  `CORTAO_PERSPECTIVE_PROJECTION`, `CORTAO_SAMPLE_COUNT`, `CORTAO_RAY_COUNT`,
  `CORTAO_OCCLUDER_LOOKUP_TABLE`, `CORTAO_BENT_NORMAL_LOOKUP_TABLE`, `CORTAO_PACKED_NORMAL_BITS`,
  `CORTAO_FILTER_BENT_NORMAL`, `CORTAO_LARGE_FILTER`, `CORTAO_FILTER_SYMMETRY`.
- **Parameters.** `Configuration` takes the same settings as the original's
  `CORTAOConfigurationParameters`. `PrepassParameters`, `MainPassParameters` and
  `FilterPassParameters` derive each pass's uniform block from it (`bytemuck::Pod`). Matrices
  are column-major, `[column][row]` (glam's `to_cols_array_2d`).
- **Lookup tables.** `tables::occluder_lookup_table_bytes(24, 16, rays)` gives the 3D table's
  texels (`R8Uint`/`R16Uint`/`R32Uint` for 8/16/32 rays).
  `tables::bent_normal_lookup_table_words(rays)` gives the bent normal table as one
  `pack4x8snorm` word per entry, for a storage buffer. It exists for 8 or 16 rays only; bind
  any small buffer when it is unused.
- **Dispatch sizes.** Prepass: workgroups of 8×8, one invocation per pixel. Main pass: 16×16,
  one per pixel. Filter pass: 8×8, one per 2×2 block, so
  `ceil((width + 1) / 16) × ceil((height + 1) / 16)`.

## What differs from the HLSL/GLSL original

- **No preprocessor:** the defines are `override` constants, as above.
- **No 16-bit path:** the arithmetic is all 32-bit. WGSL has no 16-bit integers, and `f16`
  is optional in WebGPU. The original recommends 32-bit for the main pass anyway; its filter
  pass is up to 25% faster in 16-bit.
- **No integer sampling or border clamping in WebGPU.** The callbacks read texels with
  `textureLoad` instead. `cortao_nearestMipLevel`, `cortao_nearestTexel` and
  `cortao_gatherTexels` reproduce nearest-mip selection, border clamping and gather order
  exactly.
- **The prepass is two dispatches.** WebGPU's default limit is 4 storage textures per stage, so
  `reference/prepass.wgsl` writes mips 0–3 and `reference/prepass_continue.wgsl` writes 4–7
  from mip 3 (`cortao_prepassContinue`). The result is identical to the single dispatch.
  Skip the second dispatch when `mipCount ≤ 4`.
- **One filter callback shape.** `cortao_gatherSSAO` always returns a `mat4x4<f32>` (column i
  is texel i). `cortao_writeFilterOutput` always takes a `vec4<f32>`; occlusion-only filtering
  uses `.x`.
- **The filter's small loops are unrolled by hand.** The original marks them `[unroll]`. Written
  as WGSL loops, they index matrices at runtime, and that crashed the AMD Vulkan (RADV) shader
  compiler.

## Tests

```bash
cargo test                       # parameters, tables vs. the original files, WGSL validation
cargo test --test against_original -- --ignored --nocapture   # original vs. port on the GPU
```

The GPU comparison needs `glslc` (shaderc) and a Vulkan GPU that supports SPIR-V passthrough
and border clamping. `reference/original/` holds the original's headers and tables, used
only by the tests.

## License

MIT, as the original. See [LICENSE](LICENSE). CORTAO itself © 2026 Daniel Isheden.
