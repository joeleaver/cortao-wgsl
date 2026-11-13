// CORTAO for WGSL: shared definitions.
//
// A WGSL port of the CORTAO reference implementation v1.0.1 by Daniel Isheden
// (https://github.com/theagentd/CORTAO, commit 9e8771f), MIT License.
//
// MIT License
//
// Copyright (c) 2026 Daniel Isheden
// Copyright (c) 2026 Joe Leaver (WGSL port)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Differences from the HLSL/GLSL original:
// - WGSL has no preprocessor. The original's compile-time defines are pipeline-overridable
//   constants (`override`), set per pipeline with `constants` in the pipeline descriptor.
// - There is no 16-bit path: every `cortao_half`/`cortao_short` is 32-bit here (`f16` is an
//   optional WebGPU feature and WGSL has no 16-bit integers). This matches the original with
//   CORTAO_16_BIT = 0, which the original recommends for the main pass.
// - WebGPU cannot sample integer textures and has no border-clamp address mode, so the input
//   callbacks read texels with `textureLoad`. `cortao_nearestMipLevel`, `cortao_nearestTexel` and
//   `cortao_gatherTexels` below reproduce nearest filtering, border clamping and gather exactly.

// The number of bits of the packed data buffer that hold the low-precision normal.
// Even, from 4 to 20. Must be the same for every CORTAO pass.
override CORTAO_PACKED_NORMAL_BITS: u32 = 10u;

const CORTAO_PI: f32 = 3.1415926535897932384626433832795;
const CORTAO_PI2: f32 = 6.283185307179586476925286766559; // PI * 2.0
const CORTAO_GOLDEN_ANGLE: f32 = 2.3999632297286533222315555066336; // PI * (3.0 - sqrt(5.0))

fn cortao_seed(coords: vec2<u32>) -> u32 {
    let q = coords * vec2<u32>(1719413u * 929u, 140473u * 2467u * 11u);
    return q.x ^ q.y;
}

fn cortao_hash(n: u32) -> u32 {
    return n * (n ^ (n >> 15u));
}

fn cortao_hash2(n: vec2<u32>) -> vec2<u32> {
    return n * (n ^ (n >> vec2<u32>(15u)));
}

fn cortao_unpack16(x: u32) -> vec2<u32> {
    return vec2<u32>(x & 0xFFFFu, x >> 16u);
}

fn cortao_encodeNormal(normal: vec3<f32>, dither: vec2<f32>) -> u32 {
    let bitsPerComponent = CORTAO_PACKED_NORMAL_BITS / 2u;

    let componentMask = (1u << bitsPerComponent) - 1u;
    let componentMax = componentMask >> 1u;

    let encodedNormal = normal.xy * inverseSqrt(fma(normal.z, 2.0, 2.0));

    let values = vec2<i32>(floor(fma(encodedNormal, vec2<f32>(f32(componentMax)), dither)));

    return (bitcast<u32>(values.x) & componentMask) | ((bitcast<u32>(values.y) & componentMask) << bitsPerComponent);
}

fn cortao_decodeNormal(packedData: u32) -> vec3<f32> {
    let bitsPerComponent = CORTAO_PACKED_NORMAL_BITS / 2u;

    let componentMask = (1u << bitsPerComponent) - 1u;
    let componentMax = componentMask >> 1u;

    // The original shifts left then arithmetic-shifts right to sign-extend each component;
    // extractBits on a signed value does the same.
    let packedSigned = bitcast<i32>(packedData);
    let values = vec2<i32>(
        extractBits(packedSigned, 0u, bitsPerComponent),
        extractBits(packedSigned, bitsPerComponent, bitsPerComponent),
    );

    let e = vec2<f32>(values) * (1.0 / f32(componentMax));

    let z = fma(-e.x, e.x, fma(-e.y, e.y, 1.0));
    return vec3<f32>(e * (2.0 * sqrt(z)), fma(z, 2.0, -1.0));
}

fn cortao_unpackDepth(packedData: u32) -> f32 {
    return bitcast<f32>(packedData);
}

fn cortao_unpackDepth4(packedData: vec4<u32>) -> vec4<f32> {
    return bitcast<vec4<f32>>(packedData);
}

fn cortao_getResolution(packedResolution: u32) -> vec2<u32> {
    return cortao_unpack16(packedResolution);
}

// --- Texture-read helpers for the input callbacks -------------------------------------------

// The mip level a sampler with nearest mipmap filtering picks for an explicit `lod`
// (Vulkan: ceil(lod + 0.5) - 1, clamped to the texture's levels).
fn cortao_nearestMipLevel(lod: f32, levelCount: u32) -> u32 {
    let level = ceil(max(lod, 0.0) + 0.5) - 1.0;
    return u32(clamp(level, 0.0, f32(levelCount - 1u)));
}

// The texel nearest filtering reads at `texCoords` in an image of `size`. It may lie outside
// the image; the caller decides between border (return 0) and clamp-to-edge.
fn cortao_nearestTexel(texCoords: vec2<f32>, size: vec2<u32>) -> vec2<i32> {
    return vec2<i32>(floor(texCoords * vec2<f32>(size)));
}

fn cortao_nearestTexel3(texCoords: vec3<f32>, size: vec3<u32>) -> vec3<i32> {
    return vec3<i32>(floor(texCoords * vec3<f32>(size)));
}

// Whether `texel` lies inside an image of `size`.
fn cortao_inBounds(texel: vec2<i32>, size: vec2<u32>) -> bool {
    return all(texel >= vec2<i32>(0)) && all(texel < vec2<i32>(size));
}

// The four texels a gather returns at `texCoords` plus `offset`, in gather order:
// (i0, j1), (i1, j1), (i1, j0), (i0, j0).
struct CORTAOGatherTexels {
    x: vec2<i32>,
    y: vec2<i32>,
    z: vec2<i32>,
    w: vec2<i32>,
}

fn cortao_gatherTexels(texCoords: vec2<f32>, size: vec2<u32>, offset: vec2<i32>) -> CORTAOGatherTexels {
    let base = vec2<i32>(floor(texCoords * vec2<f32>(size) - 0.5)) + offset;
    return CORTAOGatherTexels(base + vec2<i32>(0, 1), base + vec2<i32>(1, 1), base + vec2<i32>(1, 0), base);
}

// Whether all four of `texels` lie inside an image of `size`: then a hardware `textureGather` at
// `cortao_gatherCoords(texels, size)` returns exactly them, and no border is involved.
fn cortao_gatherInBounds(texels: CORTAOGatherTexels, size: vec2<u32>) -> bool {
    return cortao_inBounds(texels.w, size) && cortao_inBounds(texels.y, size);
}

// Texture coordinates at which `textureGather` fetches exactly `texels`: the shared corner of the
// four, where the gather footprint cannot round either way.
fn cortao_gatherCoords(texels: CORTAOGatherTexels, size: vec2<u32>) -> vec2<f32> {
    return (vec2<f32>(texels.w) + 1.0) / vec2<f32>(size);
}
