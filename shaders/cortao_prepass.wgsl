// CORTAO for WGSL: the prepass. Needs cortao_base.wgsl. MIT License (see cortao_base.wgsl).
//
// Linearises the depth buffer, packs a low-precision normal into its low bits, and writes the
// packed value to every mip level of the packed data buffer, each mip texel keeping one of its
// four children (chosen by a per-level dither), all from one full-resolution dispatch.
//
// WebGPU's default limit is 4 storage textures per shader stage, so a pass usually cannot bind
// all 8 mips at once. `cortao_prepass` writes levels 0 to `params.mipCount - 1`; dispatch it with
// `mipCount` at most the number of mips the pass binds (4 with the default limits), then dispatch
// `cortao_prepassContinue` over the last level written to carry on from there. Each texel of the
// continuing levels keeps exactly the value the single-dispatch original would have written.
//
// Callbacks to implement:
//   fn cortao_loadDepth(pixelCoords: vec2<i32>) -> f32
//   fn cortao_loadNormal(pixelCoords: vec2<i32>) -> vec3<f32>     // normalised, view space
//   fn cortao_writePackedData(pixelCoords: vec2<i32>, mipLevel: i32, data: u32)
// and, for `cortao_prepassContinue`:
//   fn cortao_loadPackedData(pixelCoords: vec2<i32>, mipLevel: i32) -> u32

// true for a perspective projection, false for orthographic.
override CORTAO_PERSPECTIVE_PROJECTION: bool = true;

struct CORTAOPrepassParameters {
    packedResolution: u32,
    depthScale: f32,
    depthOffset: f32,
    randomSeed: u32,

    mipCount: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

// Carries the packed value of the texel at `pixelCoords` of level `level - 1` (whose
// resolution is `resolution`) down to the levels from `level` up to `params.mipCount - 1`.
fn cortao_prepassDownsample(startCoords: vec2<u32>, startResolution: vec2<u32>, level: u32, packedData: u32, params: CORTAOPrepassParameters) {
    var pixelCoords = startCoords;
    var resolution = startResolution;
    for (var mipLevel = level; mipLevel < 8u; mipLevel++) {

        let nextCoords = pixelCoords >> vec2<u32>(1u);

        let dither = vec2<u32>(params.randomSeed) >> vec2<u32>(mipLevel * 2u, mipLevel * 2u + 1u);

        let wrongPixel = (((pixelCoords.x ^ nextCoords.y ^ dither.x) | (pixelCoords.y ^ nextCoords.x ^ dither.y)) & 0x1u) != 0u;

        pixelCoords = nextCoords;
        resolution = max(resolution >> vec2<u32>(1u), vec2<u32>(1u));

        if (mipLevel >= params.mipCount || wrongPixel || pixelCoords.x >= resolution.x || pixelCoords.y >= resolution.y) {
            return;
        }

        cortao_writePackedData(vec2<i32>(pixelCoords), i32(mipLevel), packedData);
    }
}

fn cortao_prepass(pixelCoordsIn: vec2<u32>, params: CORTAOPrepassParameters) {

    // check if our pixel is out-of-bounds and early out in that case
    let resolution = cortao_getResolution(params.packedResolution);
    if (pixelCoordsIn.x >= resolution.x || pixelCoordsIn.y >= resolution.y) {
        return;
    }

    let depth = cortao_loadDepth(vec2<i32>(pixelCoordsIn));

    var packedData = 0u;

    let skyBoxDepth = select(0.0, 1.0, params.depthScale < 0.0); // Handles both standard and inverted depth
    if (depth != skyBoxDepth) {

        // Calculate the linear depth
        var linearDepth: f32;
        if (CORTAO_PERSPECTIVE_PROJECTION) {
            linearDepth = 1.0 / fma(depth, params.depthScale, params.depthOffset);
        } else {
            linearDepth = fma(depth, params.depthScale, params.depthOffset);
        }

        let normal = cortao_loadNormal(vec2<i32>(pixelCoordsIn));

        // Pack a spheremap encoded normal
        let seed = cortao_hash(cortao_seed(pixelCoordsIn) + params.randomSeed);
        let dither = vec2<f32>(cortao_unpack16(seed)) * (1.0 / f32(0xFFFFu));
        let mask = (1u << CORTAO_PACKED_NORMAL_BITS) - 1u;
        let packedNormal = cortao_encodeNormal(normal, dither);

        // Merge the linear depth and the packed normal
        packedData = (bitcast<u32>(linearDepth) & ~mask) | (packedNormal & mask);
    }

    cortao_writePackedData(vec2<i32>(pixelCoordsIn), 0, packedData);

    cortao_prepassDownsample(pixelCoordsIn, resolution, 1u, packedData, params);
}

// Writes levels `fromLevel` up to `params.mipCount - 1`, from the texel at `pixelCoords` of
// level `fromLevel - 1`, which an earlier dispatch wrote. Dispatch it over level
// `fromLevel - 1`'s resolution.
fn cortao_prepassContinue(pixelCoords: vec2<u32>, fromLevel: u32, params: CORTAOPrepassParameters) {
    var resolution = cortao_getResolution(params.packedResolution);
    for (var level = 1u; level < fromLevel; level++) {
        resolution = max(resolution >> vec2<u32>(1u), vec2<u32>(1u));
    }
    if (pixelCoords.x >= resolution.x || pixelCoords.y >= resolution.y) {
        return;
    }
    let packedData = cortao_loadPackedData(vec2<i32>(pixelCoords), i32(fromLevel) - 1);
    cortao_prepassDownsample(pixelCoords, resolution, fromLevel, packedData, params);
}
