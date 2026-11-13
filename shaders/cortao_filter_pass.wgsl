// CORTAO for WGSL: the optional filter pass. Needs cortao_base.wgsl. MIT License (see
// cortao_base.wgsl).
//
// A cheap sparse bilateral blur meant to feed a temporal filter (TAA, temporal upscaling),
// which is not included. Each invocation filters a 2x2 block, so dispatch
// ceil((width + 1) / 16) x ceil((height + 1) / 16) workgroups of 8x8.
//
// Callbacks to implement:
//   fn cortao_gatherPackedData(texCoords: vec2<f32>, offset: vec2<i32>) -> vec4<u32>
//       the packed data buffer's level 0, gather order, texels outside the image read 0u:
//       see cortao_gatherTexels
//   fn cortao_gatherSSAO(texCoords: vec2<f32>, offset: vec2<i32>) -> mat4x4<f32>
//       the main pass's output, gather order. Column i is texel i's (occlusion, bent normal);
//       when CORTAO_FILTER_BENT_NORMAL is false only column i's x is used.
//   fn cortao_writeFilterOutput(pixelCoords: vec2<i32>, ssao: vec4<f32>)
//       occlusion in x and, when CORTAO_FILTER_BENT_NORMAL is true, the bent normal in yzw.
//
// The original's two callback shapes (occlusion only, or occlusion and bent normal) are one
// shape here; the occlusion-only filter does the same arithmetic on column x alone.

// Filter the bent normal as well as the occlusion.
override CORTAO_FILTER_BENT_NORMAL: bool = false;
// A 13-tap filter instead of a 5-tap one.
override CORTAO_LARGE_FILTER: bool = false;
// Keep kernel weights symmetric, so the kernel's centre does not skew: better contact shadows,
// less filtering near edges.
override CORTAO_FILTER_SYMMETRY: bool = false;

struct CORTAOFilterPassParameters {
    packedResolution: u32,
    depthSensitivity: f32,
    normalSensitivity: f32,
    _padding0: u32,
}

struct CORTAOWeightParams {
    depthMul: vec4<f32>,
    depthOffset: f32,
    deltaX: vec4<f32>,
    deltaY: vec4<f32>,

    centerNormalX: vec4<i32>,
    centerNormalY: vec4<i32>,

    normalErrorMul: f32,
}

fn cortao_unpackNormalX(packedData: vec4<u32>) -> vec4<i32> {
    let normalBits = CORTAO_PACKED_NORMAL_BITS / 2u;
    return extractBits(bitcast<vec4<i32>>(packedData), 0u, normalBits);
}

fn cortao_unpackNormalY(packedData: vec4<u32>) -> vec4<i32> {
    let normalBits = CORTAO_PACKED_NORMAL_BITS / 2u;
    return extractBits(bitcast<vec4<i32>>(packedData), normalBits, normalBits);
}

fn cortao_calculateWeight(params: CORTAOWeightParams, packedData: vec4<u32>, offset: vec2<i32>) -> vec4<f32> {
    let sampleDepth = fma(cortao_unpackDepth4(packedData), params.depthMul, vec4<f32>(params.depthOffset));
    var error = abs(fma(params.deltaX, vec4<f32>(f32(offset.x)), fma(params.deltaY, vec4<f32>(f32(offset.y)), sampleDepth)));

    let normalX = cortao_unpackNormalX(packedData) - params.centerNormalX;
    let normalY = cortao_unpackNormalY(packedData) - params.centerNormalY;
    let normalError = vec4<f32>(normalX * normalX + normalY * normalY);
    error = fma(normalError, vec4<f32>(params.normalErrorMul), error);

    return exp2(-error);
}

struct CORTAOFilterSums {
    // Column i: texel i's weighted (occlusion, bent normal) sum; only x is kept when the bent
    // normal is not filtered.
    ssao: mat4x4<f32>,
    weight: vec4<f32>,
}

// The original's loops over the four gathered texels are unrolled by hand: they are marked for
// unrolling there, and some drivers' compilers fail on the runtime matrix indexing a WGSL loop
// would need.
fn cortao_filterAccumulate(sums: ptr<function, CORTAOFilterSums>, texCoords: vec2<f32>, offset: vec2<i32>, weight: vec4<f32>) {
    let data = cortao_gatherSSAO(texCoords, offset);
    if (CORTAO_FILTER_BENT_NORMAL) {
        (*sums).ssao[0] = fma(data[0], vec4<f32>(weight.x), (*sums).ssao[0]);
        (*sums).ssao[1] = fma(data[1], vec4<f32>(weight.y), (*sums).ssao[1]);
        (*sums).ssao[2] = fma(data[2], vec4<f32>(weight.z), (*sums).ssao[2]);
        (*sums).ssao[3] = fma(data[3], vec4<f32>(weight.w), (*sums).ssao[3]);
    } else {
        let occlusion = vec4<f32>(data[0].x, data[1].x, data[2].x, data[3].x);
        let current = vec4<f32>((*sums).ssao[0].x, (*sums).ssao[1].x, (*sums).ssao[2].x, (*sums).ssao[3].x);
        let sum = fma(occlusion, weight, current);
        (*sums).ssao[0].x = sum.x;
        (*sums).ssao[1].x = sum.y;
        (*sums).ssao[2].x = sum.z;
        (*sums).ssao[3].x = sum.w;
    }
}

// One pair of taps, at -offset and +offset.
fn cortao_filterTaps(sums: ptr<function, CORTAOFilterSums>, weightParams: CORTAOWeightParams, texCoords: vec2<f32>, offset: vec2<i32>) {
    var weightA = cortao_calculateWeight(weightParams, cortao_gatherPackedData(texCoords, -offset), -offset);
    var weightB = cortao_calculateWeight(weightParams, cortao_gatherPackedData(texCoords, offset), offset);

    if (CORTAO_FILTER_SYMMETRY) {
        let minWeight = min(weightA, weightB);
        weightA = minWeight;
        weightB = minWeight;
        (*sums).weight = fma(minWeight, vec4<f32>(2.0), (*sums).weight);
    } else {
        (*sums).weight += weightA + weightB;
    }

    cortao_filterAccumulate(sums, texCoords, -offset, weightA);
    cortao_filterAccumulate(sums, texCoords, offset, weightB);
}

fn cortao_filterWrite(outputCoords: vec2<u32>, centerPackedData: u32, ssao: vec4<f32>, resolution: vec2<u32>) {
    if (centerPackedData != 0u && outputCoords.x < resolution.x && outputCoords.y < resolution.y) {
        cortao_writeFilterOutput(vec2<i32>(outputCoords), ssao);
    }
}

fn cortao_filterPass(threadCoords: vec2<u32>, parameters: CORTAOFilterPassParameters) {
    let resolution = cortao_getResolution(parameters.packedResolution);

    let pixelCoords = (threadCoords << vec2<u32>(1u)) | vec2<u32>(1u);

    // ">" is correct here, as we write to pixelCoords.x-1 as well.
    if (pixelCoords.x > resolution.x || pixelCoords.y > resolution.y) {
        return;
    }

    let texCoords = vec2<f32>(pixelCoords) / vec2<f32>(resolution);

    let centerPackedData = cortao_gatherPackedData(texCoords, vec2<i32>(0, 0));

    if (all(centerPackedData == vec4<u32>(0u))) {
        return;
    }

    let centerDepth = cortao_unpackDepth4(centerPackedData);
    let inverseCenterDepth = 1.0 / centerDepth;

    let deltaX = (centerDepth.xxww - centerDepth.yyzz) * inverseCenterDepth;
    let deltaY = (centerDepth.wzzw - centerDepth.xyyx) * inverseCenterDepth;

    let normalBits = CORTAO_PACKED_NORMAL_BITS / 2u;

    var weightParams: CORTAOWeightParams;
    weightParams.depthMul = parameters.depthSensitivity * inverseCenterDepth;
    weightParams.depthOffset = -parameters.depthSensitivity;
    weightParams.deltaX = deltaX * parameters.depthSensitivity;
    weightParams.deltaY = deltaY * parameters.depthSensitivity;
    weightParams.centerNormalX = cortao_unpackNormalX(centerPackedData);
    weightParams.centerNormalY = cortao_unpackNormalY(centerPackedData);
    weightParams.normalErrorMul = parameters.normalSensitivity * (1.0 / f32(1u << normalBits));

    var sums = CORTAOFilterSums(mat4x4<f32>(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0)), vec4<f32>(1.0));
    cortao_filterAccumulate(&sums, texCoords, vec2<i32>(0, 0), vec4<f32>(1.0));

    // The taps, in the original's order.
    if (CORTAO_LARGE_FILTER) {
        cortao_filterTaps(&sums, weightParams, texCoords, vec2<i32>(3, 0));
        cortao_filterTaps(&sums, weightParams, texCoords, vec2<i32>(6, 0));
        cortao_filterTaps(&sums, weightParams, texCoords, vec2<i32>(3, 3));
        cortao_filterTaps(&sums, weightParams, texCoords, vec2<i32>(0, 3));
        cortao_filterTaps(&sums, weightParams, texCoords, vec2<i32>(0, 6));
        cortao_filterTaps(&sums, weightParams, texCoords, vec2<i32>(3, -3));
    } else {
        cortao_filterTaps(&sums, weightParams, texCoords, vec2<i32>(3, 0));
        cortao_filterTaps(&sums, weightParams, texCoords, vec2<i32>(0, 3));
    }

    let inverseWeight = 1.0 / sums.weight;
    let coords = vec4<u32>(pixelCoords - 1u, pixelCoords);
    cortao_filterWrite(coords.xw, centerPackedData.x, sums.ssao[0] * inverseWeight.x, resolution);
    cortao_filterWrite(coords.zw, centerPackedData.y, sums.ssao[1] * inverseWeight.y, resolution);
    cortao_filterWrite(coords.zy, centerPackedData.z, sums.ssao[2] * inverseWeight.z, resolution);
    cortao_filterWrite(coords.xy, centerPackedData.w, sums.ssao[3] * inverseWeight.w, resolution);
}
