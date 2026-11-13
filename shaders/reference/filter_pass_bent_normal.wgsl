// CORTAO for WGSL, reference filter pass for occlusion and bent normal, reading
// main_pass_packed_bent_normal.wgsl's output. MIT License (see cortao_base.wgsl).
// Compose: cortao_base.wgsl + cortao_filter_pass.wgsl + filter_pass_common.wgsl + this file,
// with CORTAO_FILTER_BENT_NORMAL true.

@group(0) @binding(2) var ssaoBuffer: texture_2d<u32>; // r32uint, pack4x8snorm(occlusion, bent normal)
@group(0) @binding(3) var outputBuffer: texture_storage_2d<rgba8snorm, write>;

fn cortao_loadSSAOTexel(texel: vec2<i32>, size: vec2<u32>) -> vec4<f32> {
    if (!cortao_inBounds(texel, size)) {
        return vec4<f32>(0.0); // border
    }
    return unpack4x8snorm(textureLoad(ssaoBuffer, texel, 0).r);
}

fn cortao_gatherSSAO(texCoords: vec2<f32>, offset: vec2<i32>) -> mat4x4<f32> {
    let size = textureDimensions(ssaoBuffer, 0);
    let t = cortao_gatherTexels(texCoords, size, offset);
    return mat4x4<f32>(cortao_loadSSAOTexel(t.x, size), cortao_loadSSAOTexel(t.y, size), cortao_loadSSAOTexel(t.z, size), cortao_loadSSAOTexel(t.w, size));
}

fn cortao_writeFilterOutput(pixelCoords: vec2<i32>, ssao: vec4<f32>) {
    textureStore(outputBuffer, pixelCoords, ssao);
}
