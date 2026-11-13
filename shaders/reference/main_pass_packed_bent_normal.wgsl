// CORTAO for WGSL, reference main pass writing occlusion and the view-space bent normal packed
// into one r32uint texel, for the filter pass. MIT License (see cortao_base.wgsl).
// Compose: cortao_base.wgsl + cortao_main_pass.wgsl + main_pass_common.wgsl + this file.

@group(0) @binding(5) var outputBuffer: texture_storage_2d<r32uint, write>;

fn cortao_writeResult(pixelCoords: vec2<i32>, occlusion: f32, bentNormal: vec3<f32>, tangentSpaceTransform: mat3x3<f32>, rayBits: u32) {
    textureStore(outputBuffer, pixelCoords, vec4<u32>(pack4x8snorm(vec4<f32>(occlusion, bentNormal)), 0u, 0u, 0u));
}
