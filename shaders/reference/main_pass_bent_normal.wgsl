// CORTAO for WGSL, reference main pass writing occlusion and the view-space bent normal, for use
// without the filter pass. MIT License (see cortao_base.wgsl).
// Compose: cortao_base.wgsl + cortao_main_pass.wgsl + main_pass_common.wgsl + this file.

@group(0) @binding(5) var outputBuffer: texture_storage_2d<rgba8snorm, write>;

fn cortao_writeResult(pixelCoords: vec2<i32>, occlusion: f32, bentNormal: vec3<f32>, tangentSpaceTransform: mat3x3<f32>, rayBits: u32) {
    textureStore(outputBuffer, pixelCoords, vec4<f32>(occlusion, bentNormal));
}
