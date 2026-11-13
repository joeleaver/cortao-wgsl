// CORTAO for WGSL, reference prepass: levels 0 to 3 of the packed data buffer, from the depth
// buffer and a view-space normal buffer. MIT License (see cortao_base.wgsl).
// Compose: cortao_base.wgsl + cortao_prepass.wgsl + this file.
// Dispatch ceil(width / 8) x ceil(height / 8) workgroups, with `mipCount` at most 4
// (`PrepassParameters::first_dispatch`); then run prepass_continue.wgsl for levels 4 to 7.

@group(0) @binding(0) var<uniform> parameters: CORTAOPrepassParameters;
@group(0) @binding(1) var depthBuffer: texture_2d<f32>; // depth32float (as unfilterable-float) or r32float
@group(0) @binding(2) var normalBuffer: texture_2d<f32>; // view-space normals in xyz
@group(0) @binding(3) var packedMip0: texture_storage_2d<r32uint, write>;
@group(0) @binding(4) var packedMip1: texture_storage_2d<r32uint, write>;
@group(0) @binding(5) var packedMip2: texture_storage_2d<r32uint, write>;
@group(0) @binding(6) var packedMip3: texture_storage_2d<r32uint, write>;

fn cortao_loadDepth(pixelCoords: vec2<i32>) -> f32 {
    return textureLoad(depthBuffer, pixelCoords, 0).r;
}

fn cortao_loadNormal(pixelCoords: vec2<i32>) -> vec3<f32> {
    // Do any normal transformation/unpacking/decoding here.
    return textureLoad(normalBuffer, pixelCoords, 0).xyz;
}

fn cortao_writePackedData(pixelCoords: vec2<i32>, mipLevel: i32, data: u32) {
    let value = vec4<u32>(data, 0u, 0u, 0u);
    switch mipLevel {
        case 0: { textureStore(packedMip0, pixelCoords, value); }
        case 1: { textureStore(packedMip1, pixelCoords, value); }
        case 2: { textureStore(packedMip2, pixelCoords, value); }
        case 3: { textureStore(packedMip3, pixelCoords, value); }
        default: {}
    }
}

// Not called by cortao_prepass.
fn cortao_loadPackedData(pixelCoords: vec2<i32>, mipLevel: i32) -> u32 {
    return 0u;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    cortao_prepass(id.xy, parameters);
}
