// CORTAO for WGSL, reference prepass continuation: levels 4 to 7 of the packed data buffer,
// from level 3. MIT License (see cortao_base.wgsl).
// Compose: cortao_base.wgsl + cortao_prepass.wgsl + this file.
// Dispatch over level 3: ceil(max(width >> 3, 1) / 8) x ceil(max(height >> 3, 1) / 8)
// workgroups, with the full `mipCount`. Skip it when `mipCount` is 4 or less.

@group(0) @binding(0) var<uniform> parameters: CORTAOPrepassParameters;
@group(0) @binding(1) var packedMip3: texture_2d<u32>; // a view of level 3 alone
@group(0) @binding(3) var packedMip4: texture_storage_2d<r32uint, write>;
@group(0) @binding(4) var packedMip5: texture_storage_2d<r32uint, write>;
@group(0) @binding(5) var packedMip6: texture_storage_2d<r32uint, write>;
@group(0) @binding(6) var packedMip7: texture_storage_2d<r32uint, write>;

// Not called by cortao_prepassContinue.
fn cortao_loadDepth(pixelCoords: vec2<i32>) -> f32 {
    return 0.0;
}

// Not called by cortao_prepassContinue.
fn cortao_loadNormal(pixelCoords: vec2<i32>) -> vec3<f32> {
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn cortao_loadPackedData(pixelCoords: vec2<i32>, mipLevel: i32) -> u32 {
    return textureLoad(packedMip3, pixelCoords, 0).r;
}

fn cortao_writePackedData(pixelCoords: vec2<i32>, mipLevel: i32, data: u32) {
    let value = vec4<u32>(data, 0u, 0u, 0u);
    switch mipLevel {
        case 4: { textureStore(packedMip4, pixelCoords, value); }
        case 5: { textureStore(packedMip5, pixelCoords, value); }
        case 6: { textureStore(packedMip6, pixelCoords, value); }
        case 7: { textureStore(packedMip7, pixelCoords, value); }
        default: {}
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    cortao_prepassContinue(id.xy, 4u, parameters);
}
