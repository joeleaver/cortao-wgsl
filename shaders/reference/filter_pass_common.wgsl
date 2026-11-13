// CORTAO for WGSL, reference filter pass inputs shared by the filter_pass_*.wgsl entry files.
// MIT License (see cortao_base.wgsl).

@group(0) @binding(0) var<uniform> parameters: CORTAOFilterPassParameters;
@group(0) @binding(1) var packedDataBuffer: texture_2d<u32>; // r32uint; level 0 is read

fn cortao_loadPackedTexel(texel: vec2<i32>, size: vec2<u32>) -> u32 {
    if (!cortao_inBounds(texel, size)) {
        return 0u; // border
    }
    return textureLoad(packedDataBuffer, texel, 0).r;
}

fn cortao_gatherPackedData(texCoords: vec2<f32>, offset: vec2<i32>) -> vec4<u32> {
    let size = textureDimensions(packedDataBuffer, 0);
    let t = cortao_gatherTexels(texCoords, size, offset);
    return vec4<u32>(cortao_loadPackedTexel(t.x, size), cortao_loadPackedTexel(t.y, size), cortao_loadPackedTexel(t.z, size), cortao_loadPackedTexel(t.w, size));
}

// Each invocation filters a 2x2 block: dispatch ceil((width + 1) / 16) x ceil((height + 1) / 16).
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    cortao_filterPass(id.xy, parameters);
}
