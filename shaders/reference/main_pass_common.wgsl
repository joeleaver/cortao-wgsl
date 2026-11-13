// CORTAO for WGSL, reference main pass inputs shared by the main_pass_*.wgsl entry files.
// MIT License (see cortao_base.wgsl).

@group(0) @binding(0) var<uniform> parameters: CORTAOMainPassParameters;
@group(0) @binding(1) var packedDataBuffer: texture_2d<u32>; // r32uint, all its mips
@group(0) @binding(2) var normalBuffer: texture_2d<f32>; // view-space normals in xyz
@group(0) @binding(3) var occluderLookupTable: texture_3d<u32>; // r8/r16/r32uint for 8/16/32 rays
@group(0) @binding(4) var<storage, read> bentNormalLookupTable: array<u32>; // pack4x8snorm(x, y, 0, 0)

fn cortao_samplePackedData(texCoords: vec2<f32>, mipmap: f32) -> u32 {
    let level = cortao_nearestMipLevel(mipmap, textureNumLevels(packedDataBuffer));
    let size = textureDimensions(packedDataBuffer, level);
    let texel = cortao_nearestTexel(texCoords, size);
    if (!cortao_inBounds(texel, size)) {
        return 0u; // border
    }
    return textureLoad(packedDataBuffer, texel, level).r;
}

fn cortao_loadNormal(pixelCoords: vec2<i32>) -> vec3<f32> {
    // Do any normal transformation/unpacking/decoding here.
    return textureLoad(normalBuffer, pixelCoords, 0).xyz;
}

fn cortao_sampleOccluderLookupTable(texCoords: vec3<f32>) -> u32 {
    let size = textureDimensions(occluderLookupTable);
    let texel = clamp(cortao_nearestTexel3(texCoords, size), vec3<i32>(0), vec3<i32>(size) - 1);
    return textureLoad(occluderLookupTable, texel, 0).r;
}

fn cortao_loadBentNormalLookupTable(rayBits: u32) -> vec3<f32> {
    let bentNormal = unpack4x8snorm(bentNormalLookupTable[rayBits]).xy;
    return vec3<f32>(bentNormal, sqrt(max(0.0, 1.0 - dot(bentNormal, bentNormal)))); // reconstruct z
}

// Reorders the threads inside the workgroup to be more cache coherent
fn cortao_deinterleave(index: u32) -> vec2<u32> {
    var p = vec2<u32>(index, index) >> vec2<u32>(0u, 1u);

    p &= vec2<u32>(0x5555u);
    p = (p ^ (p >> vec2<u32>(1u))) & vec2<u32>(0x3333u);
    p = (p ^ (p >> vec2<u32>(2u))) & vec2<u32>(0x0f0fu);
    p = (p ^ (p >> vec2<u32>(4u))) & vec2<u32>(0x00ffu);

    return p;
}

// Dispatch ceil(width / 16) x ceil(height / 16) workgroups.
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) index: u32) {
    cortao_mainPass(group.xy * 16u + cortao_deinterleave(index), parameters);
}
