// CORTAO for WGSL: the main pass. Needs cortao_base.wgsl. MIT License (see cortao_base.wgsl).
//
// Samples the packed data buffer around each pixel, turns the samples into circular occluders,
// and traces up to 32 rays per pixel against them, giving occlusion and a bent normal.
//
// Callbacks to implement:
//   fn cortao_loadNormal(pixelCoords: vec2<i32>) -> vec3<f32>     // normalised, view space
//   fn cortao_samplePackedData(texCoords: vec2<f32>, mipmap: f32) -> u32
//       nearest filtering, border clamping to 0u: see cortao_nearestMipLevel/cortao_nearestTexel
//   fn cortao_sampleOccluderLookupTable(texCoords: vec3<f32>) -> u32
//       nearest filtering, clamp to edge. Called only when CORTAO_OCCLUDER_LOOKUP_TABLE is true;
//       otherwise any body will do.
//   fn cortao_loadBentNormalLookupTable(rayBits: u32) -> vec3<f32>
//       Called only when CORTAO_BENT_NORMAL_LOOKUP_TABLE is true; otherwise any body will do.
//   fn cortao_writeResult(pixelCoords: vec2<i32>, occlusion: f32, bentNormal: vec3<f32>,
//                         tangentSpaceTransform: mat3x3<f32>, rayBits: u32)

// true for a perspective projection, false for orthographic.
override CORTAO_PERSPECTIVE_PROJECTION: bool = true;
// Samples per pixel: the main quality/performance trade-off. At least 1; 4 to 32 recommended.
override CORTAO_SAMPLE_COUNT: u32 = 16u;
// Rays traced per pixel, 8 to 32. Must match the occluder lookup table when one is used.
override CORTAO_RAY_COUNT: u32 = 16u;
// Trace all rays against an occluder at once with the occluder lookup table (fast), or one by
// one (for testing).
override CORTAO_OCCLUDER_LOOKUP_TABLE: bool = true;
// Resolve the bent normal with the bent normal lookup table (8 or 16 rays only).
override CORTAO_BENT_NORMAL_LOOKUP_TABLE: bool = false;

struct CORTAOMainPassParameters {
    packedResolution: u32,
    randomSeed: u32,
    normalBias: f32,
    radius: f32,

    unprojectParams: vec4<f32>,

    projectionParams: vec4<f32>,

    maxApparentCircleRadiusCoefficient: f32,
    mipBias: f32,
    inverseMaxSlopeCompensation: f32,
    backfaceThreshold: f32,

    backfaceBoost: f32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

// The tangent-space direction of ray `rayIndex`: a golden-angle spiral over the hemisphere.
fn cortao_rayDirection(rayIndex: u32) -> vec3<f32> {
    let angle = f32(rayIndex) * CORTAO_GOLDEN_ANGLE;
    let radius = sqrt((f32(rayIndex) + 0.5) / f32(CORTAO_RAY_COUNT));
    return vec3<f32>(cos(angle) * radius, sin(angle) * radius, sqrt(1.0 - radius * radius));
}

fn cortao_unprojectWithOffset(unprojectParameters: vec4<f32>, texCoords: vec2<f32>, linearDepth: f32, offset: vec3<f32>) -> vec3<f32> {
    // Lerp between the frustum corner directions.
    // Optimized by precomputing the difference between the corners, to turn it into FMA.
    let direction = fma(texCoords, unprojectParameters.xy, unprojectParameters.zw);

    if (CORTAO_PERSPECTIVE_PROJECTION) {
        // Calculate the view space direction as direction * depth.
        // Also add an optional offset here, as it's free to do with FMA.
        return fma(vec3<f32>(direction, -1.0), vec3<f32>(linearDepth), offset);
    } else {
        // For orthographic projections, the "direction" is really the starting XY coordinates,
        // and only Z is affected by depth.
        return vec3<f32>(direction, -linearDepth) + offset;
    }
}

fn cortao_unproject(unprojectParameters: vec4<f32>, texCoords: vec2<f32>, linearDepth: f32) -> vec3<f32> {
    return cortao_unprojectWithOffset(unprojectParameters, texCoords, linearDepth, vec3<f32>(0.0));
}

fn cortao_computeRayOcclusion(
    offsetDither: f32,
    offsetTransform: mat3x3<f32>,
    tangentSpaceTransform: mat3x3<f32>,
    mipBias: f32,
    viewPosition: vec3<f32>,
    normal: vec3<f32>,
    viewDirection: vec3<f32>,
    maxRadius: f32,
    params: CORTAOMainPassParameters,
) -> u32 {

    var rayBits = 0u;

    for (var i = 0u; i < CORTAO_SAMPLE_COUNT; i++) {

        // Calculate the golden angle spiral direction
        let a = f32(i) * CORTAO_GOLDEN_ANGLE;
        let spiral = vec2<f32>(cos(a), sin(a));

        let inverseSampleCount = 1.0 / f32(CORTAO_SAMPLE_COUNT);

        // The skew is how much we suck the sample distribution into the center.
        let skew = 3.5;

        // Calculate an inner and outer radius for this sample. Dithering is used to choose a
        // radius between these two values.
        let innerRadius = pow((f32(i) + 0.0) * inverseSampleCount, skew * 0.5);
        let outerRadius = pow((f32(i) + 1.0) * inverseSampleCount, skew * 0.5);

        // Precompute the inner and outer spiral positions. The dithered position will lie
        // somewhere on a lerp between the two.
        let innerSpiral = spiral * innerRadius;
        let outerSpiral = spiral * outerRadius;

        // Calculate the area of the sampling circle that the sample is randomly moved over,
        // based on both a random rotation and a random radius dither to form a circular strip.
        // The weight is square-rooted to account for the fact that the weight is applied to a
        // radius, rather than an area.
        let sampleArea = outerRadius * outerRadius - innerRadius * innerRadius;
        let sampleWeight = sqrt(sampleArea);

        // The mip level depends on how far away we are from the center.
        let sampleMipmap = log2(outerRadius);

        // Optimized lerp between the inner and outer sample positions based on a random dither value.
        let sampleOffset = fma(outerSpiral - innerSpiral, vec2<f32>(offsetDither), innerSpiral);

        // Apply the sample transformation matrix and do a perspective division.
        let projectedCoords = offsetTransform * vec3<f32>(sampleOffset, 1.0);
        var sampleCoords = projectedCoords.xy;
        if (CORTAO_PERSPECTIVE_PROJECTION) {
            sampleCoords /= projectedCoords.z;
        }

        // Add the precomputed mip bias and the per-sample mip level together to get our mip
        // level, based on logarithmic identities.
        let mipmap = mipBias + sampleMipmap;

        // Sample the center pixel, and unpack depth only for now.
        let packedData = cortao_samplePackedData(sampleCoords, mipmap);
        let sampleDepth = cortao_unpackDepth(packedData);

        // Calculate this sample's view space position relative to the center pixel.
        let dPos = cortao_unprojectWithOffset(params.unprojectParams, sampleCoords, sampleDepth, -viewPosition);

        // Do an early out if the sample ends up outside the hemisphere (radius and normal check).
        // Skybox samples and samples outside the screen have a linear depth of 0.0, which is
        // always outside the hemisphere.
        let lengthSqrd = dot(dPos, dPos);
        if (lengthSqrd < maxRadius * maxRadius && dot(dPos, normal) > 0.0) {

            // Calculate the occluder direction.
            let inverseLength = inverseSqrt(lengthSqrd);
            var direction = dPos * inverseLength;

            // Unpack the sample normal from the packed data we already read.
            let sampleNormal = cortao_decodeNormal(packedData);

            // Calculate the slope compensation weight and use it to calculate the normalized
            // visual occluder radius.
            let slopeCompensation = inverseSqrt(max(dot(sampleNormal, viewDirection), params.inverseMaxSlopeCompensation));
            var occluderRadius = (maxRadius * sampleWeight) * (slopeCompensation * inverseLength);

            // Heuristically check for backface occluders.
            let sampleDot = dot(sampleNormal, direction);
            if (sampleDot < params.backfaceThreshold) {
                // Not a backface occluder. Apply a normal offset to prevent self-occlusion.
                // This moves the occluder circle by 1 radius into the occluder object. As this is
                // done after occluder normalization, re-normalize the direction after we apply the
                // offset to get the new direction, but keep the occluder radius the same.
                let adjustedOffset = -occluderRadius / max(1.0 - occluderRadius * occluderRadius * 0.5, 0.000001);
                direction = normalize(fma(sampleNormal, vec3<f32>(adjustedOffset), direction));
            } else {
                // This is a backface occluder. Don't do the normal offset, and optionally boost
                // the slope compensation to avoid light leaking.
                occluderRadius *= mix(1.0, slopeCompensation, params.backfaceBoost);
            }

            // Transform the occluder direction into the ray tracing tangent space.
            let hemisphereDirection = direction * tangentSpaceTransform;

            var occluderRayBits: u32;
            if (CORTAO_OCCLUDER_LOOKUP_TABLE) {
                // Encode the normal as a spheremap projection, which is used as the lookup
                // coordinates together with the occluder radius.
                let m = inverseSqrt(fma(hemisphereDirection.z, 8.0, 8.0));
                let sphereMapCoordinates = fma(hemisphereDirection.xy, vec2<f32>(m), vec2<f32>(0.5));
                let lookupCoords = vec3<f32>(sphereMapCoordinates, occluderRadius * sqrt(0.5));

                // Sphere map projection fails if the tangent space direction is (0, 0, -1), so
                // ignore the invalid sample we do in that case.
                occluderRayBits = cortao_sampleOccluderLookupTable(lookupCoords) & select(0u, 0xFFFFFFFFu, hemisphereDirection.z > -0.99);
            } else {
                // Calculate the angular size of the occluder. This wraps the occluder around the
                // hemisphere, matching the occluded area of the hemisphere with the area of the
                // circle linearly, to improve the scalability of CORTAO.
                let occluderCosAngle = max(0.0, 1.0 - occluderRadius * occluderRadius * 0.5);

                // Manually check each ray against the occluder's angular size.
                occluderRayBits = 0u;
                for (var j = 0u; j < CORTAO_RAY_COUNT; j++) {
                    let rayDirection = cortao_rayDirection(j);
                    if (dot(rayDirection, hemisphereDirection) > occluderCosAngle) {
                        occluderRayBits |= 1u << j;
                    }
                }
            }

            rayBits |= occluderRayBits;
        }
    }

    return rayBits;
}

struct CORTAOBasis {
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
}

// Frisvad's arbitrary orthonormal tangent basis construction.
fn cortao_constructBasis(normal: vec3<f32>) -> CORTAOBasis {
    let a = 1.0 / (1.0 + normal.z);
    let b = normal.x * normal.y * a;
    return CORTAOBasis(
        vec3<f32>(fma(normal.x * normal.x, a, -1.0), b, normal.x),
        vec3<f32>(b, fma(normal.y * normal.y, a, -1.0), normal.y),
    );
}

fn cortao_resolveOcclusion(rayBits: u32) -> f32 {
    // Calculate occlusion as 1.0 - occluded rays / total rays
    let numBits = countOneBits(rayBits);
    return fma(f32(numBits), -1.0 / f32(CORTAO_RAY_COUNT), 1.0);
}

fn cortao_resolveBentNormal(rayBits: u32) -> vec3<f32> {
    if (CORTAO_BENT_NORMAL_LOOKUP_TABLE) {
        return cortao_loadBentNormalLookupTable(rayBits);
    }
    // Calculate the bent normal as the sum of all unoccluded rays.
    var bentNormal = vec3<f32>(0.0);
    for (var i = 0u; i < CORTAO_RAY_COUNT; i++) {
        let rayDirection = cortao_rayDirection(i);
        bentNormal += rayDirection * select(0.0, 1.0, (rayBits & (1u << i)) == 0u);
    }
    return normalize(bentNormal); // Normalize the bent normal in the end.
}

fn cortao_mainPass(pixelCoords: vec2<u32>, params: CORTAOMainPassParameters) {

    // check if our pixel is out-of-bounds and early out in that case
    let resolution = cortao_getResolution(params.packedResolution);
    if (pixelCoords.x >= resolution.x || pixelCoords.y >= resolution.y) {
        return;
    }

    let texCoords = (vec2<f32>(pixelCoords) + 0.5) / vec2<f32>(resolution);

    let linearDepth = cortao_unpackDepth(cortao_samplePackedData(texCoords, 0.0));
    if (linearDepth == 0.0) {
        // This is a skybox pixel, ignore it.
        return;
    }

    let normal = cortao_loadNormal(vec2<i32>(pixelCoords));

    var viewPosition = cortao_unproject(params.unprojectParams, texCoords, linearDepth);

    // Setup a set of random variables used during the setup code
    let random = cortao_hash2(vec2<u32>(cortao_seed(pixelCoords)) + cortao_unpack16(params.randomSeed));

    let randomSampleRotation = f32(random.x) * (CORTAO_PI2 / f32(0xFFFFFFFFu));
    let offsetDither = f32(random.y) * (1.0 / f32(0xFFFFFFFFu));
    let randomTangentRotation = f32(random.x ^ random.y ^ params.randomSeed) * (CORTAO_PI2 / f32(0xFFFFFFFFu));
    let sampleRotation = vec2<f32>(sin(randomSampleRotation), cos(randomSampleRotation));

    var clampedRadius: f32;
    var viewDirection: vec3<f32>;
    var baseMipBias: f32;
    var offsetTransform: mat3x3<f32>;

    if (CORTAO_PERSPECTIVE_PROJECTION) {

        // To enforce a maximum screenspace radius, we precompute the maximum radius as a linear
        // function of depth.
        clampedRadius = min(params.radius, linearDepth * params.maxApparentCircleRadiusCoefficient);

        // As we get closer to the sampling sphere, it starts to appear as a larger and larger
        // surface due to the horizon. This code calculates the required expanded circle radius
        // to sample the entire sphere.
        let inverseCircleDistance = inverseSqrt(linearDepth * linearDepth - clampedRadius * clampedRadius);
        let apparentCircleRadius = clampedRadius * inverseCircleDistance;
        let circleRadius = apparentCircleRadius * linearDepth;

        viewDirection = -normalize(viewPosition);

        // Precompute the constant portion of the mip bias. Due to our sample projection, the
        // sample pattern becomes stretched closer to the edge of the screen. Increase the mip
        // bias to compensate.
        let stretchMultiplier = inverseSqrt(viewDirection.z);
        baseMipBias = log2(apparentCircleRadius * stretchMultiplier) + params.mipBias;

        // Push the center pixel's position out slightly in the direction of the normal to avoid
        // self-shadowing.
        let pixelSize = params.normalBias * linearDepth;
        viewPosition = fma(normal, vec3<f32>(pixelSize), viewPosition);

        // Build a 2D sample rotation matrix. Instead of scaling the circle tangents later, we
        // bake the radius scaling in here, as it's cheaper to precompute.
        let scaledRotation = sampleRotation * circleRadius;
        let rotationScaleMatrix = mat3x3<f32>(
            scaledRotation.x, -scaledRotation.y, 0.0,
            scaledRotation.y, scaledRotation.x, 0.0,
            0.0, 0.0, 1.0,
        );

        // Construct the 3D sampling circle's tangents.
        let basis = cortao_constructBasis(viewDirection);
        var right = basis.tangent;
        let up = basis.bitangent;

        // Randomly flip the right vector. This randomly switches between a clockwise and
        // counter-clockwise sample pattern spiral, which cancels out some errors.
        if ((random.x & 1u) == 0u) {
            right = -right;
        }

        // Construct a 3x3 matrix that takes 2D sample positions and transforms them onto the 3D circle.
        let circleMatrix = mat3x3<f32>(right, up, viewPosition);

        // A simplified, truncated projection matrix. Only supports FOV and off-center
        // projection; nothing else. The NDC --> texture space transformation is baked into
        // these constants on the CPU side.
        let projectionMatrix = mat3x3<f32>(
            params.projectionParams.x, 0.0, 0.0,
            0.0, params.projectionParams.y, 0.0,
            params.projectionParams.z, params.projectionParams.w, -1.0,
        );

        // Multiply the 3 matrices together
        offsetTransform = projectionMatrix * (circleMatrix * rotationScaleMatrix);

    } else {

        // In ortho projections, many parameters end up as constants.
        clampedRadius = params.radius;
        viewDirection = vec3<f32>(0.0, 0.0, 1.0);
        baseMipBias = params.mipBias;

        // Push the center pixel's position out slightly in the direction of the normal to avoid
        // self-shadowing.
        let pixelSize = params.normalBias;
        viewPosition = fma(normal, vec3<f32>(pixelSize), viewPosition);

        // Build the simplified orthographic sample transformation. This is really just a 3x2
        // matrix, but use a 3x3 matrix to match the perspective codepath.
        let rotationScaleMatrix = mat3x3<f32>(
            sampleRotation.x, -sampleRotation.y, 0.0,
            sampleRotation.y, sampleRotation.x, 0.0,
            0.0, 0.0, 1.0,
        );

        let projectionMatrix = mat3x3<f32>(
            clampedRadius * params.projectionParams.x, 0.0, 0.0,
            0.0, clampedRadius * params.projectionParams.y, 0.0,
            texCoords.x, texCoords.y, 1.0,
        );

        offsetTransform = projectionMatrix * rotationScaleMatrix;
    }

    // Generate a tangent basis for the ray directions
    let tangentBasis = cortao_constructBasis(normal);
    let t = tangentBasis.tangent;
    let bt = tangentBasis.bitangent;

    // Randomly rotate the tangent and bitangent around the normal
    let s = sin(randomTangentRotation);
    let c = cos(randomTangentRotation);
    let tangent = c * t + s * bt;
    let bitangent = c * bt - s * t;

    let tangentSpaceTransform = mat3x3<f32>(tangent, bitangent, normal);

    let rayBits = cortao_computeRayOcclusion(offsetDither, offsetTransform, tangentSpaceTransform, baseMipBias, viewPosition, normal, viewDirection, clampedRadius, params);

    let occlusion = cortao_resolveOcclusion(rayBits);
    // Transform tangent space bent normal back to view space.
    let bentNormal = tangentSpaceTransform * cortao_resolveBentNormal(rayBits);

    cortao_writeResult(vec2<i32>(pixelCoords), occlusion, bentNormal, tangentSpaceTransform, rayBits);
}
