//! The two lookup tables, generated rather than loaded: they come out bit-identical to the
//! original's pre-generated files (`tables/` in the original repository; see the tests).

use std::f64::consts::PI;

/// The occluder lookup table's recommended size for up to 32 samples per pixel:
/// 24 x 24 directions by 16 radii (the original's `24x24x16` tables).
pub const OCCLUDER_DIRECTION_RESOLUTION: u32 = 24;
pub const OCCLUDER_RADIUS_RESOLUTION: u32 = 16;

fn ray_directions(ray_count: u32) -> Vec<[f64; 3]> {
    let golden_angle = PI * (3.0 - 5.0f64.sqrt());
    (0..ray_count)
        .map(|i| {
            let angle = i as f64 * golden_angle;
            let radius = ((i as f64 + 0.5) / ray_count as f64).sqrt();
            let x = angle.cos() * radius;
            let y = angle.sin() * radius;
            [x, y, (1.0 - x * x - y * y).sqrt()]
        })
        .collect()
}

fn decode_sphere_map(x: f64, y: f64) -> [f64; 3] {
    let ex = x * 2.0 - 1.0;
    let ey = y * 2.0 - 1.0;
    let z = (1.0 - ex * ex - ey * ey).max(0.0);
    let mul = 2.0 * z.sqrt();
    [ex * mul, ey * mul, z * 2.0 - 1.0]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The occluder lookup table for `ray_count` rays (8 to 32): one ray mask per texel of a
/// `direction_resolution` x `direction_resolution` x `radius_resolution` 3D texture, X fastest,
/// then Y, then Z (the original `OccluderLookupTableGenerator.py`).
pub fn occluder_lookup_table(direction_resolution: u32, radius_resolution: u32, ray_count: u32) -> Vec<u32> {
    assert!((8..=32).contains(&ray_count), "CORTAO traces 8 to 32 rays");
    assert!(direction_resolution > 0 && radius_resolution > 0);
    let rays = ray_directions(ray_count);
    let max_occluder_radius = 2.0f64.sqrt();
    let plane = (direction_resolution * direction_resolution) as usize;
    let mut output = Vec::with_capacity(plane * radius_resolution as usize);

    for z in 0..radius_resolution {
        let occluder_radius = max_occluder_radius * (z as f64 + 0.5) / radius_resolution as f64;
        let occluder_cos_angle = 1.0 - occluder_radius * occluder_radius * 0.5;

        // For each ray, the texel whose direction is closest to it, so every ray is covered by
        // at least one texel even when the occluder is smaller than a texel.
        let mut closest = vec![(0usize, -1.0f64); ray_count as usize];
        let offset = output.len();
        for y in 0..direction_resolution {
            for x in 0..direction_resolution {
                let direction = decode_sphere_map(
                    (x as f64 + 0.5) / direction_resolution as f64,
                    (y as f64 + 0.5) / direction_resolution as f64,
                );
                let mut bits = 0u32;
                if direction[2] != -1.0 {
                    for (i, ray) in rays.iter().enumerate() {
                        let cos_angle = dot(direction, *ray);
                        if cos_angle > closest[i].1 {
                            closest[i] = ((y * direction_resolution + x) as usize, cos_angle);
                        }
                        if cos_angle > occluder_cos_angle {
                            bits |= 1 << i;
                        }
                    }
                }
                output.push(bits);
            }
        }
        for (i, &(index, _)) in closest.iter().enumerate() {
            output[offset + index] |= 1 << i;
        }
    }
    output
}

/// The occluder lookup table's texel format for `ray_count` rays: `R8Uint` up to 8 rays,
/// `R16Uint` up to 16, `R32Uint` beyond, as bytes per texel.
pub fn occluder_bytes_per_texel(ray_count: u32) -> usize {
    match ray_count {
        0..=8 => 1,
        9..=16 => 2,
        _ => 4,
    }
}

/// [`occluder_lookup_table`] as little-endian texel data for its format
/// ([`occluder_bytes_per_texel`]), ready for a 3D texture upload.
pub fn occluder_lookup_table_bytes(direction_resolution: u32, radius_resolution: u32, ray_count: u32) -> Vec<u8> {
    let table = occluder_lookup_table(direction_resolution, radius_resolution, ray_count);
    let width = occluder_bytes_per_texel(ray_count);
    table.iter().flat_map(|bits| bits.to_le_bytes().into_iter().take(width)).collect()
}

/// The bent normal lookup table for `ray_count` rays (8 or 16): for every mask of occluded rays,
/// the tangent-space direction of the unoccluded rays' sum, X and Y as signed bytes (`/ 127`),
/// Z to be rebuilt as `sqrt(max(0, 1 - x² - y²))`. A mask with every ray occluded gives (0, 0).
pub fn bent_normal_lookup_table(ray_count: u32) -> Vec<[i8; 2]> {
    assert!(ray_count == 8 || ray_count == 16, "the bent normal lookup table exists for 8 or 16 rays");
    let rays = ray_directions(ray_count);
    (0..1u32 << ray_count)
        .map(|occluded| {
            let mut sum = [0.0f64; 3];
            for (i, ray) in rays.iter().enumerate() {
                if occluded & (1 << i) == 0 {
                    for k in 0..3 {
                        sum[k] += ray[k];
                    }
                }
            }
            let length = dot(sum, sum).sqrt();
            if length == 0.0 {
                return [0, 0];
            }
            let quantise = |v: f64| (v / length * 127.0 + 0.5).floor() as i8;
            [quantise(sum[0]), quantise(sum[1])]
        })
        .collect()
}

/// [`bent_normal_lookup_table`] as one `u32` per entry, `pack4x8snorm(x, y, 0, 0)`: the layout
/// the reference main pass reads from a storage buffer.
pub fn bent_normal_lookup_table_words(ray_count: u32) -> Vec<u32> {
    bent_normal_lookup_table(ray_count)
        .into_iter()
        .map(|[x, y]| u32::from(x as u8) | (u32::from(y as u8) << 8))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn original(name: &str) -> Vec<u8> {
        std::fs::read(format!("{}/reference/original/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
    }

    #[test]
    fn occluder_tables_match_the_original_files() {
        for (rays, file) in [(8, "24x24x16x8_R8_UINT.bin"), (16, "24x24x16x16_R16_UINT.bin"), (32, "24x24x16x32_R32_UINT.bin")] {
            assert_eq!(occluder_lookup_table_bytes(24, 16, rays), original(file), "{rays} rays");
        }
    }

    #[test]
    fn eight_ray_bent_normal_table_matches_the_original() {
        let header = String::from_utf8(original("CORTAOBentNormalLookupTable8Rays.h")).unwrap();
        let body = &header[header.find("= {").unwrap() + 3..];
        let body = &body[..body.find('}').unwrap()];
        let expected: Vec<i8> = body.split(',').filter_map(|v| v.trim().parse().ok()).collect();
        let ours: Vec<i8> = bent_normal_lookup_table(8).into_iter().flatten().collect();
        assert_eq!(ours, expected);
    }

    #[test]
    fn sixteen_ray_bent_normal_table_has_every_mask() {
        let table = bent_normal_lookup_table(16);
        assert_eq!(table.len(), 65536);
        assert_eq!(table[0xFFFF], [0, 0]);
    }
}

