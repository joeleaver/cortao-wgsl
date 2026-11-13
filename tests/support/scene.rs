//! A small analytic scene traced on the CPU into a depth buffer and a view-space normal buffer:
//! a tilted ground, three spheres and a wall, with open sky above. Everything is in view
//! space (camera at the origin, looking down -Z).

use glam::{Mat4, Vec3, Vec4};

pub struct Scene {
    pub size: (u32, u32),
    pub projection: Mat4,
    /// The planes as CORTAO's configuration takes them: for the reversed infinite perspective,
    /// `near` is infinite and `far` is the camera's near distance.
    pub near: f32,
    pub far: f32,
    pub perspective: bool,
    /// One depth per pixel (row-major).
    pub depth: Vec<f32>,
    /// One view-space normal per pixel, as rgba (w = 0).
    pub normals: Vec<[f32; 4]>,
}

const SKY: f32 = 60.0;

fn hit_sphere(origin: Vec3, dir: Vec3, centre: Vec3, radius: f32) -> Option<(f32, Vec3)> {
    let oc = origin - centre;
    let b = oc.dot(dir);
    let c = oc.length_squared() - radius * radius;
    let h = b * b - c;
    if h < 0.0 {
        return None;
    }
    let t = -b - h.sqrt();
    (t > 0.0).then(|| (t, (origin + dir * t - centre) / radius))
}

fn hit_plane(origin: Vec3, dir: Vec3, point: Vec3, normal: Vec3) -> Option<(f32, Vec3)> {
    let denom = dir.dot(normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (point - origin).dot(normal) / denom;
    (t > 0.0).then_some((t, if denom < 0.0 { normal } else { -normal }))
}

fn hit_box(origin: Vec3, dir: Vec3, min: Vec3, max: Vec3) -> Option<(f32, Vec3)> {
    let inv = dir.recip();
    let t0 = (min - origin) * inv;
    let t1 = (max - origin) * inv;
    let near = t0.min(t1);
    let far = t0.max(t1);
    let enter = near.max_element();
    let exit = far.min_element();
    if enter > exit || exit < 0.0 || enter <= 0.0 {
        return None;
    }
    let normal = if enter == near.x {
        Vec3::new(-dir.x.signum(), 0.0, 0.0)
    } else if enter == near.y {
        Vec3::new(0.0, -dir.y.signum(), 0.0)
    } else {
        Vec3::new(0.0, 0.0, -dir.z.signum())
    };
    Some((enter, normal))
}

fn trace(origin: Vec3, dir: Vec3) -> Option<(Vec3, Vec3)> {
    let ground_normal = Vec3::new(0.0, 0.97, 0.24).normalize();
    let hits = [
        hit_plane(origin, dir, Vec3::new(0.0, -1.0, 0.0), ground_normal),
        hit_sphere(origin, dir, Vec3::new(0.0, -0.45, -4.0), 0.6),
        hit_sphere(origin, dir, Vec3::new(1.2, -0.75, -3.0), 0.3),
        hit_sphere(origin, dir, Vec3::new(-1.0, -0.6, -5.5), 0.5),
        hit_box(origin, dir, Vec3::new(-3.0, -1.5, -9.0), Vec3::new(-2.2, 1.5, -3.0)),
    ];
    hits.into_iter()
        .flatten()
        .filter(|(t, _)| *t < SKY)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(t, normal)| (origin + dir * t, normal))
}

impl Scene {
    /// The scene through a slightly off-centre, reversed, infinite-far perspective projection.
    pub fn perspective(size: (u32, u32)) -> Self {
        let near = 0.1;
        let mut projection = Mat4::perspective_infinite_reverse_rh(1.0, size.0 as f32 / size.1 as f32, near);
        // An off-centre projection, which CORTAO supports.
        projection.z_axis.x = 0.03;
        projection.z_axis.y = -0.02;
        Self::render(size, projection, f32::INFINITY, near, true)
    }

    /// The scene through an orthographic projection with standard depth.
    pub fn orthographic(size: (u32, u32)) -> Self {
        let aspect = size.0 as f32 / size.1 as f32;
        let projection = Mat4::orthographic_rh(-2.2 * aspect, 2.2 * aspect, -2.0, 2.4, 0.5, 20.0);
        Self::render(size, projection, 0.5, 20.0, false)
    }

    fn render(size: (u32, u32), projection: Mat4, near: f32, far: f32, perspective: bool) -> Self {
        let inverse = projection.inverse();
        let mut depth = Vec::new();
        let mut normals = Vec::new();
        for y in 0..size.1 {
            for x in 0..size.0 {
                let ndc_x = (x as f32 + 0.5) / size.0 as f32 * 2.0 - 1.0;
                let ndc_y = 1.0 - (y as f32 + 0.5) / size.1 as f32 * 2.0;
                let unproject = |z: f32| {
                    let p = inverse * Vec4::new(ndc_x, ndc_y, z, 1.0);
                    p.truncate() / p.w
                };
                let (origin, dir) = if perspective {
                    (Vec3::ZERO, unproject(1.0).normalize())
                } else {
                    // Rays start at the near plane, a little behind it so it is not missed.
                    let a = unproject(0.0);
                    (a - Vec3::Z, Vec3::NEG_Z)
                };
                match trace(origin, dir) {
                    Some((point, normal)) => {
                        let clip = projection * point.extend(1.0);
                        depth.push(clip.z / clip.w);
                        normals.push([normal.x, normal.y, normal.z, 0.0]);
                    }
                    None => {
                        // The cleared depth: 0 for reversed depth, 1 for standard.
                        depth.push(if perspective { 0.0 } else { 1.0 });
                        normals.push([0.0, 0.0, 1.0, 0.0]);
                    }
                }
            }
        }
        Self { size, projection, near, far, perspective, depth, normals }
    }
}
