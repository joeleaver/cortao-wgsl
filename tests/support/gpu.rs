//! A Vulkan wgpu device for running the original GLSL (as SPIR-V, passed straight through) and
//! the WGSL port side by side, with the few resource helpers the comparison needs.

use std::borrow::Cow;
use std::path::Path;
use std::process::Command;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// One binding of an explicit bind group layout, for the SPIR-V pipelines.
#[derive(Clone, Copy)]
pub enum Slot {
    Uniform,
    FloatTexture,
    UintTexture,
    UintTexture3,
    Storage(wgpu::TextureFormat),
    ReadOnlyBuffer,
    Sampler,
}

impl Gpu {
    /// A Vulkan device that can pass SPIR-V through and border-clamp, or None when this
    /// machine has none (the comparison then cannot run).
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let required = wgpu::Features::PASSTHROUGH_SHADERS | wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER;
        if !adapter.features().contains(required) {
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("cortao-wgsl comparison"),
            required_features: required,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        Some(Self { device, queue })
    }

    /// The SPIR-V `glslc` makes of the test wrapper `name` (in tests/glsl) with `defines`,
    /// including the original's headers unchanged.
    pub fn glsl(&self, name: &str, defines: &[(&str, u32)]) -> wgpu::ShaderModule {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let out = std::env::temp_dir().join(format!("cortao-wgsl-{}-{name}.spv", std::process::id()));
        let mut command = Command::new("glslc");
        command
            .args(["-fshader-stage=compute", "--target-env=vulkan1.1", "-O"])
            .arg("-I")
            .arg(root.join("reference/original"))
            .arg(root.join("tests/glsl").join(format!("{name}.comp")))
            .arg("-o")
            .arg(&out);
        for (key, value) in defines {
            command.arg(format!("-D{key}={value}"));
        }
        let status = command.status().expect("glslc runs (install shaderc)");
        assert!(status.success(), "glslc compiles {name}");
        let bytes = std::fs::read(&out).unwrap();
        let _ = std::fs::remove_file(&out);
        let words: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
        unsafe {
            self.device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(name),
                entry_points: Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                    name: Cow::Borrowed("main"),
                    workgroup_size: (8, 8, 1),
                }]),
                spirv: Some(Cow::Owned(words)),
                ..Default::default()
            })
        }
    }

    /// A compute pipeline over `module`'s `main`, with an explicit layout of `slots` in group 0.
    pub fn spirv_pipeline(&self, module: &wgpu::ShaderModule, slots: &[Slot]) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let entries: Vec<_> = slots
            .iter()
            .enumerate()
            .map(|(binding, slot)| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: match *slot {
                    Slot::Uniform => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    Slot::ReadOnlyBuffer => wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    Slot::FloatTexture => texture(wgpu::TextureSampleType::Float { filterable: false }, wgpu::TextureViewDimension::D2),
                    Slot::UintTexture => texture(wgpu::TextureSampleType::Uint, wgpu::TextureViewDimension::D2),
                    Slot::UintTexture3 => texture(wgpu::TextureSampleType::Uint, wgpu::TextureViewDimension::D3),
                    Slot::Storage(format) => wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    Slot::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                },
                count: None,
            })
            .collect();
        let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        (pipeline, layout)
    }

    /// A compute pipeline over the WGSL `source`'s `main`, its layout derived from the shader,
    /// with `constants` for its overrides.
    pub fn wgsl_pipeline(&self, source: &str, constants: &[(&str, f64)]) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions { constants, ..Default::default() },
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        (pipeline, layout)
    }

    pub fn dispatch(&self, pipeline: &wgpu::ComputePipeline, layout: &wgpu::BindGroupLayout, resources: &[(u32, wgpu::BindingResource<'_>)], groups: (u32, u32)) {
        let entries: Vec<_> = resources
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        self.queue.submit([encoder.finish()]);
    }

    pub fn uniform<T: bytemuck::Pod>(&self, value: &T) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytemuck::bytes_of(value));
        buffer
    }

    pub fn storage_buffer(&self, words: &[u32]) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (words.len().max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(words));
        buffer
    }

    pub fn texture(&self, size: (u32, u32), mips: u32, format: wgpu::TextureFormat) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    /// A 2D texture filled with `data` (tightly packed texels of `format`).
    pub fn texture_with(&self, size: (u32, u32), format: wgpu::TextureFormat, data: &[u8]) -> wgpu::Texture {
        let texture = self.texture(size, 1, format);
        let bytes_per_texel = format.block_copy_size(None).unwrap();
        self.queue.write_texture(
            texture.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size.0 * bytes_per_texel), rows_per_image: None },
            wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
        );
        texture
    }

    pub fn texture3_with(&self, size: (u32, u32, u32), format: wgpu::TextureFormat, data: &[u8]) -> wgpu::Texture {
        let extent = wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: size.2 };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bytes_per_texel = format.block_copy_size(None).unwrap();
        self.queue.write_texture(
            texture.as_image_copy(),
            data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size.0 * bytes_per_texel), rows_per_image: Some(size.1) },
            extent,
        );
        texture
    }

    /// The texels of `level` of `texture`, tightly packed, as 32-bit words (formats with 4 or
    /// 16 bytes per texel).
    pub fn read(&self, texture: &wgpu::Texture, level: u32) -> Vec<u32> {
        let width = (texture.width() >> level).max(1);
        let height = (texture.height() >> level).max(1);
        let bytes_per_texel = texture.format().block_copy_size(None).unwrap();
        let row = width * bytes_per_texel;
        let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture, mip_level: level, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: None },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        self.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        let mut out = Vec::with_capacity((row / 4 * height) as usize);
        for y in 0..height {
            let start = (y * padded) as usize;
            out.extend_from_slice(bytemuck::cast_slice(&mapped[start..start + row as usize]));
        }
        out
    }

    pub fn sampler(&self, address_mode: wgpu::AddressMode) -> wgpu::Sampler {
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: address_mode,
            address_mode_v: address_mode,
            address_mode_w: address_mode,
            border_color: (address_mode == wgpu::AddressMode::ClampToBorder).then_some(wgpu::SamplerBorderColor::TransparentBlack),
            ..Default::default()
        })
    }
}

fn texture(sample_type: wgpu::TextureSampleType, view_dimension: wgpu::TextureViewDimension) -> wgpu::BindingType {
    wgpu::BindingType::Texture { sample_type, view_dimension, multisampled: false }
}

/// A view of `level` alone.
pub fn level_view(texture: &wgpu::Texture, level: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor { base_mip_level: level, mip_level_count: Some(1), ..Default::default() })
}

pub fn view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&Default::default())
}
