//! GGUF quant block layouts used by DwarfStar 4.
//!
//! Ported from the `block_*` typedefs in `ds4.c` (and originally from
//! `ggml-quants.h`). Only the formats DS4 actually reads are kept:
//!
//! * `Q2_K` — routed down experts
//! * `Q4_K` — routed experts in the high-memory variant
//! * `IQ2_XXS` — routed gate/up experts
//! * `Q8_K` — temporary activation blocks used during dot products
//!
//! Layouts mirror the C structs byte-for-byte. We use `bytemuck` so casting
//! mmap slices into typed blocks is a single safe call instead of a manual
//! pointer dance.

use bytemuck::{Pod, Zeroable};

pub const QK_K: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct BlockQ2K {
    pub scales: [u8; QK_K / 16],
    pub qs:     [u8; QK_K / 4],
    pub d:      u16,    // half-float, scale
    pub dmin:   u16,    // half-float, scale-min
}
const _: () = assert!(std::mem::size_of::<BlockQ2K>() == 84);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct BlockQ4K {
    pub d:      u16,
    pub dmin:   u16,
    pub scales: [u8; 12],
    pub qs:     [u8; QK_K / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4K>() == 144);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct BlockQ8K {
    pub d:     f32,
    pub qs:    [i8; QK_K],
    pub bsums: [i16; QK_K / 16],
}
const _: () = assert!(std::mem::size_of::<BlockQ8K>() == 292);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct BlockIQ2XXS {
    pub d:  u16,
    pub qs: [u16; QK_K / 8],
}
const _: () = assert!(std::mem::size_of::<BlockIQ2XXS>() == 66);

/// Reinterpret a `[u8]` mmap slice as a slice of fixed-size quant blocks.
/// Mirrors the `(const block_q2_K*)data` cast on the C side.
pub fn blocks<T: Pod>(bytes: &[u8]) -> &[T] {
    bytemuck::cast_slice(bytes)
}
