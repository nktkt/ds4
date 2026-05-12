//! GGUF mmap loader.
//!
//! Ported from the GGUF reading section of `ds4.c` (look for `gguf_open`,
//! `gguf_read_header`, `gguf_get_tensor`). We mmap the file so the OS page
//! cache absorbs the cost of skipping unused tensors. Tensor data stays
//! borrowed from the mmap region; the model layer wraps slices into typed
//! views (f16, q2_k, iq2_xxs, ...).

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

const GGUF_MAGIC: u32 = u32::from_le_bytes(*b"GGUF");
const GGUF_VERSION_MIN: u32 = 2;
const GGUF_VERSION_MAX: u32 = 3;

#[derive(Debug)]
pub struct Gguf {
    pub mmap: Mmap,
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, Tensor>,
    pub data_offset: u64,
    pub alignment: u64,
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32),
    U64(u64), I64(i64), F32(f32), F64(f64), Bool(bool),
    String(String),
    Array(Vec<Value>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum DType {
    F32   = 0,
    F16   = 1,
    Q4_0  = 2,
    Q4_1  = 3,
    Q5_0  = 6,
    Q5_1  = 7,
    Q8_0  = 8,
    Q8_1  = 9,
    Q2_K  = 10,
    Q3_K  = 11,
    Q4_K  = 12,
    Q5_K  = 13,
    Q6_K  = 14,
    Q8_K  = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8    = 24,
    I16   = 25,
    I32   = 26,
    I64   = 27,
    F64   = 28,
    IQ1_M = 29,
    BF16  = 30,
}

impl DType {
    pub fn from_raw(v: u32) -> Result<DType> {
        // Manual tableized conversion. Avoids unsafe transmute.
        Ok(match v {
            0 => Self::F32, 1 => Self::F16, 2 => Self::Q4_0, 3 => Self::Q4_1,
            6 => Self::Q5_0, 7 => Self::Q5_1, 8 => Self::Q8_0, 9 => Self::Q8_1,
            10 => Self::Q2_K, 11 => Self::Q3_K, 12 => Self::Q4_K, 13 => Self::Q5_K,
            14 => Self::Q6_K, 15 => Self::Q8_K, 16 => Self::IQ2_XXS, 17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS, 19 => Self::IQ1_S, 20 => Self::IQ4_NL, 21 => Self::IQ3_S,
            22 => Self::IQ2_S, 23 => Self::IQ4_XS, 24 => Self::I8, 25 => Self::I16,
            26 => Self::I32, 27 => Self::I64, 28 => Self::F64, 29 => Self::IQ1_M,
            30 => Self::BF16,
            other => bail!("gguf: unknown dtype {other}"),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub size_bytes: u64,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("gguf: open {}", path.display()))?;
        // SAFETY: We rely on the OS to fault if the file is truncated.
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self> {
        let buf = &mmap[..];
        let mut r = Reader::new(buf);
        let magic = r.u32()?;
        if magic != GGUF_MAGIC {
            bail!("gguf: bad magic {:#x}", magic);
        }
        let version = r.u32()?;
        if !(GGUF_VERSION_MIN..=GGUF_VERSION_MAX).contains(&version) {
            bail!("gguf: unsupported version {version}");
        }
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;
        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let value = r.value()?;
            metadata.insert(key, value);
        }
        let alignment = match metadata.get("general.alignment") {
            Some(Value::U32(v)) => *v as u64,
            Some(Value::I32(v)) => *v as u64,
            _ => 32,
        };
        let mut tensor_descs: Vec<(String, DType, Vec<u64>, u64)> = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(r.u64()?);
            }
            let dtype = DType::from_raw(r.u32()?)?;
            let offset = r.u64()?;
            tensor_descs.push((name, dtype, shape, offset));
        }
        let data_offset_unaligned = r.pos as u64;
        let pad = (alignment - (data_offset_unaligned % alignment)) % alignment;
        let data_offset = data_offset_unaligned + pad;
        let mut tensors = BTreeMap::new();
        for (name, dtype, shape, off) in tensor_descs {
            let size = tensor_size_bytes(dtype, &shape)?;
            tensors.insert(
                name.clone(),
                Tensor { name, dtype, shape, offset: data_offset + off, size_bytes: size },
            );
        }
        Ok(Gguf { mmap, version, metadata, tensors, data_offset, alignment })
    }

    pub fn tensor_bytes(&self, t: &Tensor) -> Result<&[u8]> {
        let start = t.offset as usize;
        let end = start
            .checked_add(t.size_bytes as usize)
            .ok_or_else(|| anyhow!("gguf: tensor size overflow"))?;
        self.mmap.get(start..end)
            .ok_or_else(|| anyhow!("gguf: tensor `{}` extends past mmap", t.name))
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| match v {
            Value::U32(x) => Some(*x), Value::I32(x) => Some(*x as u32),
            Value::U64(x) => Some(*x as u32), Value::I64(x) => Some(*x as u32),
            _ => None,
        })
    }
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| match v {
            Value::F32(x) => Some(*x), Value::F64(x) => Some(*x as f32),
            _ => None,
        })
    }
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| match v {
            Value::String(s) => Some(s.as_str()), _ => None,
        })
    }
}

fn tensor_size_bytes(dtype: DType, shape: &[u64]) -> Result<u64> {
    let n: u64 = shape.iter().product();
    // (block size, bytes per block). For non-block formats block size = 1.
    let (block, bpb): (u64, u64) = match dtype {
        DType::F32 => (1, 4),
        DType::F16 => (1, 2),
        DType::BF16 => (1, 2),
        DType::F64 => (1, 8),
        DType::I8  => (1, 1),
        DType::I16 => (1, 2),
        DType::I32 => (1, 4),
        DType::I64 => (1, 8),
        DType::Q4_0 => (32, 18),
        DType::Q4_1 => (32, 20),
        DType::Q5_0 => (32, 22),
        DType::Q5_1 => (32, 24),
        DType::Q8_0 => (32, 34),
        DType::Q8_1 => (32, 36),
        DType::Q2_K => (256, 84),
        DType::Q3_K => (256, 110),
        DType::Q4_K => (256, 144),
        DType::Q5_K => (256, 176),
        DType::Q6_K => (256, 210),
        DType::Q8_K => (256, 292),
        DType::IQ2_XXS => (256, 66),
        DType::IQ2_XS  => (256, 74),
        DType::IQ3_XXS => (256, 98),
        DType::IQ1_S   => (256, 50),
        DType::IQ4_NL  => (32, 18),
        DType::IQ3_S   => (256, 110),
        DType::IQ2_S   => (256, 82),
        DType::IQ4_XS  => (256, 136),
        DType::IQ1_M   => (256, 56),
    };
    if n % block != 0 {
        bail!("gguf: element count {n} not divisible by block {block} for dtype {:?}", dtype);
    }
    Ok((n / block) * bpb)
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self { Self { buf, pos: 0 } }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| anyhow!("gguf: overflow"))?;
        let s = self.buf.get(self.pos..end).ok_or_else(|| anyhow!("gguf: short read"))?;
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self)  -> Result<u8>  { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> Result<u16> { Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap())) }
    fn u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn i8(&mut self)  -> Result<i8>  { Ok(self.u8()? as i8) }
    fn i16(&mut self) -> Result<i16> { Ok(self.u16()? as i16) }
    fn i32(&mut self) -> Result<i32> { Ok(self.u32()? as i32) }
    fn i64(&mut self) -> Result<i64> { Ok(self.u64()? as i64) }
    fn f32(&mut self) -> Result<f32> { Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn f64(&mut self) -> Result<f64> { Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn bool_(&mut self) -> Result<bool> { Ok(self.u8()? != 0) }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        Ok(std::str::from_utf8(bytes)
            .map_err(|e| anyhow!("gguf: non-utf8 string ({e})"))?
            .to_owned())
    }
    fn value(&mut self) -> Result<Value> {
        let kind = self.u32()?;
        Self::value_kind(self, kind)
    }
    fn value_kind(&mut self, kind: u32) -> Result<Value> {
        Ok(match kind {
            0 => Value::U8(self.u8()?), 1 => Value::I8(self.i8()?),
            2 => Value::U16(self.u16()?), 3 => Value::I16(self.i16()?),
            4 => Value::U32(self.u32()?), 5 => Value::I32(self.i32()?),
            6 => Value::F32(self.f32()?), 7 => Value::Bool(self.bool_()?),
            8 => Value::String(self.string()?),
            9 => {
                let elem_kind = self.u32()?;
                let n = self.u64()? as usize;
                let mut out = Vec::with_capacity(n);
                for _ in 0..n { out.push(self.value_kind(elem_kind)?); }
                Value::Array(out)
            }
            10 => Value::U64(self.u64()?), 11 => Value::I64(self.i64()?),
            12 => Value::F64(self.f64()?),
            other => bail!("gguf: unknown value kind {other}"),
        })
    }
}
