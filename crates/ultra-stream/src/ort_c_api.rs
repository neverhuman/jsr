//! Pre-allocated tensor views (`OrtTensorIn`, `OrtTensorOut`), zero-allocation `OrtRunScratch`,
//! and RAII `OrtSession` wrapper over [`OrtEnv`] (`OrtApi(17)`).

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::Path;
use std::ptr::null_mut;
use std::sync::Arc;

use crate::api::StreamError;
pub use crate::ort_ffi::OrtEnv;
use crate::ort_ffi::ORT_GRAPH_OPT_ALL;
use crate::sherpa_ffi::path_to_cstring;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum OrtElemType {
    Float32 = 1,
    Int64 = 7,
}

/// Borrowed input tensor view passed to [`OrtSession::run_into`].
pub struct OrtTensorIn<'a> {
    data_addr: usize,
    byte_len: usize,
    shape: &'a [i64],
    elem: OrtElemType,
}

impl<'a> OrtTensorIn<'a> {
    pub fn f32(slice: &'a [f32], shape: &'a [i64]) -> Self {
        Self {
            data_addr: slice.as_ptr() as usize,
            byte_len: std::mem::size_of_val(slice),
            shape,
            elem: OrtElemType::Float32,
        }
    }

    pub fn i64(slice: &'a [i64], shape: &'a [i64]) -> Self {
        Self {
            data_addr: slice.as_ptr() as usize,
            byte_len: std::mem::size_of_val(slice),
            shape,
            elem: OrtElemType::Int64,
        }
    }
}

/// Pre-allocated mutable output tensor view written in place by [`OrtSession::run_into`].
pub struct OrtTensorOut<'a> {
    data_addr: usize,
    byte_len: usize,
    shape: &'a [i64],
    elem: OrtElemType,
}

impl<'a> OrtTensorOut<'a> {
    pub fn f32(slice: &'a mut [f32], shape: &'a [i64]) -> Self {
        Self {
            data_addr: slice.as_mut_ptr() as usize,
            byte_len: std::mem::size_of_val(slice),
            shape,
            elem: OrtElemType::Float32,
        }
    }

    pub fn i64(slice: &'a mut [i64], shape: &'a [i64]) -> Self {
        Self {
            data_addr: slice.as_mut_ptr() as usize,
            byte_len: std::mem::size_of_val(slice),
            shape,
            elem: OrtElemType::Int64,
        }
    }
}

/// Reusable call scratch eliminating per-invocation `Vec` allocations across encoder/decoder/joiner.
#[derive(Default)]
pub struct OrtRunScratch {
    in_name_addrs: Vec<usize>,
    in_val_addrs: Vec<usize>,
    out_name_addrs: Vec<usize>,
    out_val_addrs: Vec<usize>,
}

impl OrtRunScratch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn retained_bytes(&self) -> usize {
        (self.in_name_addrs.capacity()
            + self.in_val_addrs.capacity()
            + self.out_name_addrs.capacity()
            + self.out_val_addrs.capacity())
            * std::mem::size_of::<usize>()
    }
}

struct ScratchReleaseGuard<'a> {
    env: &'a OrtEnv,
    scratch: &'a mut OrtRunScratch,
}

impl Drop for ScratchReleaseGuard<'_> {
    fn drop(&mut self) {
        for addr in self
            .scratch
            .out_val_addrs
            .drain(..)
            .chain(self.scratch.in_val_addrs.drain(..))
        {
            if addr != 0 {
                // SAFETY: Valid `OrtValue*` created by `CreateTensorWithDataAsOrtValue`.
                unsafe { (self.env.vtab.release_value)(addr as *mut c_void) };
            }
        }
        self.scratch.in_name_addrs.clear();
        self.scratch.out_name_addrs.clear();
    }
}

/// Thread-safe ONNX Runtime inference session (`Send + Sync` via immutable weights).
pub struct OrtSession {
    env: Arc<OrtEnv>,
    sess_addr: usize,
}

impl Drop for OrtSession {
    fn drop(&mut self) {
        if self.sess_addr != 0 {
            // SAFETY: Created once in `OrtSession::new`, destroyed once here.
            unsafe { (self.env.vtab.release_session)(self.sess_addr as *mut c_void) };
        }
    }
}

impl OrtSession {
    pub fn new(
        env: Arc<OrtEnv>,
        model_path: &Path,
        intra_threads: i32,
    ) -> Result<Self, StreamError> {
        let c_model: CString = path_to_cstring(model_path)?;
        // SAFETY: Options handle is scoped and always freed; session handle checked before return.
        unsafe {
            let mut opts: *mut c_void = null_mut();
            env.vtab.check(
                (env.vtab.create_session_options)(&mut opts),
                "CreateSessionOptions",
            )?;
            let res = (|| -> Result<usize, StreamError> {
                env.vtab.check(
                    (env.vtab.set_graph_opt)(opts, ORT_GRAPH_OPT_ALL),
                    "SetSessionGraphOptimizationLevel",
                )?;
                env.vtab.check(
                    (env.vtab.set_intra_threads)(opts, intra_threads.max(1)),
                    "SetIntraOpNumThreads",
                )?;
                env.vtab.check(
                    (env.vtab.set_inter_threads)(opts, 1),
                    "SetInterOpNumThreads",
                )?;
                let mut sess: *mut c_void = null_mut();
                env.vtab.check(
                    (env.vtab.create_session)(
                        env.env_addr as *mut c_void,
                        c_model.as_ptr(),
                        opts,
                        &mut sess,
                    ),
                    "CreateSession",
                )?;
                Ok(sess as usize)
            })();
            (env.vtab.release_opts)(opts);
            let sess_addr = res?;
            Ok(Self { env, sess_addr })
        }
    }

    pub fn run_into(
        &self,
        inputs: &[(&CStr, OrtTensorIn<'_>)],
        outputs: &mut [(&CStr, OrtTensorOut<'_>)],
    ) -> Result<(), StreamError> {
        let mut scratch = OrtRunScratch::new();
        self.run_into_with_scratch(inputs, outputs, &mut scratch)
    }

    pub fn run_into_with_scratch(
        &self,
        inputs: &[(&CStr, OrtTensorIn<'_>)],
        outputs: &mut [(&CStr, OrtTensorOut<'_>)],
        scratch: &mut OrtRunScratch,
    ) -> Result<(), StreamError> {
        const {
            assert!(std::mem::size_of::<usize>() == std::mem::size_of::<*const c_char>());
            assert!(std::mem::size_of::<usize>() == std::mem::size_of::<*mut c_void>());
        }
        let guard = ScratchReleaseGuard {
            env: &self.env,
            scratch,
        };
        guard.scratch.in_name_addrs.clear();
        guard.scratch.in_val_addrs.clear();
        guard.scratch.out_name_addrs.clear();
        guard.scratch.out_val_addrs.clear();

        for (name, t) in inputs {
            validate_tensor_bytes(name, t.shape, t.elem, t.byte_len)?;
            let mut val: *mut c_void = null_mut();
            // SAFETY: Backing Rust slice outlives `guard` and is read-only during `Run`.
            unsafe {
                self.env.vtab.check(
                    (self.env.vtab.create_tensor)(
                        self.env.mem_addr as *mut c_void,
                        t.data_addr as *mut c_void,
                        t.byte_len,
                        t.shape.as_ptr(),
                        t.shape.len(),
                        t.elem as c_int,
                        &mut val,
                    ),
                    "CreateTensor(in)",
                )?;
            }
            guard.scratch.in_name_addrs.push(name.as_ptr() as usize);
            guard.scratch.in_val_addrs.push(val as usize);
        }

        for (name, t) in outputs.iter_mut() {
            validate_tensor_bytes(name, t.shape, t.elem, t.byte_len)?;
            let mut val: *mut c_void = null_mut();
            // SAFETY: Mutable Rust slice is exclusively borrowed for the duration of `Run`.
            unsafe {
                self.env.vtab.check(
                    (self.env.vtab.create_tensor)(
                        self.env.mem_addr as *mut c_void,
                        t.data_addr as *mut c_void,
                        t.byte_len,
                        t.shape.as_ptr(),
                        t.shape.len(),
                        t.elem as c_int,
                        &mut val,
                    ),
                    "CreateTensor(out)",
                )?;
            }
            guard.scratch.out_name_addrs.push(name.as_ptr() as usize);
            guard.scratch.out_val_addrs.push(val as usize);
        }

        // SAFETY: `usize` buffers have identical layout to pointer arrays and remain live across `Run`.
        unsafe {
            self.env.vtab.check(
                (self.env.vtab.run)(
                    self.sess_addr as *mut c_void,
                    std::ptr::null(),
                    guard.scratch.in_name_addrs.as_ptr().cast::<*const c_char>(),
                    guard.scratch.in_val_addrs.as_ptr().cast::<*const c_void>(),
                    guard.scratch.in_val_addrs.len(),
                    guard
                        .scratch
                        .out_name_addrs
                        .as_ptr()
                        .cast::<*const c_char>(),
                    guard.scratch.out_val_addrs.len(),
                    guard
                        .scratch
                        .out_val_addrs
                        .as_mut_ptr()
                        .cast::<*mut c_void>(),
                ),
                "Run",
            )?;
        }
        Ok(())
    }
}

fn validate_tensor_bytes(
    name: &CStr,
    shape: &[i64],
    elem: OrtElemType,
    byte_len: usize,
) -> Result<(), StreamError> {
    let elem_bytes = match elem {
        OrtElemType::Float32 => 4usize,
        OrtElemType::Int64 => 8usize,
    };
    let expected_bytes = shape
        .iter()
        .try_fold(elem_bytes, |acc, &d| {
            usize::try_from(d).ok().and_then(|u| acc.checked_mul(u))
        })
        .unwrap_or(usize::MAX);
    if expected_bytes != byte_len {
        return Err(StreamError::Protocol(format!(
            "ORT tensor '{}' byte_len {byte_len} != shape {shape:?} * {elem_bytes}",
            name.to_string_lossy()
        )));
    }
    Ok(())
}
