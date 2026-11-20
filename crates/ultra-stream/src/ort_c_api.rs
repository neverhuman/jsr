//! Pre-allocated tensor views (`OrtTensorIn`, `OrtTensorOut`) and RAII `OrtSession` wrapper
//! over [`OrtEnv`] (`OrtApi(17)`).

use std::ffi::{c_int, c_void, CStr, CString};
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

struct ScopedValue<'a> {
    env: &'a OrtEnv,
    addr: usize,
}

impl Drop for ScopedValue<'_> {
    fn drop(&mut self) {
        if self.addr != 0 {
            // SAFETY: Valid `OrtValue*` created by `CreateTensorWithDataAsOrtValue`.
            unsafe { (self.env.vtab.release_value)(self.addr as *mut c_void) };
        }
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
        let mut in_guards = Vec::with_capacity(inputs.len());
        let mut in_names = Vec::with_capacity(inputs.len());
        let mut in_ptrs = Vec::with_capacity(inputs.len());

        for (name, t) in inputs {
            let expected_elems: usize = t
                .shape
                .iter()
                .map(|&d| usize::try_from(d).unwrap_or(0))
                .product();
            let elem_bytes = match t.elem {
                OrtElemType::Float32 => 4,
                OrtElemType::Int64 => 8,
            };
            if expected_elems * elem_bytes != t.byte_len {
                return Err(StreamError::Protocol(format!(
                    "ORT input '{}' byte_len {} != shape {:?} * {elem_bytes}",
                    name.to_string_lossy(),
                    t.byte_len,
                    t.shape
                )));
            }
            let mut val: *mut c_void = null_mut();
            // SAFETY: Backing Rust slice outlives `in_guards` and is read-only during `Run`.
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
            in_names.push(name.as_ptr());
            in_ptrs.push(val as *const c_void);
            in_guards.push(ScopedValue {
                env: &self.env,
                addr: val as usize,
            });
        }

        let mut out_guards = Vec::with_capacity(outputs.len());
        let mut out_names = Vec::with_capacity(outputs.len());
        let mut out_ptrs = Vec::with_capacity(outputs.len());

        for (name, t) in outputs.iter_mut() {
            let expected_elems: usize = t
                .shape
                .iter()
                .map(|&d| usize::try_from(d).unwrap_or(0))
                .product();
            let elem_bytes = match t.elem {
                OrtElemType::Float32 => 4,
                OrtElemType::Int64 => 8,
            };
            if expected_elems * elem_bytes != t.byte_len {
                return Err(StreamError::Protocol(format!(
                    "ORT output '{}' byte_len {} != shape {:?} * {elem_bytes}",
                    name.to_string_lossy(),
                    t.byte_len,
                    t.shape
                )));
            }
            let mut val: *mut c_void = null_mut();
            // SAFETY: Mutable Rust slice is exclusively borrowed for the duration of `run_into`.
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
            out_names.push(name.as_ptr());
            out_ptrs.push(val);
            out_guards.push(ScopedValue {
                env: &self.env,
                addr: val as usize,
            });
        }

        // SAFETY: Session, input/output name arrays, and `OrtValue*` arrays are valid for `Run`.
        unsafe {
            self.env.vtab.check(
                (self.env.vtab.run)(
                    self.sess_addr as *mut c_void,
                    std::ptr::null(),
                    in_names.as_ptr(),
                    in_ptrs.as_ptr(),
                    in_ptrs.len(),
                    out_names.as_ptr(),
                    out_ptrs.len(),
                    out_ptrs.as_mut_ptr(),
                ),
                "Run",
            )?;
        }
        drop(out_guards);
        drop(in_guards);
        Ok(())
    }
}
