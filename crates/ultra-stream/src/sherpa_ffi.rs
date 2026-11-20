//! Dynamic `dlopen` symbol table for `libsherpa-onnx-c-api.so` without raw-pointer `Send`/`Sync`
//! `unsafe impl`s (handles are stored as `usize` addresses and cast only at the FFI call site).

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::api::StreamError;
pub use crate::sherpa_c_types::*;

pub(crate) type FnGetStr = unsafe extern "C" fn() -> *const c_char;
pub(crate) type FnCreateOnlineRec = unsafe extern "C" fn(
    *const SherpaOnnxOnlineRecognizerConfig,
) -> *const SherpaOnnxOnlineRecognizer;
pub(crate) type FnDestroyOnlineRec = unsafe extern "C" fn(*const SherpaOnnxOnlineRecognizer);
pub(crate) type FnCreateOnlineStream =
    unsafe extern "C" fn(*const SherpaOnnxOnlineRecognizer) -> *const SherpaOnnxOnlineStream;
pub(crate) type FnDestroyOnlineStream = unsafe extern "C" fn(*const SherpaOnnxOnlineStream);
pub(crate) type FnAcceptOnline =
    unsafe extern "C" fn(*const SherpaOnnxOnlineStream, i32, *const f32, i32);
pub(crate) type FnIsReady =
    unsafe extern "C" fn(*const SherpaOnnxOnlineRecognizer, *const SherpaOnnxOnlineStream) -> i32;
pub(crate) type FnDecodeOnline =
    unsafe extern "C" fn(*const SherpaOnnxOnlineRecognizer, *const SherpaOnnxOnlineStream);
pub(crate) type FnDecodeMultiOnline = unsafe extern "C" fn(
    *const SherpaOnnxOnlineRecognizer,
    *const *const SherpaOnnxOnlineStream,
    i32,
);
pub(crate) type FnInputFinished = unsafe extern "C" fn(*const SherpaOnnxOnlineStream);
pub(crate) type FnGetOnlineRes = unsafe extern "C" fn(
    *const SherpaOnnxOnlineRecognizer,
    *const SherpaOnnxOnlineStream,
) -> *const SherpaOnnxOnlineRecognizerResult;
pub(crate) type FnDestroyOnlineRes = unsafe extern "C" fn(*const SherpaOnnxOnlineRecognizerResult);

pub(crate) type FnCreateOfflineRec = unsafe extern "C" fn(
    *const SherpaOnnxOfflineRecognizerConfig,
) -> *const SherpaOnnxOfflineRecognizer;
pub(crate) type FnDestroyOfflineRec = unsafe extern "C" fn(*const SherpaOnnxOfflineRecognizer);
pub(crate) type FnCreateOfflineStream =
    unsafe extern "C" fn(*const SherpaOnnxOfflineRecognizer) -> *const SherpaOnnxOfflineStream;
pub(crate) type FnDestroyOfflineStream = unsafe extern "C" fn(*const SherpaOnnxOfflineStream);
pub(crate) type FnAcceptOffline =
    unsafe extern "C" fn(*const SherpaOnnxOfflineStream, i32, *const f32, i32);
pub(crate) type FnDecodeOffline =
    unsafe extern "C" fn(*const SherpaOnnxOfflineRecognizer, *const SherpaOnnxOfflineStream);
pub(crate) type FnDecodeMultiOffline = unsafe extern "C" fn(
    *const SherpaOnnxOfflineRecognizer,
    *const *const SherpaOnnxOfflineStream,
    i32,
);
pub(crate) type FnGetOfflineRes = unsafe extern "C" fn(
    *const SherpaOnnxOfflineStream,
) -> *const SherpaOnnxOfflineRecognizerResult;
pub(crate) type FnDestroyOfflineRes =
    unsafe extern "C" fn(*const SherpaOnnxOfflineRecognizerResult);

/// Dynamically loaded `libsherpa-onnx-c-api.so` function table (`Send + Sync` via `usize` handles).
pub struct SherpaLib {
    _ort_addr: usize,
    _lib_addr: usize,
    pub path: PathBuf,
    pub(crate) get_version: FnGetStr,
    pub(crate) get_ort_version: FnGetStr,
    pub(crate) create_online_rec: FnCreateOnlineRec,
    pub(crate) destroy_online_rec: FnDestroyOnlineRec,
    pub(crate) create_online_stream: FnCreateOnlineStream,
    pub(crate) destroy_online_stream: FnDestroyOnlineStream,
    pub(crate) accept_online: FnAcceptOnline,
    pub(crate) is_ready: FnIsReady,
    pub(crate) decode_online: FnDecodeOnline,
    pub(crate) decode_multi_online: FnDecodeMultiOnline,
    pub(crate) input_finished: FnInputFinished,
    pub(crate) get_online_res: FnGetOnlineRes,
    pub(crate) destroy_online_res: FnDestroyOnlineRes,
    pub(crate) create_offline_rec: FnCreateOfflineRec,
    pub(crate) destroy_offline_rec: FnDestroyOfflineRec,
    pub(crate) create_offline_stream: FnCreateOfflineStream,
    pub(crate) destroy_offline_stream: FnDestroyOfflineStream,
    pub(crate) accept_offline: FnAcceptOffline,
    pub(crate) decode_offline: FnDecodeOffline,
    pub(crate) decode_multi_offline: FnDecodeMultiOffline,
    pub(crate) get_offline_res: FnGetOfflineRes,
    pub(crate) destroy_offline_res: FnDestroyOfflineRes,
}

impl SherpaLib {
    /// Discover and load `libsherpa-onnx-c-api.so` from `ULTRA_SHERPA_LIB` or the repo `.venv`.
    pub fn load_default() -> Result<Arc<Self>, StreamError> {
        let path = candidate_so_paths()
            .into_iter()
            .find(|p| p.exists())
            .ok_or_else(|| {
                StreamError::BackendUnavailable(
                    "libsherpa-onnx-c-api.so not found (set ULTRA_SHERPA_LIB or run `just venv`)"
                        .into(),
                )
            })?;
        Self::load_from(&path)
    }

    pub fn load_from(path: &Path) -> Result<Arc<Self>, StreamError> {
        let ort_addr = if let Some(parent) = path.parent() {
            let ort = parent.join("libonnxruntime.so");
            if ort.exists() {
                let c_ort = path_to_cstring(&ort)?;
                // SAFETY: Valid NUL-terminated path; RTLD_NOW | RTLD_GLOBAL preloads sibling ORT.
                (unsafe { libc::dlopen(c_ort.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) })
                    as usize
            } else {
                0
            }
        } else {
            0
        };

        let c_path = path_to_cstring(path)?;
        // SAFETY: Valid NUL-terminated path; handle checked for null before symbol lookup.
        let handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        if handle.is_null() {
            return Err(StreamError::BackendUnavailable(format!(
                "dlopen({}) failed: {}",
                path.display(),
                dlerror_string()
            )));
        }

        // SAFETY: Each symbol name matches the exact `c-api.h` C prototype verified by test.
        unsafe {
            Ok(Arc::new(Self {
                _ort_addr: ort_addr,
                _lib_addr: handle as usize,
                path: path.to_path_buf(),
                get_version: std::mem::transmute::<*mut c_void, FnGetStr>(load_sym(
                    handle,
                    b"SherpaOnnxGetVersionStr\0",
                )?),
                get_ort_version: std::mem::transmute::<*mut c_void, FnGetStr>(load_sym(
                    handle,
                    b"SherpaOnnxGetOnnxruntimeVersionStr\0",
                )?),
                create_online_rec: std::mem::transmute::<*mut c_void, FnCreateOnlineRec>(load_sym(
                    handle,
                    b"SherpaOnnxCreateOnlineRecognizer\0",
                )?),
                destroy_online_rec: std::mem::transmute::<*mut c_void, FnDestroyOnlineRec>(
                    load_sym(handle, b"SherpaOnnxDestroyOnlineRecognizer\0")?,
                ),
                create_online_stream: std::mem::transmute::<*mut c_void, FnCreateOnlineStream>(
                    load_sym(handle, b"SherpaOnnxCreateOnlineStream\0")?,
                ),
                destroy_online_stream: std::mem::transmute::<*mut c_void, FnDestroyOnlineStream>(
                    load_sym(handle, b"SherpaOnnxDestroyOnlineStream\0")?,
                ),
                accept_online: std::mem::transmute::<*mut c_void, FnAcceptOnline>(load_sym(
                    handle,
                    b"SherpaOnnxOnlineStreamAcceptWaveform\0",
                )?),
                is_ready: std::mem::transmute::<*mut c_void, FnIsReady>(load_sym(
                    handle,
                    b"SherpaOnnxIsOnlineStreamReady\0",
                )?),
                decode_online: std::mem::transmute::<*mut c_void, FnDecodeOnline>(load_sym(
                    handle,
                    b"SherpaOnnxDecodeOnlineStream\0",
                )?),
                decode_multi_online: std::mem::transmute::<*mut c_void, FnDecodeMultiOnline>(
                    load_sym(handle, b"SherpaOnnxDecodeMultipleOnlineStreams\0")?,
                ),
                input_finished: std::mem::transmute::<*mut c_void, FnInputFinished>(load_sym(
                    handle,
                    b"SherpaOnnxOnlineStreamInputFinished\0",
                )?),
                get_online_res: std::mem::transmute::<*mut c_void, FnGetOnlineRes>(load_sym(
                    handle,
                    b"SherpaOnnxGetOnlineStreamResult\0",
                )?),
                destroy_online_res: std::mem::transmute::<*mut c_void, FnDestroyOnlineRes>(
                    load_sym(handle, b"SherpaOnnxDestroyOnlineRecognizerResult\0")?,
                ),
                create_offline_rec: std::mem::transmute::<*mut c_void, FnCreateOfflineRec>(
                    load_sym(handle, b"SherpaOnnxCreateOfflineRecognizer\0")?,
                ),
                destroy_offline_rec: std::mem::transmute::<*mut c_void, FnDestroyOfflineRec>(
                    load_sym(handle, b"SherpaOnnxDestroyOfflineRecognizer\0")?,
                ),
                create_offline_stream: std::mem::transmute::<*mut c_void, FnCreateOfflineStream>(
                    load_sym(handle, b"SherpaOnnxCreateOfflineStream\0")?,
                ),
                destroy_offline_stream: std::mem::transmute::<*mut c_void, FnDestroyOfflineStream>(
                    load_sym(handle, b"SherpaOnnxDestroyOfflineStream\0")?,
                ),
                accept_offline: std::mem::transmute::<*mut c_void, FnAcceptOffline>(load_sym(
                    handle,
                    b"SherpaOnnxAcceptWaveformOffline\0",
                )?),
                decode_offline: std::mem::transmute::<*mut c_void, FnDecodeOffline>(load_sym(
                    handle,
                    b"SherpaOnnxDecodeOfflineStream\0",
                )?),
                decode_multi_offline: std::mem::transmute::<*mut c_void, FnDecodeMultiOffline>(
                    load_sym(handle, b"SherpaOnnxDecodeMultipleOfflineStreams\0")?,
                ),
                get_offline_res: std::mem::transmute::<*mut c_void, FnGetOfflineRes>(load_sym(
                    handle,
                    b"SherpaOnnxGetOfflineStreamResult\0",
                )?),
                destroy_offline_res: std::mem::transmute::<*mut c_void, FnDestroyOfflineRes>(
                    load_sym(handle, b"SherpaOnnxDestroyOfflineRecognizerResult\0")?,
                ),
            }))
        }
    }

    pub fn version(&self) -> String {
        // SAFETY: Returns static C string owned by sherpa-onnx.
        unsafe { cstr_or_empty((self.get_version)()) }
    }

    pub fn onnxruntime_version(&self) -> String {
        // SAFETY: Returns static C string owned by sherpa-onnx.
        unsafe { cstr_or_empty((self.get_ort_version)()) }
    }
}

/// Verify no CUDA/ROCm execution provider shared object is mapped into this process.
pub fn verify_cpu_only_process() -> bool {
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
        return false;
    };
    !maps.contains("libonnxruntime_providers_cuda")
        && !maps.contains("libonnxruntime_providers_tensorrt")
        && !maps.contains("libonnxruntime_providers_rocm")
        && !maps.contains("libcuda.so")
}

pub(crate) fn path_to_cstring(p: &Path) -> Result<CString, StreamError> {
    CString::new(p.to_string_lossy().as_bytes())
        .map_err(|e| StreamError::BackendUnavailable(format!("invalid path: {e}")))
}

/// Convert a nullable NUL-terminated C string pointer into an owned `String`.
///
/// # Safety
///
/// If `p` is non-null, it must point to a valid NUL-terminated C string that remains valid
/// for the duration of this call.
pub(crate) unsafe fn cstr_or_empty(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        // SAFETY: Non-null C string returned by sherpa-onnx / libc.
        unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() }
    }
}

fn candidate_so_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("ULTRA_SHERPA_LIB") {
        out.push(PathBuf::from(p));
    }
    let rel = ".venv/lib/python3.12/site-packages/sherpa_onnx/lib/libsherpa-onnx-c-api.so";
    out.push(PathBuf::from(rel));
    out.push(PathBuf::from("/home/ubuntu/jsr").join(rel));
    if let Ok(cwd) = std::env::current_dir() {
        for anc in cwd.ancestors().take(6) {
            out.push(anc.join(rel));
        }
    }
    out
}

unsafe fn load_sym(handle: *mut c_void, nul_terminated: &[u8]) -> Result<*mut c_void, StreamError> {
    // SAFETY: Caller supplies a valid dlopen handle and NUL-terminated symbol slice.
    let ptr = unsafe { libc::dlsym(handle, nul_terminated.as_ptr().cast::<c_char>()) };
    if ptr.is_null() {
        let name = String::from_utf8_lossy(&nul_terminated[..nul_terminated.len() - 1]);
        Err(StreamError::BackendUnavailable(format!(
            "missing symbol `{name}`: {}",
            dlerror_string()
        )))
    } else {
        Ok(ptr)
    }
}

fn dlerror_string() -> String {
    // SAFETY: dlerror takes no arguments and returns either NULL or a valid C string.
    let p = unsafe { libc::dlerror() };
    // SAFETY: Guarded for null inside cstr_or_empty.
    unsafe { cstr_or_empty(p) }
}
