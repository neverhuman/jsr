//! Dynamic `dlopen` vtable loader and RAII `OrtEnv` for ONNX Runtime's frozen `OrtApi(17)` C ABI.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::{Path, PathBuf};
use std::ptr::null_mut;
use std::sync::Arc;

use ultra_core::receipt::sha256_hex;

use crate::api::StreamError;
use crate::sherpa_ffi::path_to_cstring;

pub(crate) const ORT_API_VERSION: u32 = 17;
const ORT_LOGGING_LEVEL_ERROR: c_int = 3;
pub(crate) const ORT_GRAPH_OPT_ALL: c_int = 99;
const ORT_ARENA_ALLOCATOR: c_int = 1;
const ORT_MEM_TYPE_DEFAULT: c_int = 0;

#[repr(C)]
struct OrtApiBase {
    get_api: unsafe extern "C" fn(u32) -> *const *const c_void,
    get_version_string: unsafe extern "C" fn() -> *const c_char,
}

type FnGetApiBase = unsafe extern "C" fn() -> *const OrtApiBase;
pub(crate) type FnGetErrorMessage = unsafe extern "C" fn(*mut c_void) -> *const c_char;
pub(crate) type FnRelease = unsafe extern "C" fn(*mut c_void);
pub(crate) type FnCreateEnv =
    unsafe extern "C" fn(c_int, *const c_char, *mut *mut c_void) -> *mut c_void;
pub(crate) type FnCreateSessionOptions = unsafe extern "C" fn(*mut *mut c_void) -> *mut c_void;
pub(crate) type FnSetSessionInt = unsafe extern "C" fn(*mut c_void, c_int) -> *mut c_void;
pub(crate) type FnCreateSession =
    unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_void, *mut *mut c_void) -> *mut c_void;
pub(crate) type FnCreateCpuMemoryInfo =
    unsafe extern "C" fn(c_int, c_int, *mut *mut c_void) -> *mut c_void;
pub(crate) type FnCreateTensorWithData = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    usize,
    *const i64,
    usize,
    c_int,
    *mut *mut c_void,
) -> *mut c_void;
pub(crate) type FnRun = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    *const *const c_char,
    *const *const c_void,
    usize,
    *const *const c_char,
    usize,
    *mut *mut c_void,
) -> *mut c_void;

pub(crate) struct OrtVtable {
    pub get_error_message: FnGetErrorMessage,
    pub create_env: FnCreateEnv,
    pub create_session: FnCreateSession,
    pub run: FnRun,
    pub create_session_options: FnCreateSessionOptions,
    pub set_graph_opt: FnSetSessionInt,
    pub set_intra_threads: FnSetSessionInt,
    pub set_inter_threads: FnSetSessionInt,
    pub create_tensor: FnCreateTensorWithData,
    pub create_cpu_mem: FnCreateCpuMemoryInfo,
    pub release_env: FnRelease,
    pub release_status: FnRelease,
    pub release_mem: FnRelease,
    pub release_session: FnRelease,
    pub release_value: FnRelease,
    pub release_opts: FnRelease,
}

impl OrtVtable {
    /// # Safety
    /// `status` must be either `NULL` or a live `OrtStatus*` returned by the same `OrtEnv` vtable.
    pub(crate) unsafe fn check(&self, status: *mut c_void, ctx: &str) -> Result<(), StreamError> {
        if status.is_null() {
            return Ok(());
        }
        // SAFETY: Non-null `OrtStatus*` returned by ORT C API; message copied before release.
        let msg = unsafe {
            let p = (self.get_error_message)(status);
            let s = if p.is_null() {
                "unknown ORT error".to_owned()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            };
            (self.release_status)(status);
            s
        };
        Err(StreamError::Runtime(format!("ORT {ctx}: {msg}")))
    }
}

/// Shared ONNX Runtime library + environment + CPU `OrtMemoryInfo`.
pub struct OrtEnv {
    _lib_addr: usize,
    pub path: PathBuf,
    pub sha256: String,
    pub version: String,
    pub(crate) vtab: OrtVtable,
    pub(crate) env_addr: usize,
    pub(crate) mem_addr: usize,
}

impl Drop for OrtEnv {
    fn drop(&mut self) {
        // SAFETY: `mem_addr` and `env_addr` were allocated once in `load_from` and outlived sessions.
        unsafe {
            if self.mem_addr != 0 {
                (self.vtab.release_mem)(self.mem_addr as *mut c_void);
            }
            if self.env_addr != 0 {
                (self.vtab.release_env)(self.env_addr as *mut c_void);
            }
        }
    }
}

fn candidate_ort_paths() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("ULTRA_ORT_LIB") {
        if !p.is_empty() {
            out.push(PathBuf::from(p));
        }
    }
    for root in ["/home/ubuntu/jsr/.venv", ".venv"] {
        let sp = Path::new(root).join("lib/python3.12/site-packages");
        let capi = sp.join("onnxruntime/capi");
        if let Ok(rd) = std::fs::read_dir(&capi) {
            let mut found: Vec<PathBuf> = rd
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("libonnxruntime.so"))
                })
                .collect();
            found.sort();
            out.extend(found);
        }
        out.push(sp.join("sherpa_onnx/lib/libonnxruntime.so"));
    }
    out
}

unsafe fn vtab_slot<F: Copy>(table: *const *const c_void, idx: usize) -> Result<F, StreamError> {
    const {
        assert!(std::mem::size_of::<*const c_void>() == std::mem::size_of::<Option<F>>());
    }
    // SAFETY: `OrtApi(17)` has >100 C function-pointer slots; `Option<extern "C" fn>` has the
    // exact nullable C function-pointer layout verified by the compile-time size assertion above.
    let opt = unsafe { table.add(idx).cast::<Option<F>>().read() };
    opt.ok_or_else(|| {
        StreamError::BackendUnavailable(format!("OrtApi({ORT_API_VERSION}) slot {idx} is NULL"))
    })
}

impl OrtEnv {
    pub fn load_default() -> Result<Arc<Self>, StreamError> {
        let path = candidate_ort_paths()
            .into_iter()
            .find(|p| p.is_file())
            .ok_or_else(|| {
                StreamError::BackendUnavailable(
                    "libonnxruntime.so not found (set ULTRA_ORT_LIB or run `just venv`)".into(),
                )
            })?;
        Self::load_from(&path)
    }

    pub fn load_from(path: &Path) -> Result<Arc<Self>, StreamError> {
        let lib_bytes = std::fs::read(path).map_err(|e| {
            StreamError::BackendUnavailable(format!("read {}: {e}", path.display()))
        })?;
        let sha256 = sha256_hex(&lib_bytes);
        let c_path = path_to_cstring(path)?;
        // SAFETY: Valid NUL-terminated path; RTLD_LOCAL | RTLD_DEEPBIND isolates ORT symbols.
        let mut handle = unsafe {
            libc::dlopen(
                c_path.as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_DEEPBIND,
            )
        };
        if handle.is_null() {
            // SAFETY: Retry without RTLD_DEEPBIND on platforms/sanitizers that disallow it.
            handle = unsafe { libc::dlopen(c_path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
        }
        if handle.is_null() {
            return Err(StreamError::BackendUnavailable(format!(
                "dlopen({}) failed",
                path.display()
            )));
        }

        // SAFETY: `OrtGetApiBase` is the single exported C entrypoint of `libonnxruntime.so`.
        let sym = unsafe { libc::dlsym(handle, c"OrtGetApiBase".as_ptr()) };
        let _: [(); std::mem::size_of::<*mut c_void>()] =
            [(); std::mem::size_of::<Option<FnGetApiBase>>()];
        // SAFETY: Nullable `Option<FnGetApiBase>` matches POSIX `dlsym` function-pointer ABI.
        let get_base = unsafe { (&raw const sym).cast::<Option<FnGetApiBase>>().read() }
            .ok_or_else(|| {
                StreamError::BackendUnavailable(format!(
                    "OrtGetApiBase missing in {}",
                    path.display()
                ))
            })?;

        // SAFETY: `get_base` is `OrtGetApiBase`; `OrtApi(17)` vtable slots are ABI-frozen.
        unsafe {
            let base = get_base();
            if base.is_null() {
                return Err(StreamError::BackendUnavailable(
                    "OrtGetApiBase returned NULL".into(),
                ));
            }
            let ver_ptr = ((*base).get_version_string)();
            let version = if ver_ptr.is_null() {
                "unknown".into()
            } else {
                CStr::from_ptr(ver_ptr).to_string_lossy().into_owned()
            };
            let table = ((*base).get_api)(ORT_API_VERSION);
            if table.is_null() {
                return Err(StreamError::BackendUnavailable(format!(
                    "OrtApi({ORT_API_VERSION}) unsupported by ORT {version}"
                )));
            }
            let vtab = OrtVtable {
                get_error_message: vtab_slot(table, 2)?,
                create_env: vtab_slot(table, 3)?,
                create_session: vtab_slot(table, 7)?,
                run: vtab_slot(table, 9)?,
                create_session_options: vtab_slot(table, 10)?,
                set_graph_opt: vtab_slot(table, 23)?,
                set_intra_threads: vtab_slot(table, 24)?,
                set_inter_threads: vtab_slot(table, 25)?,
                create_tensor: vtab_slot(table, 49)?,
                create_cpu_mem: vtab_slot(table, 69)?,
                release_env: vtab_slot(table, 92)?,
                release_status: vtab_slot(table, 93)?,
                release_mem: vtab_slot(table, 94)?,
                release_session: vtab_slot(table, 95)?,
                release_value: vtab_slot(table, 96)?,
                release_opts: vtab_slot(table, 100)?,
            };

            let mut env_ptr: *mut c_void = null_mut();
            vtab.check(
                (vtab.create_env)(ORT_LOGGING_LEVEL_ERROR, c"ultra".as_ptr(), &mut env_ptr),
                "CreateEnv",
            )?;
            let mut mem_ptr: *mut c_void = null_mut();
            if let Err(e) = vtab.check(
                (vtab.create_cpu_mem)(ORT_ARENA_ALLOCATOR, ORT_MEM_TYPE_DEFAULT, &mut mem_ptr),
                "CreateCpuMemoryInfo",
            ) {
                (vtab.release_env)(env_ptr);
                return Err(e);
            }

            Ok(Arc::new(Self {
                _lib_addr: handle as usize,
                path: path.to_path_buf(),
                sha256,
                version,
                vtab,
                env_addr: env_ptr as usize,
                mem_addr: mem_ptr as usize,
            }))
        }
    }
}
