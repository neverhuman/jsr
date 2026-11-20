//! Checked copying and ownership of native offline results (WP5 hardening).

use crate::api::StreamError;
use crate::sherpa_c_offline::SherpaOnnxOfflineRecognizerResult;
use crate::sherpa_ffi::FnDestroyOfflineRes;

/// # Safety
/// A non-NULL `result` and its `text` field must be live native pointers compatible with `free`.
pub(super) unsafe fn copy_offline_result_and_free(
    result: *const SherpaOnnxOfflineRecognizerResult,
    free: FnDestroyOfflineRes,
) -> Result<String, StreamError> {
    if result.is_null() {
        return Err(StreamError::Runtime("native offline result missing".into()));
    }
    let copied = (|| {
        // SAFETY: `result` is non-null and valid until `free(result)` below.
        let text = unsafe { (*result).text };
        if text.is_null() {
            return Err(StreamError::Runtime("native offline text missing".into()));
        }
        // SAFETY: `text` is a NUL-terminated buffer owned by `result` until `free(result)` below.
        unsafe { std::ffi::CStr::from_ptr(text) }
            .to_str()
            .map(str::to_owned)
            .map_err(|_| StreamError::Runtime("native offline text is not UTF-8".into()))
    })();
    // SAFETY: Non-null `result` is freed exactly once on both Ok and Err paths.
    unsafe { free(result) };
    copied
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static OFFLINE_FREED: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn record_free(_: *const SherpaOnnxOfflineRecognizerResult) {
        OFFLINE_FREED.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn offline_checked_result_rejects_null_and_bad_utf8_and_always_frees() {
        OFFLINE_FREED.store(0, Ordering::SeqCst);
        // SAFETY: NULL handled before `free`.
        assert!(unsafe { copy_offline_result_and_free(std::ptr::null(), record_free) }.is_err());
        assert_eq!(OFFLINE_FREED.load(Ordering::SeqCst), 0);

        for (text, expected) in [
            (std::ptr::null(), None),
            (c"".as_ptr(), Some("")),
            (c"   speech  ".as_ptr(), Some("   speech  ")),
            (c"\xff".as_ptr(), None),
        ] {
            let res = SherpaOnnxOfflineRecognizerResult {
                text,
                timestamps: std::ptr::null_mut(),
                count: 0,
                tokens: std::ptr::null(),
                tokens_arr: std::ptr::null(),
                json: std::ptr::null(),
                lang: std::ptr::null(),
                emotion: std::ptr::null(),
                event: std::ptr::null(),
                durations: std::ptr::null_mut(),
                ys_log_probs: std::ptr::null_mut(),
                segment_timestamps: std::ptr::null(),
                segment_durations: std::ptr::null(),
                segment_texts: std::ptr::null(),
                segment_texts_arr: std::ptr::null(),
                segment_count: 0,
            };
            let before = OFFLINE_FREED.load(Ordering::SeqCst);
            // SAFETY: Stack `res` + static `CStr` outlive call; mock callback only increments counter.
            let got = unsafe { copy_offline_result_and_free(&res, record_free) };
            assert_eq!(got.ok().as_deref(), expected);
            assert_eq!(OFFLINE_FREED.load(Ordering::SeqCst), before + 1);
        }
    }
}
