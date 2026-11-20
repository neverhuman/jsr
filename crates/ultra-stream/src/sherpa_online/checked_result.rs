//! Checked copying and ownership of native online results.

use crate::api::StreamError;

/// # Safety
/// A non-NULL result and its text must be live native pointers compatible with `free`.
pub(super) unsafe fn copy_result_and_free(
    result: *const crate::sherpa_c_types::SherpaOnnxOnlineRecognizerResult,
    free: crate::sherpa_ffi::FnDestroyOnlineRes,
) -> Result<String, StreamError> {
    if result.is_null() {
        return Err(StreamError::Runtime("native online result missing".into()));
    }
    let copied = (|| {
        // SAFETY: the caller guarantees a live result until free below.
        let text = unsafe { (*result).text };
        if text.is_null() {
            return Err(StreamError::Runtime("native online text missing".into()));
        }
        // SAFETY: the native result owns NUL-terminated text until free below.
        unsafe { std::ffi::CStr::from_ptr(text) }
            .to_str()
            .map(str::to_owned)
            .map_err(|_| StreamError::Runtime("native online text is not UTF-8".into()))
    })();
    // SAFETY: a non-NULL native result is freed exactly once, including failure paths.
    unsafe {
        free(result);
    }
    copied
}

#[cfg(test)]
mod checked_result_tests {
    use super::*;
    use crate::sherpa_c_types::SherpaOnnxOnlineRecognizerResult;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FREED: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn record_free(_: *const SherpaOnnxOnlineRecognizerResult) {
        FREED.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn checked_results_distinguish_absence_and_utf8_failure_from_empty_text() {
        FREED.store(0, Ordering::SeqCst);
        // SAFETY: NULL is explicitly handled without invoking the freeing callback.
        assert!(unsafe { copy_result_and_free(std::ptr::null(), record_free) }.is_err());
        assert_eq!(FREED.load(Ordering::SeqCst), 0);
        for (text, expected) in [
            (std::ptr::null(), None),
            (c"".as_ptr(), Some("")),
            (c"hello".as_ptr(), Some("hello")),
            (c"\xff".as_ptr(), None),
        ] {
            let result = SherpaOnnxOnlineRecognizerResult {
                text,
                tokens: std::ptr::null(),
                tokens_arr: std::ptr::null(),
                timestamps: std::ptr::null_mut(),
                count: 0,
                json: std::ptr::null(),
            };
            let before = FREED.load(Ordering::SeqCst);
            // SAFETY: result/text live for the call; mock free only records invocation.
            let got = unsafe { copy_result_and_free(&result, record_free) };
            assert_eq!(got.ok().as_deref(), expected);
            assert_eq!(FREED.load(Ordering::SeqCst), before + 1);
        }
    }
}
