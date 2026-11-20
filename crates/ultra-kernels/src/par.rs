//! Static row partitions across threads (rule N3): each thread owns a contiguous,
//! alignment-respecting row range, so integer results never depend on thread count.

use crate::{Backend, Frames, Gemv, KernelError};

/// `m.gemv` split over `threads` scoped threads by static row partition. Threads are
/// spawned per call (benchmark use; a runtime would keep a pool).
pub fn gemv_par<M: Gemv>(
    m: M,
    backend: Backend,
    x: &Frames<'_>,
    out: &mut [i32],
    threads: usize,
) -> Result<(), KernelError> {
    let (rows, f_n, align) = (m.rows(), x.count(), m.row_align());
    if out.len() != rows * f_n {
        return Err(KernelError::Shape("out.len() != rows * frames"));
    }
    let per = rows.div_ceil(align).div_ceil(threads.max(1)) * align;
    if per >= rows {
        return m.gemv(backend, x, out);
    }
    std::thread::scope(|s| {
        let handles: Vec<_> = out
            .chunks_mut(per * f_n)
            .enumerate()
            .map(|(i, o)| {
                let (start, end) = (i * per, ((i + 1) * per).min(rows));
                s.spawn(move || m.slice_rows(start, end)?.gemv(backend, x, o))
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|h| h.join().unwrap_or(Err(KernelError::WorkerPanicked)))
    })
}
