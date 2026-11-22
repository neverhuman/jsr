//! Regression test for per-thread `sched_getaffinity` verification in
//! `pin_current_thread_verified` (caught on live `xbabe0` multi-worker rehearsal: `/proc/self/status`
//! reads the main thread's mask rather than the calling thread's mask).

use ultra_stream::pin_current_thread_verified;

#[test]
fn spawned_thread_pin_reads_calling_thread_affinity_not_main_thread() {
    let parent_cpus = pin_current_thread_verified(&[]).expect("query parent affinity");
    if parent_cpus.len() < 2 {
        return;
    }
    let child_target = vec![parent_cpus[1]];
    let child_target_clone = child_target.clone();
    let handle = std::thread::spawn(move || {
        let eff = pin_current_thread_verified(&child_target_clone)
            .expect("child thread pin must match calling thread affinity");
        assert_eq!(eff, child_target_clone);
    });
    handle.join().expect("child thread join");

    // Parent thread's affinity mask must remain untouched.
    let parent_after = pin_current_thread_verified(&[]).expect("query parent affinity after");
    assert_eq!(parent_after, parent_cpus);
}
