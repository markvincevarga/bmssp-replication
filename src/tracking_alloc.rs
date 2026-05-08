use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let size = layout.size();
            let current = CURRENT.fetch_add(size, Ordering::Relaxed) + size;
            PEAK.fetch_max(current, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

pub fn peak_bytes() -> usize {
    PEAK.load(Ordering::Relaxed)
}

pub fn current_bytes() -> usize {
    CURRENT.load(Ordering::Relaxed)
}

pub fn reset() {
    PEAK.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
}
