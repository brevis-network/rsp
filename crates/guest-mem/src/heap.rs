//! The guest's heap.
//!
//! pico-sdk gives a guest a bump allocator, `sys_alloc_aligned`: blocks are handed out left to
//! right from the end of the program image and nothing is ever taken back. `pico_sdk::entrypoint!`
//! installs it as the global allocator as it is. [`BumpHeap`] is a global allocator over the same
//! bump allocator with two changes. Neither frees or reuses anything: a byte of the heap is
//! still handed out at most once.
//!
//! **Every block starts on an 8-byte boundary.** `Vec<u8>` asks for an alignment of 1, and got
//! whatever address the block before it ended at. This target has no misaligned loads, so the
//! word-at-a-time paths of `memcpy`, `memcmp` and the keccak sponge are only taken for a buffer
//! that happens to start on a word boundary; everything else goes a byte at a time.
//!
//! **The block at the end of the heap grows where it is.** `GlobalAlloc::realloc` by default
//! allocates a new block and copies the old one into it, which is what every doubling of a
//! growing `Vec` paid, and what padding a contract's code after reading it paid once per
//! contract. A buffer being appended to is usually the last thing that was allocated, and the
//! bytes after the last block are the next ones the heap would hand out anyway.
//!
//! Measured on 13 mainnet blocks, in a Pico guest: 166.9 M -> 161.3 M instructions a block.

use core::alloc::{GlobalAlloc, Layout};

/// A bump allocator.
///
/// # Safety
///
/// `next(bytes, align)`, for `align` a power of two, returns the first address at or after the
/// end of the heap that is a multiple of `align`, and moves the end of the heap to `bytes` past
/// that address; the `bytes` bytes there are valid for reads and writes from then on and are
/// never returned again. In particular `next(0, 1)` returns the end of the heap and moves
/// nothing, and `next(n, 1)` right after it returns that same address.
pub unsafe trait Bump {
    /// The next `bytes` bytes of the heap, at a multiple of `align`.
    ///
    /// # Safety
    ///
    /// `align` is a power of two. Not reentrant.
    unsafe fn next(&self, bytes: usize, align: usize) -> *mut u8;
}

/// The alignment every block of a [`BumpHeap`] has at least: the word of the target.
pub const MIN_ALIGN: usize = 8;

/// A global allocator over a bump allocator; see the module.
pub struct BumpHeap<B>(pub B);

unsafe impl<B: Bump> GlobalAlloc for BumpHeap<B> {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.0.next(layout.size(), layout.align().max(MIN_ALIGN))
    }

    #[inline]
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old_size = layout.size();
        if new_size <= old_size {
            // Nothing is freed, so there is nothing to give back: the block stays.
            return ptr;
        }
        if ptr.add(old_size) == self.0.next(0, 1) {
            // The last block. The bytes after it are the next the heap hands out, and asking for
            // them with an alignment of 1 leaves no gap.
            self.0.next(new_size - old_size, 1);
            return ptr;
        }
        let new = self.0.next(new_size, layout.align().max(MIN_ALIGN));
        core::ptr::copy_nonoverlapping(ptr, new, old_size);
        new
    }
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use super::*;

    /// A bump allocator over a buffer, as `sys_alloc_aligned` is over the guest's memory. It
    /// starts at an odd address, as a heap that follows a program image may.
    struct Arena {
        memory: Vec<u8>,
        end: Cell<usize>,
    }

    impl Arena {
        fn new(bytes: usize) -> Self {
            let memory = vec![0u8; bytes + 16];
            let start = memory.as_ptr() as usize;
            Self { end: Cell::new(start + 1 + (start % 2)), memory }
        }

        fn end(&self) -> usize {
            self.end.get()
        }
    }

    unsafe impl Bump for Arena {
        unsafe fn next(&self, bytes: usize, align: usize) -> *mut u8 {
            assert!(align.is_power_of_two());
            let at = self.end.get().next_multiple_of(align);
            let limit = self.memory.as_ptr() as usize + self.memory.len();
            assert!(at + bytes <= limit, "the test arena is too small");
            self.end.set(at + bytes);
            at as *mut u8
        }
    }

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    unsafe fn fill(ptr: *mut u8, len: usize, seed: u8) {
        for i in 0..len {
            ptr.add(i).write(seed.wrapping_add(i as u8).wrapping_mul(31));
        }
    }

    unsafe fn holds(ptr: *const u8, len: usize, seed: u8) -> bool {
        (0..len).all(|i| ptr.add(i).read() == seed.wrapping_add(i as u8).wrapping_mul(31))
    }

    #[test]
    fn every_block_starts_on_a_word_boundary_or_a_stricter_one() {
        let heap = BumpHeap(Arena::new(1 << 16));
        assert_eq!(heap.0.end() % 2, 1, "the arena is meant to start at an odd address");
        for size in 1..=40 {
            for align in [1, 2, 4, 8, 16, 64] {
                let block = unsafe { heap.alloc(layout(size, align)) } as usize;
                assert_eq!(block % align.max(MIN_ALIGN), 0, "size {size}, align {align}");
                assert_eq!(heap.0.end(), block + size, "the block is the end of the heap");
            }
        }
    }

    #[test]
    fn the_last_block_grows_in_place() {
        let heap = BumpHeap(Arena::new(1 << 12));
        unsafe {
            let _earlier = heap.alloc(layout(24, 8));
            let block = heap.alloc(layout(10, 1));
            fill(block, 10, 7);
            // Three doublings, as a growing `Vec` asks for them.
            let mut size = 10;
            for new_size in [20, 40, 100] {
                let grown = heap.realloc(block, layout(size, 1), new_size);
                assert_eq!(grown, block, "grown to {new_size}");
                assert_eq!(heap.0.end(), block as usize + new_size, "no gap, nothing skipped");
                size = new_size;
            }
            assert!(holds(block, 10, 7));
            // The next block follows it, on its own boundary.
            let next = heap.alloc(layout(3, 1)) as usize;
            assert_eq!(next, (block as usize + 100).next_multiple_of(MIN_ALIGN));
        }
    }

    #[test]
    fn a_block_that_is_not_the_last_moves_to_the_end() {
        let heap = BumpHeap(Arena::new(1 << 12));
        unsafe {
            let block = heap.alloc(layout(10, 1));
            fill(block, 10, 3);
            let later = heap.alloc(layout(1, 1));
            later.write(0xee);
            let end = heap.0.end();

            let moved = heap.realloc(block, layout(10, 1), 21);
            assert!(moved as usize >= end, "the new block is past everything handed out");
            assert_eq!(moved as usize % MIN_ALIGN, 0);
            assert_eq!(heap.0.end(), moved as usize + 21);
            assert!(holds(moved, 10, 3), "the contents came along");
            assert!(holds(block, 10, 3), "and the old block was only read");
            assert_eq!(later.read(), 0xee);
        }
    }

    #[test]
    fn a_block_one_byte_short_of_the_end_is_not_the_last() {
        // The in-place test is on the block's own end, so a block that ends just before the
        // heap's end (it was shrunk, or something of one byte follows it) must move.
        let heap = BumpHeap(Arena::new(1 << 12));
        unsafe {
            let block = heap.alloc(layout(32, 1));
            fill(block, 32, 9);
            assert_eq!(heap.realloc(block, layout(32, 1), 31), block);
            let end = heap.0.end();
            assert_eq!(end, block as usize + 32, "shrinking gives nothing back");

            let moved = heap.realloc(block, layout(31, 1), 64);
            assert_ne!(moved, block);
            assert!(moved as usize >= end);
            assert!(holds(moved, 31, 9));
        }
    }

    #[test]
    fn a_block_that_shrinks_or_keeps_its_size_stays() {
        let heap = BumpHeap(Arena::new(1 << 12));
        unsafe {
            let block = heap.alloc(layout(100, 16));
            fill(block, 100, 5);
            let end = heap.0.end();
            for new_size in [100, 99, 16, 1] {
                assert_eq!(heap.realloc(block, layout(100, 16), new_size), block);
                assert_eq!(heap.0.end(), end);
            }
            assert!(holds(block, 100, 5));
        }
    }

    #[test]
    fn a_moved_block_keeps_the_alignment_it_was_allocated_with() {
        let heap = BumpHeap(Arena::new(1 << 14));
        unsafe {
            for align in [1usize, 8, 32, 128] {
                let block = heap.alloc(layout(48, align));
                let _later = heap.alloc(layout(5, 1));
                let moved = heap.realloc(block, layout(48, align), 200);
                assert_eq!(moved as usize % align.max(MIN_ALIGN), 0, "align {align}");
            }
        }
    }

    /// Random allocations, growths and shrinks against a model that keeps each block's bytes:
    /// every block always holds what was written to it, and no two blocks overlap.
    #[test]
    fn blocks_keep_their_contents_and_never_overlap() {
        struct Block {
            ptr: *mut u8,
            layout: Layout,
            bytes: Vec<u8>,
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut random = move |below: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % below as u64) as usize
        };

        let heap = BumpHeap(Arena::new(8 << 20));
        let mut blocks: Vec<Block> = Vec::new();
        let (mut in_place, mut moved) = (0, 0);
        for step in 0..20_000 {
            if blocks.is_empty() || random(3) == 0 {
                let layout = layout(1 + random(96), 1 << random(5));
                let ptr = unsafe { heap.alloc(layout) };
                let bytes: Vec<u8> = (0..layout.size()).map(|_| random(256) as u8).collect();
                unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
                blocks.push(Block { ptr, layout, bytes });
            } else {
                // Mostly the newest block, as a buffer being built is; sometimes an older one.
                let i = if random(4) == 0 { random(blocks.len()) } else { blocks.len() - 1 };
                let block = &mut blocks[i];
                let new_size = 1 + random(2 * block.layout.size() + 8);
                let ptr = unsafe { heap.realloc(block.ptr, block.layout, new_size) };
                if new_size > block.layout.size() {
                    if ptr == block.ptr {
                        in_place += 1
                    } else {
                        moved += 1
                    }
                } else {
                    assert_eq!(ptr, block.ptr, "step {step}");
                }
                assert_eq!(ptr as usize % block.layout.align().max(MIN_ALIGN), 0);
                let kept = block.bytes.len().min(new_size);
                block.bytes.truncate(kept);
                // Fill what was added, so that a later overlap would be seen.
                while block.bytes.len() < new_size {
                    let byte = random(256) as u8;
                    unsafe { ptr.add(block.bytes.len()).write(byte) };
                    block.bytes.push(byte);
                }
                block.ptr = ptr;
                block.layout = layout(new_size, block.layout.align());
            }
        }
        assert!(in_place > 1000 && moved > 1000, "both paths ran: {in_place} / {moved}");

        for (i, block) in blocks.iter().enumerate() {
            let held = unsafe { core::slice::from_raw_parts(block.ptr, block.bytes.len()) };
            assert_eq!(held, &block.bytes[..], "block {i}");
        }
        let mut ranges: Vec<(usize, usize)> =
            blocks.iter().map(|b| (b.ptr as usize, b.ptr as usize + b.bytes.len())).collect();
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            assert!(pair[0].1 <= pair[1].0, "blocks overlap: {pair:x?}");
        }
        assert!(ranges.last().unwrap().1 <= heap.0.end());
    }
}
