// brevis-vm's `sdk/sdk/src/memcpy.s`, as `__wrap_memcpy`: the guest links with `--wrap=memcpy` (see
// `bin/client/build-guest.sh`). The `.attribute` lines are dropped; the object's ISA attributes are
// the crate's. The function has a section of its own, not `.text`, so that `--gc-sections` drops it
// when nothing calls it: `build-guest.sh` reads its presence in the ELF as the wrap having taken.
//
// memcpy for RV64IM, written for Pico.
//
// 16 bytes or more go to the MEMCPY ecall, which the MemoryCopy chip proves in the chunk at one row
// per destination dword, whatever the two addresses' offsets: up to 2048 bytes a call, which leaves
// a0, a1 and a2 as they were. Shorter copies stay here, where every load and store is naturally
// aligned. A misaligned access runs, but each one is proven on the MemoryUnaligned chip, which
// holds few rows per chunk.
//
// - Under 8 bytes: one byte at a time.
// - Otherwise the destination is aligned first, copying its up to 7 leading bytes one at a time.
//   Then, with the source aligned too: 64, then 32, then 8 bytes a step, in `ld` / `sd`. With the
//   source at an offset within its word: each output word is put together from the two aligned
//   source words it straddles, `(w0 >> 8k) | (w1 << (64 - 8k))`, 16 bytes a step.
// - The last 0 to 7 bytes: the aligned source words holding them are loaded and shifted into
//   place, and stored with `sw` / `sh` / `sb`, which the 8-aligned destination keeps aligned.
//
// A load reads a whole aligned word only when the word holds a byte of the source. Such a word
// cannot cross into another memory region, each of which starts at a multiple of 8.

	.section	.text.__wrap_memcpy,"ax",@progbits
	.globl	__wrap_memcpy
	.p2align	2
	.type	__wrap_memcpy,@function
// a0 = dst, returned; a1 = src; a2 = n. t0 walks dst.
__wrap_memcpy:
	li	t1, 16
	bltu	a2, t1, .Lmemcpy_sw
	li	t1, 2048
	bltu	t1, a2, .Lmemcpy_calls
	li	t0, 0x136
	ecall
	ret
	// Over 2048 bytes: a call per 2048, a0 and a1 moving on, the last call taking the rest. The
	// ranges do not overlap (`memmove` copies overlapping ones itself), so the order is free.
.Lmemcpy_calls:
	mv	t2, a0
	mv	t3, a2
.Lmemcpy_call:
	mv	a2, t3
	bgeu	t1, t3, .Lmemcpy_last
	mv	a2, t1
.Lmemcpy_last:
	li	t0, 0x136
	ecall
	add	a0, a0, a2
	add	a1, a1, a2
	sub	t3, t3, a2
	bnez	t3, .Lmemcpy_call
	mv	a0, t2
	ret
.Lmemcpy_sw:
	mv	t0, a0
	li	t1, 8
	bltu	a2, t1, .Lmemcpy_bytes
	// Align dst, taking the leading bytes off n (n >= 8 > the head).
	andi	t1, t0, 7
	beqz	t1, .Lmemcpy_dst_aligned
	li	t2, 8
	sub	t1, t2, t1
	sub	a2, a2, t1
.Lmemcpy_head:
	lbu	t2, 0(a1)
	sb	t2, 0(t0)
	addi	a1, a1, 1
	addi	t0, t0, 1
	addi	t1, t1, -1
	bnez	t1, .Lmemcpy_head
.Lmemcpy_dst_aligned:
	andi	t1, a1, 7
	bnez	t1, .Lmemcpy_shift

	// Both aligned.
	li	t1, 64
	bltu	a2, t1, .Lmemcpy_a32
.Lmemcpy_a64:
	ld	a3, 0(a1)
	ld	a4, 8(a1)
	ld	a5, 16(a1)
	ld	a6, 24(a1)
	sd	a3, 0(t0)
	sd	a4, 8(t0)
	sd	a5, 16(t0)
	sd	a6, 24(t0)
	ld	a3, 32(a1)
	ld	a4, 40(a1)
	ld	a5, 48(a1)
	ld	a6, 56(a1)
	sd	a3, 32(t0)
	sd	a4, 40(t0)
	sd	a5, 48(t0)
	sd	a6, 56(t0)
	addi	a1, a1, 64
	addi	t0, t0, 64
	addi	a2, a2, -64
	bgeu	a2, t1, .Lmemcpy_a64
.Lmemcpy_a32:
	li	t1, 32
	bltu	a2, t1, .Lmemcpy_a8_check
	ld	a3, 0(a1)
	ld	a4, 8(a1)
	ld	a5, 16(a1)
	ld	a6, 24(a1)
	sd	a3, 0(t0)
	sd	a4, 8(t0)
	sd	a5, 16(t0)
	sd	a6, 24(t0)
	addi	a1, a1, 32
	addi	t0, t0, 32
	addi	a2, a2, -32
.Lmemcpy_a8_check:
	li	t1, 8
	bltu	a2, t1, .Lmemcpy_aligned_tail
.Lmemcpy_a8:
	ld	a3, 0(a1)
	sd	a3, 0(t0)
	addi	a1, a1, 8
	addi	t0, t0, 8
	addi	a2, a2, -8
	bgeu	a2, t1, .Lmemcpy_a8
.Lmemcpy_aligned_tail:
	beqz	a2, .Lmemcpy_done
	// The word at src holds all the n < 8 bytes left.
	ld	a3, 0(a1)
	j	.Lmemcpy_store_tail

	// src is t1 = 1..7 bytes into the aligned word a1 points to after this; a3 holds that word.
.Lmemcpy_shift:
	slli	t2, t1, 3
	neg	t3, t2
	andi	a1, a1, -8
	ld	a3, 0(a1)
	li	t4, 16
	bltu	a2, t4, .Lmemcpy_s8
.Lmemcpy_s16:
	ld	a4, 8(a1)
	srl	a5, a3, t2
	sll	a6, a4, t3
	or	a5, a5, a6
	sd	a5, 0(t0)
	ld	a3, 16(a1)
	srl	a5, a4, t2
	sll	a6, a3, t3
	or	a5, a5, a6
	sd	a5, 8(t0)
	addi	a1, a1, 16
	addi	t0, t0, 16
	addi	a2, a2, -16
	bgeu	a2, t4, .Lmemcpy_s16
.Lmemcpy_s8:
	li	t4, 8
	bltu	a2, t4, .Lmemcpy_shift_tail
	ld	a4, 8(a1)
	srl	a5, a3, t2
	sll	a6, a4, t3
	or	a5, a5, a6
	sd	a5, 0(t0)
	mv	a3, a4
	addi	a1, a1, 8
	addi	t0, t0, 8
	addi	a2, a2, -8
.Lmemcpy_shift_tail:
	beqz	a2, .Lmemcpy_done
	// The n < 8 bytes left start t1 bytes into a3, and run into the next word if t1 + n > 8.
	srl	a3, a3, t2
	add	t4, t1, a2
	li	t5, 8
	bgeu	t5, t4, .Lmemcpy_store_tail
	ld	a4, 8(a1)
	sll	a4, a4, t3
	or	a3, a3, a4

	// a3's low n bytes, 0 < n < 8, to the 8-aligned t0.
.Lmemcpy_store_tail:
	andi	t1, a2, 4
	beqz	t1, .Lmemcpy_tail2
	sw	a3, 0(t0)
	srli	a3, a3, 32
	addi	t0, t0, 4
.Lmemcpy_tail2:
	andi	t1, a2, 2
	beqz	t1, .Lmemcpy_tail1
	sh	a3, 0(t0)
	srli	a3, a3, 16
	addi	t0, t0, 2
.Lmemcpy_tail1:
	andi	t1, a2, 1
	beqz	t1, .Lmemcpy_done
	sb	a3, 0(t0)
.Lmemcpy_done:
	ret

.Lmemcpy_bytes:
	beqz	a2, .Lmemcpy_done
.Lmemcpy_byte:
	lbu	t1, 0(a1)
	sb	t1, 0(t0)
	addi	a1, a1, 1
	addi	t0, t0, 1
	addi	a2, a2, -1
	bnez	a2, .Lmemcpy_byte
	ret
.Lmemcpy_end:
	.size	__wrap_memcpy, .Lmemcpy_end-__wrap_memcpy
