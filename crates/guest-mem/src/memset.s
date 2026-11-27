// brevis-vm's `sdk/sdk/src/memset.s`, as `__wrap_memset`: the guest links with `--wrap=memset` (see
// `bin/client/build-guest.sh`). The `.attribute` lines are dropped; the object's ISA attributes are
// the crate's. The function has a section of its own, not `.text`, so that `--gc-sections` drops it
// when nothing calls it: `build-guest.sh` reads its presence in the ELF as the wrap having taken.
//
// memset for RV64IM, written for Pico: every store is naturally aligned. A misaligned access runs,
// but each one is proven on the MemoryUnaligned chip, which holds few rows per chunk.
//
// - Under 8 bytes: one byte at a time.
// - Otherwise the byte is copied into all eight lanes of a word, and the destination is aligned
//   with at most one `sb`, `sh` and `sw`, in that order. Then 64 bytes a step in `sd`, then 8,
//   and the last 0 to 7 bytes with `sw` / `sh` / `sb`, which the 8-aligned destination keeps
//   aligned.

	.section	.text.__wrap_memset,"ax",@progbits
	.globl	__wrap_memset
	.p2align	2
	.type	__wrap_memset,@function
// a0 = dst, returned; a1 = the byte, in its low 8 bits; a2 = n. t0 walks dst.
//
// 16 bytes or more go to the MEMSET ecall, as `memcpy`'s do to MEMCPY: up to 2048 bytes a call,
// a1 the byte alone.
__wrap_memset:
	andi	a1, a1, 255
	li	t1, 16
	bltu	a2, t1, .Lmemset_sw
	li	t1, 2048
	bltu	t1, a2, .Lmemset_calls
	li	t0, 0x137
	ecall
	ret
.Lmemset_calls:
	mv	t2, a0
	mv	t3, a2
.Lmemset_call:
	mv	a2, t3
	bgeu	t1, t3, .Lmemset_last
	mv	a2, t1
.Lmemset_last:
	li	t0, 0x137
	ecall
	add	a0, a0, a2
	sub	t3, t3, a2
	bnez	t3, .Lmemset_call
	mv	a0, t2
	ret
.Lmemset_sw:
	mv	t0, a0
	li	t1, 8
	bltu	a2, t1, .Lmemset_bytes
	slli	t1, a1, 8
	or	a1, a1, t1
	slli	t1, a1, 16
	or	a1, a1, t1
	slli	t1, a1, 32
	or	a1, a1, t1
	// Align dst (n >= 8 > the head).
	andi	t1, t0, 1
	beqz	t1, .Lmemset_head2
	sb	a1, 0(t0)
	addi	t0, t0, 1
	addi	a2, a2, -1
.Lmemset_head2:
	andi	t1, t0, 2
	beqz	t1, .Lmemset_head4
	sh	a1, 0(t0)
	addi	t0, t0, 2
	addi	a2, a2, -2
.Lmemset_head4:
	andi	t1, t0, 4
	beqz	t1, .Lmemset_aligned
	sw	a1, 0(t0)
	addi	t0, t0, 4
	addi	a2, a2, -4
.Lmemset_aligned:
	li	t1, 64
	bltu	a2, t1, .Lmemset_8_check
.Lmemset_64:
	sd	a1, 0(t0)
	sd	a1, 8(t0)
	sd	a1, 16(t0)
	sd	a1, 24(t0)
	sd	a1, 32(t0)
	sd	a1, 40(t0)
	sd	a1, 48(t0)
	sd	a1, 56(t0)
	addi	t0, t0, 64
	addi	a2, a2, -64
	bgeu	a2, t1, .Lmemset_64
.Lmemset_8_check:
	li	t1, 8
	bltu	a2, t1, .Lmemset_tail4
.Lmemset_8:
	sd	a1, 0(t0)
	addi	t0, t0, 8
	addi	a2, a2, -8
	bgeu	a2, t1, .Lmemset_8
	// n < 8 left, at the 8-aligned t0.
.Lmemset_tail4:
	andi	t1, a2, 4
	beqz	t1, .Lmemset_tail2
	sw	a1, 0(t0)
	addi	t0, t0, 4
.Lmemset_tail2:
	andi	t1, a2, 2
	beqz	t1, .Lmemset_tail1
	sh	a1, 0(t0)
	addi	t0, t0, 2
.Lmemset_tail1:
	andi	t1, a2, 1
	beqz	t1, .Lmemset_done
	sb	a1, 0(t0)
.Lmemset_done:
	ret

.Lmemset_bytes:
	beqz	a2, .Lmemset_done
.Lmemset_byte:
	sb	a1, 0(t0)
	addi	t0, t0, 1
	addi	a2, a2, -1
	bnez	a2, .Lmemset_byte
	ret
.Lmemset_end:
	.size	__wrap_memset, .Lmemset_end-__wrap_memset
