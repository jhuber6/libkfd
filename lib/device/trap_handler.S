//===-- lib/device/trap_handler.S - Device trap handler ---------*- ASM -*-===//
//
// AMDGPU second-level trap handler
//
// Derived from the ROCr runtime trap handler (rocr-runtime), copyright Advanced
// Micro Devices, Inc., licensed under the University of Illinois/NCSA Open
// Source License.
//
// Handles software traps (s_trap) and hardware exceptions for all waves in the
// process. The kernel's first-level handler (CWSR) dispatches here with the
// TTMP ABI described below. We classify the event, send an MSG_INTERRUPT to
// KFD with the exception code and doorbell ID, then park and halt the wave so
// a debugger or the runtime can inspect it.
//
// No PC-sampling support host trap, stochastic perf traps are dismissed.
//
// Supported generations: GFX9 (incl. 9.4/9.5), GFX10.1, GFX10.3, GFX11,
//                        GFX12 (incl. 12.5), GFX13.
//
//===----------------------------------------------------------------------===//

//==============================================================================
// Generation detection
//==============================================================================

#if defined(__GFX13__)
#define GEN 13
#elif defined(__GFX12__)
#define GEN 12
#elif defined(__GFX11__)
#define GEN 11
#elif defined(__GFX10__)
#define GEN 10
#elif defined(__GFX9__)
#define GEN 9
#else
#error "Unsupported AMDGPU target"
#endif

// GFX9.4+ (gfx940/941/942/950): IB_STS saved in ttmp13, TTMP setup in ttmp11.
#if defined(__gfx9_4_generic__) || defined(__gfx940__) || defined(__gfx941__) || \
    defined(__gfx942__) || defined(__gfx943__) || defined(__gfx950__)
#define GFX94X 1
#else
#define GFX94X 0
#endif

// GFX10.1 (Navi10/12/14): IB_STS has extra REPLAY_W64H bit.
#if defined(__gfx10_1_generic__) || defined(__gfx1010__) || \
    defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1013__)
#define GFX101 1
#else
#define GFX101 0
#endif

// Wave parking strategy: save original PC into TTMPs and redirect to a spin
// loop. GFX9.4 and GFX10.3 just halt in place without parking.
#if GEN >= 12
#define PARK_TO_TTMP10 1
#define PARK_TO_TTMP7  0
#elif (GEN == 9 && !GFX94X) || GFX101 || GEN == 11
#define PARK_TO_TTMP10 0
#define PARK_TO_TTMP7  1
#else
#define PARK_TO_TTMP10 0
#define PARK_TO_TTMP7  0
#endif

//==============================================================================
// Constants
//==============================================================================

// Exception codes packed into m0 for MSG_INTERRUPT.
// Layout: [31:DOORBELL_ID_SIZE] = exception flags, [9:0] = doorbell ID.
.set DOORBELL_ID_SIZE,              10
.set DOORBELL_ID_MASK,              ((1 << DOORBELL_ID_SIZE) - 1)
.set EC_QUEUE_WAVE_ABORT_M0,        (1 << (DOORBELL_ID_SIZE + 0))
.set EC_QUEUE_WAVE_TRAP_M0,         (1 << (DOORBELL_ID_SIZE + 1))
.set EC_QUEUE_WAVE_MATH_ERROR_M0,   (1 << (DOORBELL_ID_SIZE + 2))
.set EC_QUEUE_WAVE_ILLEGAL_INST_M0, (1 << (DOORBELL_ID_SIZE + 3))
.set EC_QUEUE_WAVE_MEM_VIOL_M0,     (1 << (DOORBELL_ID_SIZE + 4))
.set EC_QUEUE_WAVE_APERTURE_M0,     (1 << (DOORBELL_ID_SIZE + 5))

// Trap IDs from s_trap instructions.
.set TRAP_ID_ABORT,                 2
.set TRAP_ID_DEBUGTRAP,             3

// TTMP6: saved trap state (all generations).
.set TTMP6_WAVE_STOPPED_SHIFT,      30
.set TTMP6_SAVED_STATUS_HALT_SHIFT, 29
.set TTMP6_SAVED_STATUS_HALT_MASK,  (1 << 29)
.set TTMP6_SAVED_TRAP_ID_SHIFT,     25
.set TTMP6_SAVED_TRAP_ID_MASK,      (0xF << 25)
.set TTMP6_SAVED_TRAP_ID_BFE,       (25 | (4 << 16))

// PC_HI address mask and storage shift.
.set PC_HI_ADDR_MASK,               0xFFFF
.set TTMP_PC_HI_SHIFT,              7

// Debug enabled flag in TTMP11 (or TTMP13 on GFX9.4+).
.set TTMP_DEBUG_ENABLED_SHIFT,      23

#if GEN <= 11
// ---- GFX9 / GFX10 / GFX11 constants ----

// Trap ID in ttmp1[23:16], 8-bit field.
.set PC_HI_TRAP_ID_BFE,            (16 | (8 << 16))
.set PC_HI_HT_SHIFT,               24

// TRAPSTS exception bits.
.set TRAPSTS_MEM_VIOL_SHIFT,        8
.set TRAPSTS_ILLEGAL_INST_SHIFT,    11
.set TRAPSTS_XNACK_ERROR_SHIFT,     28
.set TRAPSTS_MATH_EXCP_MASK,        0x7F

// MODE register math exception enable.
.set MODE_EXCP_EN_SHIFT,            12

// STATUS register.
.set STATUS_HALT_SHIFT,             13
.set STATUS_HALT_BFE,               (13 | (1 << 16))
.set STATUS_SKIP_EXPORT_SHIFT,      18

#if GEN == 9
#if GFX94X
.set TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT, 26
.set IB_STS_FIRST_REPLAY_SHIFT,     15
.set IB_STS_RCNT_FIRST_REPLAY_MASK, 0x1F8000
.set TTMP11_TTMPS_SETUP_SHIFT,      31
.set TTMP6_SPI_TTMPS_DISABLED_SHIFT, 31
.set TRAPSTS_PERF_SNAPSHOT_SHIFT,   26
.set TRAPSTS_HOST_TRAP_SHIFT,       22
#else
.set TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT, 26
.set IB_STS_FIRST_REPLAY_SHIFT,     15
.set IB_STS_RCNT_FIRST_REPLAY_MASK, 0x1F8000
#endif
#elif GFX101
.set TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT, 24
.set TTMP_SAVE_REPLAY_W64H_SHIFT,  31
.set IB_STS_FIRST_REPLAY_SHIFT,     15
.set IB_STS_REPLAY_W64H_SHIFT,      25
.set IB_STS_RCNT_FIRST_REPLAY_MASK, 0x3F8000
.set IB_STS_REPLAY_W64H_MASK,       0x2000000
#endif

#else // GEN >= 12
// ---- GFX12+ constants ----

// Trap ID in ttmp1[31:28], 4-bit field.
.set PC_HI_TRAP_ID_BFE,            (28 | (4 << 16))

// EXCP_FLAG_PRIV bits.
.set EXCP_PRIV_ADDR_WATCH_MASK,     ((1 << 4) - 1)
.set EXCP_PRIV_MEMVIOL_SHIFT,       4
.set EXCP_PRIV_ILLEGAL_INST_SHIFT,  6
.set EXCP_PRIV_HT_SHIFT,           7
.set EXCP_PRIV_WAVE_START_SHIFT,    8
.set EXCP_PRIV_WAVE_END_SHIFT,      9
.set EXCP_PRIV_PERF_SNAPSHOT_SHIFT, 10
.set EXCP_PRIV_TRAP_AFTER_SHIFT,    11
.set EXCP_PRIV_XNACK_ERROR_SHIFT,  12

// EXCP_FLAG_USER math exception.
.set EXCP_USER_MATH_SHIFT,          0
.set EXCP_USER_MATH_SIZE,           7

// TRAP_CTRL bits.
.set TRAP_CTRL_MATH_EXCP_MASK,      ((1 << 7) - 1)
.set TRAP_CTRL_ADDR_WATCH_SHIFT,    7
.set TRAP_CTRL_WAVE_END_SHIFT,      8
.set TRAP_CTRL_TRAP_AFTER_SHIFT,    9

// STATE_PRIV register.
.set STATE_PRIV_HALT_SHIFT,         14
.set STATE_PRIV_HALT_BFE,           (14 | (1 << 16))
.set STATE_PRIV_BARRIER_COMPLETE,   2

// Debug / TTMP flags.
.set TTMP8_DEBUG_FLAG_SHIFT,        31

// SCHED_MODE (gfx12.0 only - guarded by assembler .if at point of use).
.set TTMP1_SCHED_MODE_MASK,         0xC000000
.set TTMP11_SCHED_MODE_MASK,        0xC000000
.set TTMP11_SCHED_MODE_BFE,         (26 | (2 << 16))

#endif // GEN

//===----------------------------------------------------------------------===//
// Trap handler entry point
//===----------------------------------------------------------------------===//

// ABI from the first-level trap handler (hardware/firmware):
//   ttmp0       = PC[31:0]
//   ttmp1       = {metadata} | PC[47:32]    (metadata layout varies by gen)
//   ttmp12      = SQ_WAVE_STATUS (gfx9-11) / SQ_WAVE_STATE_PRIV (gfx12+)
//   ttmp[14:15] = TMA (Trap Memory Argument) pointer
//
// gfx9:   ttmp8/9/10 = WorkgroupId X/Y/Z
// gfx12+: ttmp9 = WgIdX, ttmp7 = {WgIdZ[15:0], WgIdY[15:0]}
//
// Register budget: ttmp[2:5], ttmp7 (gfx9), ttmp13 are scratch.
// Must preserve: ttmp[14:15] (TMA), ttmp6 (flags), ttmp[8:11] (varies).

.globl trap_entry
.type trap_entry, @function
trap_entry:

#if GEN == 12
    // gfx12.0: save SCHED_MODE from ttmp1 into ttmp11 for later restore.
    // GFX12.5+ and GFX13 do not use SCHED_MODE.
.if .amdgcn.gfx_generation_minor == 0
    s_andn2_b32      ttmp11, ttmp11, TTMP11_SCHED_MODE_MASK
    s_and_b32        ttmp2,  ttmp1,  TTMP1_SCHED_MODE_MASK
    s_or_b32         ttmp11, ttmp11, ttmp2
.endif
#endif

    // ttmp3 accumulates exception flags throughout the handler.
    s_mov_b32        ttmp3, 0

    //===----------------------------------------------------------------------===//
    // Dismiss PC-sampling traps (host trap / stochastic)
    //===----------------------------------------------------------------------===//

#if GEN >= 12
    s_getreg_b32     ttmp2, hwreg(HW_REG_EXCP_FLAG_PRIV)

    s_bitcmp1_b32    ttmp2, EXCP_PRIV_HT_SHIFT
    s_cbranch_scc0   .check_stochastic
    s_setreg_imm32_b32 hwreg(HW_REG_EXCP_FLAG_PRIV, EXCP_PRIV_HT_SHIFT, 1), 0
    s_branch         .exit_trap

.check_stochastic:
    s_bitcmp1_b32    ttmp2, EXCP_PRIV_PERF_SNAPSHOT_SHIFT
    s_cbranch_scc0   .classify_trap
    s_setreg_imm32_b32 hwreg(HW_REG_EXCP_FLAG_PRIV, EXCP_PRIV_PERF_SNAPSHOT_SHIFT, 1), 0
    s_branch         .exit_trap

.classify_trap:
#endif

    //===----------------------------------------------------------------------===//
    // Classify: s_trap vs hardware exception
    //===----------------------------------------------------------------------===//

    s_bfe_u32        ttmp2, ttmp1, PC_HI_TRAP_ID_BFE
#if GEN == 9 && GFX94X
    // GFX9.4+: trap_id==0 might be a stochastic perf trap; check first.
    s_cbranch_scc0   .trap_id_zero
#else
    s_cbranch_scc0   .check_exceptions
#endif

#if GEN <= 11
    // GFX9/10/11: host trap is indicated by a bit in ttmp1.
    // Dismiss it - we don't do PC sampling.
    s_bitcmp1_b32    ttmp1, PC_HI_HT_SHIFT
#if GEN == 9 && GFX94X
    s_cbranch_scc0   .not_host_trap
    s_setreg_imm32_b32 hwreg(HW_REG_TRAPSTS, TRAPSTS_HOST_TRAP_SHIFT, 1), 0
    s_branch         .exit_trap
.not_host_trap:
#else
    s_cbranch_scc1   .exit_trap
#endif
#endif

    // It's an s_trap instruction - advance PC past it.
    s_add_u32        ttmp0, ttmp0, 0x4
    s_addc_u32       ttmp1, ttmp1, 0x0

    // --- Debug trap (trap_id == 3) ---
    // If debugger is not attached, fall through to check for any concurrent
    // hardware exceptions before exiting.
    s_cmp_eq_u32     ttmp2, TRAP_ID_DEBUGTRAP
    s_cbranch_scc0   .not_debugtrap
#if GEN == 9 && GFX94X
    s_bitcmp1_b32    ttmp13, TTMP_DEBUG_ENABLED_SHIFT
#else
    s_bitcmp1_b32    ttmp11, TTMP_DEBUG_ENABLED_SHIFT
#endif
    s_cbranch_scc0   .check_exceptions
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0
    s_branch         .check_exceptions

    // --- Abort trap (trap_id == 2) ---
.not_debugtrap:
    s_cmp_eq_u32     ttmp2, TRAP_ID_ABORT
    s_cbranch_scc0   .not_abort
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_ABORT_M0
    s_branch         .check_exceptions

    // --- Any other s_trap: generic trap ---
.not_abort:
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

    //===----------------------------------------------------------------------===//
    // Check hardware exceptions
    //===----------------------------------------------------------------------===//

#if GEN == 9 && GFX94X
    // GFX9.4+: trap_id == 0 could be a stochastic perf trap.
.trap_id_zero:
    s_getreg_b32     ttmp7, hwreg(HW_REG_TRAPSTS)
    s_bitcmp1_b32    ttmp7, TRAPSTS_PERF_SNAPSHOT_SHIFT
    s_cbranch_scc0   .check_exceptions
    s_setreg_imm32_b32 hwreg(HW_REG_TRAPSTS, TRAPSTS_PERF_SNAPSHOT_SHIFT, 1), 0
    s_branch         .exit_trap
#endif

.check_exceptions:

#if GEN <= 11
    // GFX9/10/11: exceptions reported via TRAPSTS register.
    s_getreg_b32     ttmp2, hwreg(HW_REG_TRAPSTS)

    s_bitcmp1_b32    ttmp2, TRAPSTS_XNACK_ERROR_SHIFT
    s_cbranch_scc0   .not_mem_viol
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_MEM_VIOL_M0
    s_branch         .not_aperture_viol

.not_mem_viol:
    s_bitcmp1_b32    ttmp2, TRAPSTS_MEM_VIOL_SHIFT
    s_cbranch_scc0   .not_aperture_viol
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_APERTURE_M0

.not_aperture_viol:
    s_bitcmp1_b32    ttmp2, TRAPSTS_ILLEGAL_INST_SHIFT
    s_cbranch_scc0   .not_illegal
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_ILLEGAL_INST_M0

.not_illegal:
    s_and_b32        ttmp2, ttmp2, TRAPSTS_MATH_EXCP_MASK
    s_cbranch_scc0   .not_math
    s_getreg_b32     ttmp7, hwreg(HW_REG_MODE)
    s_lshl_b32       ttmp2, ttmp2, MODE_EXCP_EN_SHIFT
    s_and_b32        ttmp2, ttmp2, ttmp7
    s_cbranch_scc0   .not_math
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_MATH_ERROR_M0

.not_math:

#else // GEN >= 12
    // GFX12+: exceptions split across EXCP_FLAG_PRIV and TRAP_CTRL.
    s_getreg_b32     ttmp2,  hwreg(HW_REG_EXCP_FLAG_PRIV)
    s_getreg_b32     ttmp13, hwreg(HW_REG_TRAP_CTRL)

    s_bitcmp1_b32    ttmp2, EXCP_PRIV_XNACK_ERROR_SHIFT
    s_cbranch_scc0   .not_mem_viol
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_MEM_VIOL_M0
    s_branch         .not_aperture_viol

.not_mem_viol:
    s_bitcmp1_b32    ttmp2, EXCP_PRIV_MEMVIOL_SHIFT
    s_cbranch_scc0   .not_aperture_viol
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_APERTURE_M0

.not_aperture_viol:
    s_bitcmp1_b32    ttmp2, EXCP_PRIV_ILLEGAL_INST_SHIFT
    s_cbranch_scc0   .not_illegal
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_ILLEGAL_INST_M0

.not_illegal:
    s_bitcmp1_b32    ttmp2, EXCP_PRIV_WAVE_START_SHIFT
    s_cbranch_scc0   .not_wave_start
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_wave_start:
    s_bitcmp1_b32    ttmp2, EXCP_PRIV_WAVE_END_SHIFT
    s_cbranch_scc0   .not_wave_end
    s_bitcmp1_b32    ttmp13, TRAP_CTRL_WAVE_END_SHIFT
    s_cbranch_scc0   .not_wave_end
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_wave_end:
    s_bitcmp1_b32    ttmp13, TRAP_CTRL_TRAP_AFTER_SHIFT
    s_cbranch_scc0   .not_trap_after
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_trap_after:
    s_and_b32        ttmp2, ttmp2, EXCP_PRIV_ADDR_WATCH_MASK
    s_cbranch_scc0   .not_addr_watch
    s_bitcmp1_b32    ttmp13, TRAP_CTRL_ADDR_WATCH_SHIFT
    s_cbranch_scc0   .not_addr_watch
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_TRAP_M0

.not_addr_watch:
    s_getreg_b32     ttmp2, hwreg(HW_REG_EXCP_FLAG_USER, EXCP_USER_MATH_SHIFT, EXCP_USER_MATH_SIZE)
    s_and_b32        ttmp13, ttmp13, TRAP_CTRL_MATH_EXCP_MASK
    s_and_b32        ttmp2, ttmp2, ttmp13
    s_cbranch_scc0   .not_math
    s_or_b32         ttmp3, ttmp3, EC_QUEUE_WAVE_MATH_ERROR_M0

.not_math:
#endif // GEN

    // If nothing was flagged, return to user code.
    s_cmp_eq_u32     ttmp3, 0
    s_cbranch_scc1   .exit_trap

    //===----------------------------------------------------------------------===//
    // Send interrupt to KFD.
    //===----------------------------------------------------------------------===//

    // Fetch the doorbell ID that identifies our queue.
#if GEN >= 11
    s_sendmsg_rtn_b32 ttmp2, sendmsg(MSG_RTN_GET_DOORBELL)
#if GEN >= 12
    s_wait_kmcnt     0
#else
    s_waitcnt        lgkmcnt(0)
#endif
    s_and_b32        ttmp2, ttmp2, DOORBELL_ID_MASK
#else
    // GFX9/10: doorbell fetch requires clobbering EXEC and spinning.
    s_mov_b32        ttmp2, exec_lo
    s_mov_b32        ttmp4, exec_hi
    s_mov_b32        exec_lo, 0x80000000
    s_sendmsg        sendmsg(MSG_GET_DOORBELL)
.wait_doorbell:
    s_nop            0x7
    s_bitcmp0_b32    exec_lo, 0x1F
    s_cbranch_scc0   .wait_doorbell
    s_mov_b32        exec_hi, ttmp4
    s_and_b32        ttmp4, exec_lo, DOORBELL_ID_MASK
    s_mov_b32        exec_lo, ttmp2
    s_mov_b32        ttmp2, ttmp4
#endif
    s_or_b32         ttmp3, ttmp2, ttmp3

    // Save trap ID and halt status in ttmp6 for the debugger.
    s_andn2_b32      ttmp6, ttmp6, (TTMP6_SAVED_TRAP_ID_MASK | TTMP6_SAVED_STATUS_HALT_MASK)
    s_bfe_u32        ttmp2, ttmp1, PC_HI_TRAP_ID_BFE
    s_min_u32        ttmp2, ttmp2, 0xF
    s_lshl_b32       ttmp2, ttmp2, TTMP6_SAVED_TRAP_ID_SHIFT
    s_or_b32         ttmp6, ttmp6, ttmp2
#if GEN <= 11
    s_bfe_u32        ttmp2, ttmp12, STATUS_HALT_BFE
#else
    s_bfe_u32        ttmp2, ttmp12, STATE_PRIV_HALT_BFE
#endif
    s_lshl_b32       ttmp2, ttmp2, TTMP6_SAVED_STATUS_HALT_SHIFT
    s_or_b32         ttmp6, ttmp6, ttmp2

    // Fire MSG_INTERRUPT: m0 = (exception_flags << 10) | doorbell_id
    s_mov_b32        ttmp2, m0
    s_mov_b32        m0, ttmp3
    s_nop            0x0
    s_sendmsg        sendmsg(MSG_INTERRUPT)
#if GEN >= 12
    s_wait_kmcnt     0
#else
    s_waitcnt        lgkmcnt(0)
#endif
    s_mov_b32        m0, ttmp2

    //===----------------------------------------------------------------------===//
    // Park the wave.
    //===----------------------------------------------------------------------===//

    // Save the original PC into preserved TTMPs so a debugger can find
    // it, then redirect execution to a spin loop.

#if PARK_TO_TTMP7
    // GFX9 (non-9.4) / GFX10.1 / GFX11: save PC to ttmp7 + ttmp11.
    s_mov_b32        ttmp7, ttmp0
    s_and_b32        ttmp1, ttmp1, PC_HI_ADDR_MASK
    s_lshl_b32       ttmp1, ttmp1, TTMP_PC_HI_SHIFT
    s_andn2_b32      ttmp11, ttmp11, (PC_HI_ADDR_MASK << TTMP_PC_HI_SHIFT)
    s_or_b32         ttmp11, ttmp11, ttmp1
    s_getpc_b64      [ttmp0, ttmp1]
    s_add_u32        ttmp0, ttmp0, .parked - .
    s_addc_u32       ttmp1, ttmp1, 0x0
#elif PARK_TO_TTMP10
    // GFX12+: save PC to ttmp10 + ttmp11.
    s_mov_b32        ttmp10, ttmp0
    s_and_b32        ttmp1, ttmp1, PC_HI_ADDR_MASK
    s_lshl_b32       ttmp1, ttmp1, TTMP_PC_HI_SHIFT
    s_andn2_b32      ttmp11, ttmp11, (PC_HI_ADDR_MASK << TTMP_PC_HI_SHIFT)
    s_or_b32         ttmp11, ttmp11, ttmp1
    s_getpc_b64      [ttmp0, ttmp1]
    s_add_u32        ttmp0, ttmp0, .parked - .
    s_addc_u32       ttmp1, ttmp1, 0x0
#endif
    // GFX9.4 / GFX10.3: no parking - wave halts at the current PC.

    //===----------------------------------------------------------------------===//
    // Halt the wave.
    //===----------------------------------------------------------------------===//

    s_bitset1_b32    ttmp6, TTMP6_WAVE_STOPPED_SHIFT
#if GEN <= 11
    s_bitset1_b32    ttmp12, STATUS_HALT_SHIFT
    // SKIP_EXPORT tells the debugger the trap handler intentionally halted.
    s_bitset1_b32    ttmp12, STATUS_SKIP_EXPORT_SHIFT
#else
    s_bitset1_b32    ttmp12, STATE_PRIV_HALT_SHIFT
#endif

    // Initialize preserved TTMP registers on first trap entry.
    // The debugger reads ttmp4/5 for saved state; they must be zeroed
    // before first use so stale data doesn't confuse it.
#if GEN == 9 && GFX94X
    s_bitcmp1_b32    ttmp11, TTMP11_TTMPS_SETUP_SHIFT
    s_cbranch_scc1   .ttmps_initialized
    s_mov_b32        ttmp4, 0
    s_mov_b32        ttmp5, 0
    s_bitset0_b32    ttmp6, TTMP6_SPI_TTMPS_DISABLED_SHIFT
    s_bitset1_b32    ttmp11, TTMP11_TTMPS_SETUP_SHIFT
.ttmps_initialized:
#elif GEN >= 12
    s_bitcmp1_b32    ttmp8, TTMP8_DEBUG_FLAG_SHIFT
    s_cbranch_scc1   .ttmps_initialized
    s_mov_b32        ttmp4, 0
    s_mov_b32        ttmp5, 0
    s_bitset1_b32    ttmp8, TTMP8_DEBUG_FLAG_SHIFT
.ttmps_initialized:
#endif

    //===----------------------------------------------------------------------===//
    // Exit: restore hardware state and return from exception.
    //===----------------------------------------------------------------------===//

.exit_trap:

    // Restore IB_STS replay counters (GFX9 / GFX10.1 only).
#if GEN == 9 && !GFX94X
    s_lshr_b32       ttmp2, ttmp11, (TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT - IB_STS_FIRST_REPLAY_SHIFT)
    s_and_b32        ttmp2, ttmp2, IB_STS_RCNT_FIRST_REPLAY_MASK
    s_setreg_b32     hwreg(HW_REG_IB_STS), ttmp2
#elif GEN == 9 && GFX94X
    s_lshr_b32       ttmp2, ttmp13, (TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT - IB_STS_FIRST_REPLAY_SHIFT)
    s_and_b32        ttmp2, ttmp2, IB_STS_RCNT_FIRST_REPLAY_MASK
    s_setreg_b32     hwreg(HW_REG_IB_STS), ttmp2
#elif GFX101
    s_lshr_b32       ttmp2, ttmp11, (TTMP_SAVE_RCNT_FIRST_REPLAY_SHIFT - IB_STS_FIRST_REPLAY_SHIFT)
    s_and_b32        ttmp4, ttmp2, IB_STS_RCNT_FIRST_REPLAY_MASK
    s_lshr_b32       ttmp2, ttmp11, (TTMP_SAVE_REPLAY_W64H_SHIFT - IB_STS_REPLAY_W64H_SHIFT)
    s_and_b32        ttmp2, ttmp2, IB_STS_REPLAY_W64H_MASK
    s_or_b32         ttmp2, ttmp2, ttmp4
    s_setreg_b32     hwreg(HW_REG_IB_STS), ttmp2
#endif

#if GEN == 12
    // Restore SCHED_MODE (gfx12.0 only).
.if .amdgcn.gfx_generation_minor == 0
    s_bfe_u32        ttmp2, ttmp11, TTMP11_SCHED_MODE_BFE
    s_setreg_b32     hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), ttmp2
.endif
#endif

    // Restore STATUS / STATE_PRIV.
    s_and_b64        exec, exec, exec
    s_and_b64        vcc, vcc, vcc
#if GEN <= 11
    s_setreg_b32     hwreg(HW_REG_STATUS), ttmp12
#else
    // GFX12+: STATE_PRIV must be written in two halves to avoid clobbering
    // the read-only BARRIER_COMPLETE bit (bit 2).
    s_setreg_b32     hwreg(HW_REG_STATE_PRIV, 0, STATE_PRIV_BARRIER_COMPLETE), ttmp12
    s_lshr_b32       ttmp12, ttmp12, (STATE_PRIV_BARRIER_COMPLETE + 1)
    s_setreg_b32     hwreg(HW_REG_STATE_PRIV, STATE_PRIV_BARRIER_COMPLETE + 1, 32 - STATE_PRIV_BARRIER_COMPLETE - 1), ttmp12
#endif

    s_rfe_b64        [ttmp0, ttmp1]

    //===----------------------------------------------------------------------===//
    // Parked loop - wave spins here after being halted
    //===----------------------------------------------------------------------===//
.parked:
    s_trap           0x2
    s_branch         .parked

    // Pad with s_code_end so the instruction cache prefetcher always
    // has valid data to read past the end of the handler.
#if GEN >= 11
    .p2align 6
    .rept 64
      s_code_end
    .endr
#endif
