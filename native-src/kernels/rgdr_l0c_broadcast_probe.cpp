// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void rgdr_v310_l0c_broadcast_probe(GM_ADDR input, GM_ADDR output)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    constexpr uint32_t LANES = 16;
    constexpr uint32_t TILE = LANES * LANES;
    TPipe pipe;
    TBuf<TPosition::VECCALC> sourceBuf;
    TQue<TPosition::CO1, 1> l0cQueue;
    TQue<TPosition::CO2, 1> ubQueue;
    pipe.InitBuffer(sourceBuf, LANES * sizeof(half));
    pipe.InitBuffer(l0cQueue, 1, TILE * sizeof(half));
    pipe.InitBuffer(ubQueue, 1, TILE * sizeof(half));

    GlobalTensor<half> inGm;
    GlobalTensor<half> outGm;
    inGm.SetGlobalBuffer((__gm__ half*)input, LANES);
    outGm.SetGlobalBuffer((__gm__ half*)output, TILE);
    auto source = sourceBuf.Get<half>();
    auto accumulator = l0cQueue.AllocTensor<half>();
    DataCopy(source, inGm, LANES);
    // This probe fences all pipes so it can characterize the destination
    // layout independently of the instruction's pipeline assignment.
    PipeBarrier<PIPE_ALL>();
    broadcast_ub_to_cc((__cc__ half*)accumulator.GetPhyAddr(),
                       (__ubuf__ half*)source.GetPhyAddr(),
                       uint8_t{1}, uint8_t{1}, uint8_t{0}, uint8_t{0});
    PipeBarrier<PIPE_ALL>();
    l0cQueue.EnQue(accumulator);
    accumulator = l0cQueue.DeQue<half>();

    auto out = ubQueue.AllocTensor<half>();
    DataCopyEnhancedParams drain{};
    drain.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(out, accumulator, DataCopyParams{1, 1, 0, 0}, drain);
    ubQueue.EnQue(out);
    out = ubQueue.DeQue<half>();
    PipeBarrier<PIPE_ALL>();
    DataCopy(outGm, out, TILE);
    PipeBarrier<PIPE_ALL>();
    ubQueue.FreeTensor(out);
    l0cQueue.FreeTensor(accumulator);
}

namespace {
constexpr uint32_t C0 = 16;
constexpr uint32_t ROWS = 128;
constexpr uint32_t REDUCTION = 144;
constexpr uint32_t TILE = C0 * C0;
constexpr uint32_t RESULT = ROWS * C0;
constexpr uint32_t A_ELEMENTS = ROWS * REDUCTION;
constexpr uint32_t B_ELEMENTS = REDUCTION * C0;

enum Probe { GATHER, BROADCAST_RAW, BROADCAST, MMAD, SERIAL, INTERLEAVED };

__aicore__ inline void Drain(LocalTensor<half> dst, LocalTensor<half> src)
{
    DataCopyEnhancedParams drain{};
    drain.blockMode = BlockMode::BLOCK_MODE_MATRIX;
    DataCopy(dst, src, DataCopyParams{1, ROWS / C0, 0, 0}, drain);
}

__aicore__ inline void ReadExpandedDelta(LocalTensor<half> dst, LocalTensor<half> scratch,
                                        LocalTensor<half> l0c)
{
    Drain(scratch, l0c);
    PipeBarrier<PIPE_ALL>();
    for (uint32_t tile = 0; tile < ROWS / C0; ++tile) {
        Transpose(dst[tile * TILE], scratch[tile * TILE]);
    }
}

template <Probe MODE>
__aicore__ inline void Run(GM_ADDR delta, GM_ADDR offsets, GM_ADDR lhs, GM_ADDR rhs,
                            GM_ADDR expandedOut, GM_ADDR matrixOut, uint32_t repeats)
{
    constexpr bool HAS_BROADCAST = MODE != GATHER && MODE != MMAD;
    constexpr bool HAS_MMAD = MODE == MMAD || MODE == SERIAL || MODE == INTERLEAVED;
    TPipe pipe;
    TBuf<TPosition::VECCALC> deltaBuf, offsetsBuf, expandedBuf, drainBuf, matBuf;
    TBuf<TPosition::A1> a1Buf;
    TBuf<TPosition::B1> b1Buf;
    TQue<TPosition::A2, 1> a2Queue;
    TQue<TPosition::B2, 1> b2Queue;
    TQue<TPosition::CO1, 1> l0cQueue;

    LocalTensor<half> values, expanded, scratch, matrix, l0c, a2, b2;
    LocalTensor<uint32_t> indices;
    if constexpr (MODE != MMAD) {
        pipe.InitBuffer(deltaBuf, ROWS * sizeof(half));
        pipe.InitBuffer(expandedBuf, RESULT * sizeof(half));
        GlobalTensor<half> input;
        input.SetGlobalBuffer((__gm__ half*)delta, ROWS);
        values = deltaBuf.Get<half>();
        expanded = expandedBuf.Get<half>();
        DataCopy(values, input, ROWS);
    }
    if constexpr (MODE == GATHER) {
        pipe.InitBuffer(offsetsBuf, RESULT * sizeof(uint32_t));
        GlobalTensor<uint32_t> indexGm;
        indexGm.SetGlobalBuffer((__gm__ uint32_t*)offsets, RESULT);
        indices = offsetsBuf.Get<uint32_t>();
        DataCopy(indices, indexGm, RESULT);
    } else {
        // Each half is 4096 B. Both L0C starts meet the 256-element alignment.
        pipe.InitBuffer(l0cQueue, 1, 2 * RESULT * sizeof(half));
        l0c = l0cQueue.AllocTensor<half>();
    }
    if constexpr (HAS_BROADCAST) {
        pipe.InitBuffer(drainBuf, RESULT * sizeof(half));
        scratch = drainBuf.Get<half>();
    }
    if constexpr (HAS_MMAD) {
        pipe.InitBuffer(a1Buf, A_ELEMENTS * sizeof(half));
        pipe.InitBuffer(b1Buf, B_ELEMENTS * sizeof(half));
        pipe.InitBuffer(a2Queue, 1, A_ELEMENTS * sizeof(half));
        pipe.InitBuffer(b2Queue, 1, B_ELEMENTS * sizeof(half));
        pipe.InitBuffer(matBuf, RESULT * sizeof(half));
        auto a1 = a1Buf.Get<half>();
        auto b1 = b1Buf.Get<half>();
        matrix = matBuf.Get<half>();
        GlobalTensor<half> aGm;
        GlobalTensor<half> bGm;
        aGm.SetGlobalBuffer((__gm__ half*)lhs, A_ELEMENTS);
        bGm.SetGlobalBuffer((__gm__ half*)rhs, B_ELEMENTS);
        DataCopy(a1, aGm, A_ELEMENTS);
        DataCopy(b1, bGm, B_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        a2 = a2Queue.AllocTensor<half>();
        b2 = b2Queue.AllocTensor<half>();
        LoadData2DParams load{};
        load.srcStride = 1;
        load.repeatTimes = A_ELEMENTS / TILE;
        LoadData(a2, a1, load);
        load.repeatTimes = B_ELEMENTS / TILE;
        LoadData(b2, b1, load);
    }
    PipeBarrier<PIPE_ALL>();

    MmadParams mad{};
    mad.m = ROWS;
    mad.n = C0;
    mad.k = REDUCTION;
    mad.cmatrixInitVal = true;
    for (uint32_t i = 0; i < repeats; ++i) {
        if constexpr (HAS_MMAD) {
            Mmad(l0c[RESULT], a2, b2, mad);
        }
        if constexpr (MODE == SERIAL) {
            PipeBarrier<PIPE_ALL>();
        }
        if constexpr (HAS_BROADCAST) {
            // One burst of eight 16-element chunks produces eight L0C tiles.
            broadcast_ub_to_cc((__cc__ half*)l0c.GetPhyAddr(),
                               (__ubuf__ half*)values.GetPhyAddr(),
                               uint8_t{1}, uint8_t{ROWS / C0}, uint8_t{0}, uint8_t{0});
        }
        if constexpr (MODE == GATHER) {
            Gather(expanded, values, indices, 0, RESULT);
        }
        PipeBarrier<PIPE_ALL>();
        if constexpr (HAS_BROADCAST && MODE != BROADCAST_RAW) {
            ReadExpandedDelta(expanded, scratch, l0c);
        }
        if constexpr (HAS_MMAD) {
            Drain(matrix, l0c[RESULT]);
        }
        if constexpr (HAS_MMAD || (HAS_BROADCAST && MODE != BROADCAST_RAW)) {
            PipeBarrier<PIPE_ALL>();
        }
    }

    // The raw case drains only once, so varying repeats amortizes that cost.
    if constexpr (MODE == BROADCAST_RAW) {
        ReadExpandedDelta(expanded, scratch, l0c);
        PipeBarrier<PIPE_ALL>();
    }
    if constexpr (MODE != MMAD) {
        GlobalTensor<half> out;
        out.SetGlobalBuffer((__gm__ half*)expandedOut, RESULT);
        DataCopy(out, expanded, RESULT);
    }
    if constexpr (HAS_MMAD) {
        GlobalTensor<half> out;
        out.SetGlobalBuffer((__gm__ half*)matrixOut, RESULT);
        DataCopy(out, matrix, RESULT);
    }
    PipeBarrier<PIPE_ALL>();
    if constexpr (HAS_MMAD) {
        a2Queue.FreeTensor(a2);
        b2Queue.FreeTensor(b2);
    }
    if constexpr (MODE != GATHER) {
        l0cQueue.FreeTensor(l0c);
    }
}
} // namespace

extern "C" __global__ __aicore__ void rgdr_v310_l0c_gather(
    GM_ADDR delta, GM_ADDR offsets, GM_ADDR lhs, GM_ADDR rhs,
    GM_ADDR expandedOut, GM_ADDR matrixOut, uint32_t repeats)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    Run<GATHER>(delta, offsets, lhs, rhs, expandedOut, matrixOut, repeats);
}

extern "C" __global__ __aicore__ void rgdr_v310_l0c_broadcast_raw(
    GM_ADDR delta, GM_ADDR offsets, GM_ADDR lhs, GM_ADDR rhs,
    GM_ADDR expandedOut, GM_ADDR matrixOut, uint32_t repeats)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    Run<BROADCAST_RAW>(delta, offsets, lhs, rhs, expandedOut, matrixOut, repeats);
}

extern "C" __global__ __aicore__ void rgdr_v310_l0c_broadcast(
    GM_ADDR delta, GM_ADDR offsets, GM_ADDR lhs, GM_ADDR rhs,
    GM_ADDR expandedOut, GM_ADDR matrixOut, uint32_t repeats)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    Run<BROADCAST>(delta, offsets, lhs, rhs, expandedOut, matrixOut, repeats);
}

extern "C" __global__ __aicore__ void rgdr_v310_l0c_mmad(
    GM_ADDR delta, GM_ADDR offsets, GM_ADDR lhs, GM_ADDR rhs,
    GM_ADDR expandedOut, GM_ADDR matrixOut, uint32_t repeats)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    Run<MMAD>(delta, offsets, lhs, rhs, expandedOut, matrixOut, repeats);
}

extern "C" __global__ __aicore__ void rgdr_v310_l0c_serial(
    GM_ADDR delta, GM_ADDR offsets, GM_ADDR lhs, GM_ADDR rhs,
    GM_ADDR expandedOut, GM_ADDR matrixOut, uint32_t repeats)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    Run<SERIAL>(delta, offsets, lhs, rhs, expandedOut, matrixOut, repeats);
}

extern "C" __global__ __aicore__ void rgdr_v310_l0c_interleaved(
    GM_ADDR delta, GM_ADDR offsets, GM_ADDR lhs, GM_ADDR rhs,
    GM_ADDR expandedOut, GM_ADDR matrixOut, uint32_t repeats)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    Run<INTERLEAVED>(delta, offsets, lhs, rhs, expandedOut, matrixOut, repeats);
}
