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
