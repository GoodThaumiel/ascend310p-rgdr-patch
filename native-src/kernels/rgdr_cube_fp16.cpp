// SPDX-License-Identifier: Apache-2.0
// Uniform MTP windows on the coupled 310P AI Core.
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t K = 128;
constexpr uint32_t V = 128;
constexpr uint32_t HK = 8;
constexpr uint32_t HV = 16;
constexpr uint32_t TILE = 16;
constexpr uint32_t FRACTAL = TILE * TILE;
constexpr uint32_t K_BLOCKS = K / TILE + 1;
constexpr uint32_t STATE_TILE = TILE * K;

template <HardEvent Event>
__aicore__ inline void Sync()
{
    const auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

class CubeFp16RGDR {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR gate,
                                GM_ADDR state, GM_ADDR indices, GM_ADDR accepted, GM_ADDR out,
                                uint32_t batch, uint32_t window, uint32_t slots, float scale,
                                bool deq)
    {
        q_.SetGlobalBuffer((__gm__ half*)q);
        k_.SetGlobalBuffer((__gm__ half*)k);
        v_.SetGlobalBuffer((__gm__ half*)v);
        beta_.SetGlobalBuffer((__gm__ half*)beta);
        gate_.SetGlobalBuffer((__gm__ half*)gate);
        state_.SetGlobalBuffer((__gm__ half*)state);
        indices_.SetGlobalBuffer((__gm__ int32_t*)indices);
        accepted_.SetGlobalBuffer((__gm__ int32_t*)accepted);
        out_.SetGlobalBuffer((__gm__ half*)out);
        batch_ = batch;
        window_ = window;
        slots_ = slots;
        scale_ = static_cast<half>(scale);
        deq_ = deq;

        pipe_.InitBuffer(a1_, K_BLOCKS * FRACTAL * sizeof(half));
        pipe_.InitBuffer(b1_, K_BLOCKS * FRACTAL * sizeof(half));
        pipe_.InitBuffer(a2_, 1, K_BLOCKS * FRACTAL * sizeof(half));
        pipe_.InitBuffer(b2_, 1, K_BLOCKS * FRACTAL * sizeof(half));
        pipe_.InitBuffer(c1_, 1, FRACTAL * sizeof(half));
        pipe_.InitBuffer(c2_, 1, FRACTAL * sizeof(half));
        pipe_.InitBuffer(stateTile_, STATE_TILE * sizeof(half));
        pipe_.InitBuffer(update_, STATE_TILE * sizeof(half));
        pipe_.InitBuffer(qk_, 3 * K * sizeof(half));
        pipe_.InitBuffer(rhs_, K_BLOCKS * FRACTAL * sizeof(half));
        pipe_.InitBuffer(transpose_, 3 * FRACTAL * sizeof(half));
        pipe_.InitBuffer(scalars_, 3 * TILE * sizeof(half));

        auto rhs = rhs_.Get<half>();
        Duplicate(rhs[(K_BLOCKS - 1) * FRACTAL], static_cast<half>(0), FRACTAL);
        Sync<HardEvent::V_S>();
        rhs.SetValue((K_BLOCKS - 1) * FRACTAL, static_cast<half>(-1));
        Sync<HardEvent::S_V>();
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t tilesPerHead = V / TILE;
        constexpr uint32_t tasksPerRequest = HV * tilesPerHead;
        for (uint32_t task = GetBlockIdx(); task < batch_ * tasksPerRequest; task += GetBlockNum()) {
            const uint32_t row = task / tasksPerRequest;
            const uint32_t head = (task / tilesPerHead) % HV;
            const uint32_t vStart = (task % tilesPerHead) * TILE;
            const uint32_t start = row * window_;
            const int32_t accepted = accepted_.GetValue(row);
            if (accepted <= 0 || accepted > static_cast<int32_t>(window_)) {
                continue;
            }
            const int32_t firstSlot = indices_.GetValue(start + accepted - 1);
            if (!ValidSlot(firstSlot)) {
                continue;
            }
            auto state = stateTile_.Get<half>();
            DataCopy(state, state_[StateOffset(firstSlot, head, vStart)], STATE_TILE);
            for (uint32_t token = start; token < start + window_; ++token) {
                const int32_t slot = indices_.GetValue(token);
                if (!ValidSlot(slot)) {
                    break;
                }
                Step(token, head, vStart, state);
                auto y = transpose_.Get<half>()[FRACTAL + TILE];
                Sync<HardEvent::V_MTE3>();
                DataCopy(out_[(static_cast<uint64_t>(token) * HV + head) * V + vStart], y, TILE);
                DataCopy(state_[StateOffset(slot, head, vStart)], state, STATE_TILE);
                // Reuse the same UB state for the next token after the snapshot drains.
                Sync<HardEvent::MTE3_V>();
            }
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

private:
    __aicore__ inline bool ValidSlot(int32_t slot) const
    {
        return slot > 0 && static_cast<uint32_t>(slot) < slots_;
    }

    __aicore__ inline uint64_t StateOffset(int32_t slot, uint32_t head, uint32_t vStart) const
    {
        return ((static_cast<uint64_t>(slot) * HV + head) * V + vStart) * K;
    }

    __aicore__ inline void Step(uint32_t token, uint32_t head, uint32_t vStart, LocalTensor<half> state)
    {
        const uint64_t qkOffset = (static_cast<uint64_t>(token) * HK + head / (HV / HK)) * K;
        const uint64_t headOffset = static_cast<uint64_t>(token) * HV;
        auto inputs = qk_.Get<half>();
        auto key = inputs;
        auto query = inputs[K];
        auto product = inputs[2 * K];
        auto scratch = transpose_.Get<half>();
        auto columns = scratch[FRACTAL];
        auto valueFractal = scratch[2 * FRACTAL];
        auto scalar = scalars_.Get<half>();

        DataCopy(key, k_[qkOffset], K);
        DataCopy(query, q_[qkOffset], K);
        DataCopy(scalar, gate_[headOffset], HV);
        DataCopy(scalar[TILE], beta_[headOffset], HV);
        DataCopy(columns, v_[((headOffset + head) * V) + vStart], TILE);
        Sync<HardEvent::MTE2_V>();
        Exp(scalar, scalar, HV);
        Mul(product, key, query, K);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(scalar[2 * TILE], product, K, 1, 1, 1, K / TILE);
        Sync<HardEvent::V_S>();
        const half decay = scalar.GetValue(head);
        const half beta = scalar.GetValue(TILE + head);
        const half kq = scalar.GetValue(2 * TILE);

        // Form [Hg | v] and [k, q; -1, 0]. One Mmad writes both
        // Hg*k-v and Hg*q directly to FP16 L0C.
        Muls(state, state, decay, STATE_TILE);
        Duplicate(scratch, static_cast<half>(0), FRACTAL);
        PipeBarrier<PIPE_V>();
        Adds(scratch, columns, static_cast<half>(0), TILE);
        PipeBarrier<PIPE_V>();
        Transpose(valueFractal, scratch);
        auto rhs = rhs_.Get<half>();
        for (uint32_t block = 0; block < K / TILE; ++block) {
            Adds(scratch, key[block * TILE], static_cast<half>(0), TILE);
            Adds(scratch[TILE], query[block * TILE], static_cast<half>(0), TILE);
            PipeBarrier<PIPE_V>();
            Transpose(rhs[block * FRACTAL], scratch);
            PipeBarrier<PIPE_V>();
        }
        auto c = Matvec(state, valueFractal, rhs);
        DataCopyEnhancedParams drain{};
        drain.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        Drain(c, columns, drain);
        auto delta = columns;
        if (deq_) {
            // Scalar DEQ broadcasts the same FP16 scale to the entire fractal.
            // Keep Hg*q from the plain drain and only consume column zero here.
            drain.deqScale = DeqScale::DEQ;
            // DEQ takes the FP16 multiplier bits in the low 16 bits.
            drain.deqValue = scalar.ReinterpretCast<uint16_t>().GetValue(TILE + head) ^ 0x8000u;
            Drain(c, scratch, drain);
            delta = scratch;
        } else {
            Muls(delta, delta, static_cast<half>(-static_cast<float>(beta)), TILE);
            PipeBarrier<PIPE_V>();
        }
        c1_.FreeTensor(c);
        auto y = columns[TILE];
        auto update = update_.Get<half>();
        uint32_t shape[2] = {TILE, K};
        uint32_t deltaShape[2] = {TILE, 1};
        Broadcast<half, 2, 1>(update, delta, shape, deltaShape);
        PipeBarrier<PIPE_V>();
        Mul(update, update, key, K, TILE, {1, 1, 1, K / TILE, K / TILE, 0});
        Muls(valueFractal, delta, kq, TILE);
        PipeBarrier<PIPE_V>();
        Add(state, state, update, STATE_TILE);
        Add(y, y, valueFractal, TILE);
        PipeBarrier<PIPE_V>();
        Muls(y, y, scale_, TILE);
        Sync<HardEvent::V_MTE2>();
    }

    __aicore__ inline void Drain(LocalTensor<half> c, LocalTensor<half> columns,
                                 const DataCopyEnhancedParams& params)
    {
        auto tile = c2_.AllocTensor<half>();
        DataCopy(tile, c, DataCopyParams{1, 1, 0, 0}, params);
        c2_.EnQue(tile);
        tile = c2_.DeQue<half>();
        PipeBarrier<PIPE_V>();
        Transpose(columns, tile);
        PipeBarrier<PIPE_V>();
        c2_.FreeTensor(tile);
    }

    __aicore__ inline LocalTensor<half> Matvec(LocalTensor<half> state, LocalTensor<half> value,
                                              LocalTensor<half> rhs)
    {
        auto a1 = a1_.Get<half>();
        auto b1 = b1_.Get<half>();
        Sync<HardEvent::V_MTE3>();
        for (uint32_t block = 0; block < K / TILE; ++block) {
            DataCopy(a1[block * FRACTAL], state[block * TILE],
                     DataCopyParams{TILE, 1, K / TILE - 1, 0});
        }
        DataCopy(a1[(K_BLOCKS - 1) * FRACTAL], value, DataCopyParams{1, FRACTAL / TILE, 0, 0});
        DataCopy(b1, rhs, DataCopyParams{1, K_BLOCKS * FRACTAL / TILE, 0, 0});
        Sync<HardEvent::MTE3_MTE1>();

        auto a2 = a2_.AllocTensor<half>();
        auto b2 = b2_.AllocTensor<half>();
        LoadData2DParams load{};
        load.repeatTimes = K_BLOCKS;
        load.srcStride = 1;
        LoadData(a2, a1, load);
        load.ifTranspose = true;
        LoadData(b2, b1, load);
        a2_.EnQue(a2);
        b2_.EnQue(b2);
        a2 = a2_.DeQue<half>();
        b2 = b2_.DeQue<half>();

        auto c = c1_.AllocTensor<half>();
        MmadParams params{};
        params.m = TILE;
        params.n = TILE;
        params.k = K_BLOCKS * TILE;
        params.cmatrixInitVal = true;
        Mmad(c, a2, b2, params);
        c1_.EnQue(c);
        a2_.FreeTensor(a2);
        b2_.FreeTensor(b2);
        return c1_.DeQue<half>();
    }

    TPipe pipe_;
    TBuf<TPosition::A1> a1_;
    TBuf<TPosition::B1> b1_;
    TQue<TPosition::A2, 1> a2_;
    TQue<TPosition::B2, 1> b2_;
    TQue<TPosition::CO1, 1> c1_;
    TQue<TPosition::CO2, 1> c2_;
    TBuf<TPosition::VECCALC> stateTile_, update_, qk_, rhs_, transpose_, scalars_;
    GlobalTensor<half> q_, k_, v_, beta_, gate_, state_, out_;
    GlobalTensor<int32_t> indices_, accepted_;
    uint32_t batch_, window_, slots_;
    half scale_;
    bool deq_;
};
} // namespace

extern "C" __global__ __aicore__ void rgdr_v310_cube_fp16(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR gate, GM_ADDR state,
    GM_ADDR indices, GM_ADDR accepted, GM_ADDR out, uint32_t batch, uint32_t window,
    uint32_t slots, float scale, uint32_t deq)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    CubeFp16RGDR op;
    op.Init(q, k, v, beta, gate, state, indices, accepted, out, batch, window, slots, scale, deq != 0);
    op.Process();
}
