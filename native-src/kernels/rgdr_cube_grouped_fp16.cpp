// SPDX-License-Identifier: Apache-2.0
// One 310P3 coupled core owns a request's pair of value heads sharing Q/K.
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t K = 128;
constexpr uint32_t V = 128;
constexpr uint32_t HK = 8;
constexpr uint32_t HV = 16;
constexpr uint32_t PAIR = HV / HK;
constexpr uint32_t MAX_WINDOW = 8;
constexpr uint32_t C0 = 16;
constexpr uint32_t FRACTAL = C0 * C0;
constexpr uint32_t K_BLOCKS = K / C0 + 1;
constexpr uint32_t M_BLOCKS = V / C0;
constexpr uint32_t STATE = V * K;
constexpr uint32_t LHS = V * K_BLOCKS * C0;
constexpr uint32_t RHS = K_BLOCKS * FRACTAL;
constexpr uint32_t RESULT = V * C0;
constexpr uint32_t PAIR_STATE = PAIR * STATE;
constexpr uint32_t PAIR_VALUE = PAIR * V;

template <HardEvent Event>
__aicore__ inline void Sync()
{
    const auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

template <bool PACKED = false>
class GroupedFp16RGDR {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR gate,
                                GM_ADDR state, GM_ADDR indices, GM_ADDR accepted, GM_ADDR out,
                                uint32_t batch, uint32_t window, uint32_t slots, float scale,
                                GM_ADDR rowOffsets = nullptr, uint32_t tokens = 0, uint64_t stateStride = HV * STATE)
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
        if constexpr (PACKED) {
            rowOffsets_.SetGlobalBuffer((__gm__ int32_t*)rowOffsets);
            tokens_ = tokens;
            stateStride_ = stateStride;
        }
        batch_ = batch;
        window_ = window;
        slots_ = slots;
        scale_ = static_cast<half>(scale);

        pipe_.InitBuffer(a1_, LHS * sizeof(half));
        pipe_.InitBuffer(b1_, RHS * sizeof(half));
        pipe_.InitBuffer(a2_, 1, LHS * sizeof(half));
        pipe_.InitBuffer(b2_, 1, RHS * sizeof(half));
        pipe_.InitBuffer(c1_, 1, RESULT * sizeof(half));
        pipe_.InitBuffer(c2_, 1, RESULT * sizeof(half));

        pipe_.InitBuffer(states_, PAIR_STATE * sizeof(half));
        pipe_.InitBuffer(update_, STATE * sizeof(half));
        pipe_.InitBuffer(qkv_, window_ * (2 * K + PAIR_VALUE) * sizeof(half));
        pipe_.InitBuffer(gates_, 2 * window_ * HV * sizeof(half));
        pipe_.InitBuffer(rhs_, RHS * sizeof(half));
        pipe_.InitBuffer(valueTail_, RESULT * sizeof(half));
        pipe_.InitBuffer(scratch_, FRACTAL * sizeof(half));
        pipe_.InitBuffer(vectors_, (5 * V + PAIR_VALUE) * sizeof(half));
        pipe_.InitBuffer(gatherOffsets_, 3 * V * sizeof(uint32_t));

        auto rhs = rhs_.Get<half>();
        Duplicate(rhs[(K_BLOCKS - 1) * FRACTAL], static_cast<half>(0), FRACTAL);
        auto index = gatherOffsets_.Get<uint32_t>();
        for (uint32_t row = 0; row < V; ++row) {
            index.SetValue(row, row * C0 * sizeof(half));
            index.SetValue(V + row, (row * C0 + 1) * sizeof(half));
        }
        Sync<HardEvent::V_S>();
        rhs.SetValue((K_BLOCKS - 1) * FRACTAL, static_cast<half>(-1));
        Sync<HardEvent::S_V>();
    }

    __aicore__ inline void Process()
    {
        if constexpr (PACKED) {
            const int32_t end = rowOffsets_.GetValue(batch_);
            if (end < 0 || static_cast<uint32_t>(end) > tokens_) {
                return;
            }
            // Graph padding can leave a tail outside the packed query rows.
            if (static_cast<uint32_t>(end) < tokens_) {
                ZeroOutputTail(static_cast<uint32_t>(end));
            }
        }
        for (uint32_t task = GetBlockIdx(); task < batch_ * HK; task += GetBlockNum()) {
            const uint32_t row = task / HK;
            const uint32_t keyHead = task % HK;
            const uint32_t head = keyHead * PAIR;
            uint32_t start = row * window_;
            uint32_t length = window_;
            if constexpr (PACKED) {
                const int32_t begin = rowOffsets_.GetValue(row);
                const int32_t end = rowOffsets_.GetValue(row + 1);
                if (begin < 0 || end <= begin || static_cast<uint32_t>(end) > tokens_ ||
                    static_cast<uint32_t>(end - begin) > window_) {
                    continue;
                }
                start = static_cast<uint32_t>(begin);
                length = static_cast<uint32_t>(end - begin);
            }
            const int32_t accepted = accepted_.GetValue(row);
            if (accepted <= 0 || accepted > static_cast<int32_t>(window_)) {
                continue;
            }
            uint64_t offsets[MAX_WINDOW];
            // The selected previous state can lie beyond this step's short
            // query window. The row's full slot table preserves that state.
            const uint32_t slotRow = row * window_;
            if (!ReadStateOffsets(slotRow, head, length, offsets)) {
                continue;
            }
            const uint32_t acceptedIndex = static_cast<uint32_t>(accepted - 1);
            uint64_t initialOffset;
            if (acceptedIndex < length) {
                initialOffset = offsets[acceptedIndex];
            } else if (!ReadStateOffset(slotRow + acceptedIndex, head, initialOffset)) {
                continue;
            }
            ProcessPair(start, length, keyHead, head, offsets, initialOffset);
        }
    }

private:
    __aicore__ inline void ZeroOutputTail(uint32_t start)
    {
        auto zero = vectors_.Get<half>();
        Duplicate(zero, static_cast<half>(0), PAIR_VALUE);
        Sync<HardEvent::V_MTE3>();
        for (uint32_t chunk = GetBlockIdx(); chunk < (tokens_ - start) * HK; chunk += GetBlockNum()) {
            DataCopy(out_[static_cast<uint64_t>(start) * HV * V + static_cast<uint64_t>(chunk) * PAIR_VALUE],
                     zero, PAIR_VALUE);
        }
        Sync<HardEvent::MTE3_V>();
    }

    __aicore__ inline bool ReadStateOffset(uint32_t index, uint32_t head, uint64_t& offset)
    {
        const int32_t slot = indices_.GetValue(index);
        if (slot <= 0 || static_cast<uint32_t>(slot) >= slots_) {
            return false;
        }
        const uint64_t stateStride = PACKED ? stateStride_ : HV * STATE;
        offset = static_cast<uint64_t>(slot) * stateStride + head * STATE;
        return true;
    }

    __aicore__ inline bool ReadStateOffsets(uint32_t start, uint32_t head, uint32_t length, uint64_t* offsets)
    {
        for (uint32_t t = 0; t < length; ++t) {
            if (!ReadStateOffset(start + t, head, offsets[t])) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void ProcessPair(uint32_t start, uint32_t length, uint32_t keyHead, uint32_t head,
                                      const uint64_t* offsets, uint64_t initialOffset)
    {
        auto states = states_.Get<half>();
        auto qkv = qkv_.Get<half>();
        auto keys = qkv;
        auto queries = qkv[window_ * K];
        auto values = qkv[2 * window_ * K];
        auto gates = gates_.Get<half>();
        auto negativeBeta = gates[window_ * HV];
        auto update = update_.Get<half>();
        // The tail of vectors_ is the paired contiguous output. Its first V
        // elements are scratch for the dot products before any token executes.
        auto vectors = vectors_.Get<half>();
        auto dots = vectors;
        auto output = vectors[5 * V];
        const uint64_t qkBase = (static_cast<uint64_t>(start) * HK + keyHead) * K;
        const uint64_t gateBase = static_cast<uint64_t>(start) * HV;
        const uint64_t valueBase = (gateBase + head) * V;

        DataCopy(states, state_[initialOffset], PAIR_STATE);
        const DataCopyParams qkCopy{static_cast<uint16_t>(length), K / C0, (HK - 1) * K / C0, 0};
        DataCopy(keys, k_[qkBase], qkCopy);
        DataCopy(queries, q_[qkBase], qkCopy);
        DataCopy(values, v_[valueBase],
                 DataCopyParams{static_cast<uint16_t>(length), PAIR_VALUE / C0,
                                (HV * V - PAIR_VALUE) / C0, 0});
        DataCopy(gates, gate_[gateBase], length * HV);
        DataCopy(negativeBeta, beta_[gateBase], length * HV);
        Sync<HardEvent::MTE2_V>();
        Exp(gates, gates, length * HV);
        Muls(negativeBeta, negativeBeta, static_cast<half>(-1), length * HV);
        Mul(update, keys, queries, length * K);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dots, update, K, length, 1, 1, K / C0);

        uint64_t outOffset = valueBase;
        for (uint32_t t = 0; t < length; ++t, outOffset += HV * V) {
            auto key = keys[t * K];
            PackRhs(key, queries[t * K]);
            // RHS packing can run while the previous candidate drains to GM.
            if (t != 0) {
                Sync<HardEvent::MTE3_V>();
            }
            auto b2 = LoadRhs();
            BroadcastLane(vectors[V], dots, t);
            for (uint32_t h = 0; h < PAIR; ++h) {
                const uint32_t lane = head + h;
                auto state = states[h * STATE];
                BroadcastLane(vectors[2 * V], gates[t * HV], lane);
                BroadcastLane(vectors[3 * V], negativeBeta[t * HV], lane);
                Mul(state, state, vectors[2 * V], K, V, {1, 1, 1, K / C0, K / C0, 0});
                PackValue(values[t * PAIR_VALUE + h * V]);
                auto c = Matvec(state, b2);
                Update(state, key, c, output[h * V]);
            }
            b2_.FreeTensor(b2);
            Sync<HardEvent::V_MTE3>();
            DataCopy(out_[outOffset], output, PAIR_VALUE);
            DataCopy(state_[offsets[t]], states, PAIR_STATE);
        }
        Sync<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void BroadcastLane(LocalTensor<half> dst, LocalTensor<half> src, uint32_t lane)
    {
        LocalTensor<uint32_t> offset = gatherOffsets_.Get<uint32_t>()[2 * V];
        Duplicate(offset.ReinterpretCast<int32_t>(), static_cast<int32_t>(lane * sizeof(half)), V);
        PipeBarrier<PIPE_V>();
        Gather(dst, src, offset, 0, V);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PackRhs(LocalTensor<half> key, LocalTensor<half> query)
    {
        auto scratch = scratch_.Get<half>();
        auto rhs = rhs_.Get<half>();
        Duplicate(scratch, static_cast<half>(0), FRACTAL);
        PipeBarrier<PIPE_V>();
        for (uint32_t block = 0; block < K / C0; ++block) {
            Adds(scratch, key[block * C0], static_cast<half>(0), C0);
            Adds(scratch[C0], query[block * C0], static_cast<half>(0), C0);
            PipeBarrier<PIPE_V>();
            Transpose(rhs[block * FRACTAL], scratch);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PackValue(LocalTensor<half> value)
    {
        auto scratch = scratch_.Get<half>();
        auto tail = valueTail_.Get<half>();
        Duplicate(scratch, static_cast<half>(0), FRACTAL);
        PipeBarrier<PIPE_V>();
        for (uint32_t block = 0; block < M_BLOCKS; ++block) {
            Adds(scratch, value[block * C0], static_cast<half>(0), C0);
            PipeBarrier<PIPE_V>();
            Transpose(tail[block * FRACTAL], scratch);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline LocalTensor<half> LoadRhs()
    {
        auto b1 = b1_.Get<half>();
        auto b2 = b2_.AllocTensor<half>();
        Sync<HardEvent::V_MTE3>();
        DataCopy(b1, rhs_.Get<half>(), DataCopyParams{1, RHS / C0, 0, 0});
        Sync<HardEvent::MTE3_MTE1>();
        LoadData2DParams load{};
        load.repeatTimes = K_BLOCKS;
        load.srcStride = 1;
        load.ifTranspose = true;
        LoadData(b2, b1, load);
        b2_.EnQue(b2);
        return b2_.DeQue<half>();
    }

    __aicore__ inline LocalTensor<half> Matvec(LocalTensor<half> state, LocalTensor<half> b2)
    {
        auto a1 = a1_.Get<half>();
        Sync<HardEvent::V_MTE3>();
        for (uint32_t block = 0; block < K / C0; ++block) {
            DataCopy(a1[block * RESULT], state[block * C0], DataCopyParams{V, 1, K / C0 - 1, 0});
        }
        DataCopy(a1[(K_BLOCKS - 1) * RESULT], valueTail_.Get<half>(), DataCopyParams{1, RESULT / C0, 0, 0});
        Sync<HardEvent::MTE3_MTE1>();

        auto a2 = a2_.AllocTensor<half>();
        LoadData2DParams load{};
        load.repeatTimes = K_BLOCKS;
        load.srcStride = M_BLOCKS;
        for (uint32_t block = 0; block < M_BLOCKS; ++block) {
            LoadData(a2[block * K_BLOCKS * FRACTAL], a1[block * FRACTAL], load);
        }
        a2_.EnQue(a2);
        a2 = a2_.DeQue<half>();
        auto c = c1_.AllocTensor<half>();
        MmadParams params{};
        params.m = V;
        params.n = C0;
        params.k = K_BLOCKS * C0;
        params.cmatrixInitVal = true;
        Mmad(c, a2, b2, params);
        c1_.EnQue(c);
        a2_.FreeTensor(a2);
        return c1_.DeQue<half>();
    }

    __aicore__ inline void Update(LocalTensor<half> state, LocalTensor<half> key,
                                  LocalTensor<half> c, LocalTensor<half> output)
    {
        auto tile = c2_.AllocTensor<half>();
        DataCopyEnhancedParams drain{};
        drain.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(tile, c, DataCopyParams{1, M_BLOCKS, 0, 0}, drain);
        c2_.EnQue(tile);
        tile = c2_.DeQue<half>();
        PipeBarrier<PIPE_V>();

        auto vectors = vectors_.Get<half>();
        auto delta = vectors[4 * V];
        auto scratch = vectors[2 * V];
        auto index = gatherOffsets_.Get<uint32_t>();
        Gather(delta, tile, index, 0, V);
        Gather(output, tile, index[V], 0, V);
        PipeBarrier<PIPE_V>();
        c2_.FreeTensor(tile);
        c1_.FreeTensor(c);

        Mul(delta, delta, vectors[3 * V], V);
        PipeBarrier<PIPE_V>();
        auto update = update_.Get<half>();
        uint32_t shape[2] = {V, K};
        uint32_t deltaShape[2] = {V, 1};
        Broadcast<half, 2, 1>(update, delta, shape, deltaShape);
        Mul(scratch, delta, vectors[V], V);
        PipeBarrier<PIPE_V>();
        Mul(update, update, key, K, V, {1, 1, 1, K / C0, K / C0, 0});
        Add(output, output, scratch, V);
        PipeBarrier<PIPE_V>();
        Add(state, state, update, STATE);
        Muls(output, output, scale_, V);
        PipeBarrier<PIPE_V>();
    }

    TPipe pipe_;
    TBuf<TPosition::A1> a1_;
    TBuf<TPosition::B1> b1_;
    TQue<TPosition::A2, 1> a2_;
    TQue<TPosition::B2, 1> b2_;
    TQue<TPosition::CO1, 1> c1_;
    TQue<TPosition::CO2, 1> c2_;
    TBuf<TPosition::VECCALC> states_, update_, qkv_, gates_, rhs_, valueTail_, scratch_, vectors_, gatherOffsets_;
    GlobalTensor<half> q_, k_, v_, beta_, gate_, state_, out_;
    GlobalTensor<int32_t> indices_, accepted_, rowOffsets_;
    uint32_t batch_, window_, slots_, tokens_;
    uint64_t stateStride_;
    half scale_;
};
} // namespace

extern "C" __global__ __aicore__ void rgdr_v310_cube_grouped_fp16(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR gate, GM_ADDR state,
    GM_ADDR indices, GM_ADDR accepted, GM_ADDR out, uint32_t batch, uint32_t window,
    uint32_t slots, float scale)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    GroupedFp16RGDR<> op;
    op.Init(q, k, v, beta, gate, state, indices, accepted, out, batch, window, slots, scale);
    op.Process();
}

extern "C" __global__ __aicore__ void rgdr_v310_cube_grouped_packed_fp16(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR gate, GM_ADDR state,
    GM_ADDR indices, GM_ADDR accepted, GM_ADDR out, uint32_t batch, uint32_t window,
    uint32_t slots, float scale, GM_ADDR rowOffsets, uint32_t tokens, uint64_t stateStride)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC);
    GroupedFp16RGDR<true> op;
    op.Init(q, k, v, beta, gate, state, indices, accepted, out, batch, window, slots, scale, rowOffsets, tokens, stateStride);
    op.Process();
}
