# vLLM Ascend 0.23.0 的 grouped RGDR monkey patch

适用于 vLLM 0.23.0 与 vLLM Ascend 0.23.0 或 0.23.0rc1 搭配的 Ascend310P3 镜像。
每 rank 支持 Q/K 与 V head 数为 8/16 或 16/32，K=V=128，使用 FP16 SSM cache。
例如 Qwen 35B-A3B W8A8 的 attention TP2 或 attention DP2 + MoE EP2。
默认使用 **8 核 grouped baseline** 替换普通 GDN decode 的 recurrent 算子。
可选 `ASCEND310P_RGDR_PATCH_MODE=spec` 启用投机路径，或 `both` 启用两条路径。

插件在每个 vLLM 进程中替换 `vllm_ascend._310p.ops.fla.gdn_310` 的一个 Python 函数。
普通 decode 按 W=1 传入原生 block 槽位，每个请求从同一槽位读 state 并原位写回。
Prefill、卷积、采样、调度和 cache 分配沿用镜像原逻辑。形状或 cache layout 超出范围时
沿用原生函数。独立扩展使用全 FP16 递推，原生 gate 输入是 FP32，输出可能有舍入差。

## 1. 一键安装并启动

在已安装 vLLM、vLLM Ascend、Torch、torch_npu 和 CANN Toolkit 的目标镜像中运行。

```bash
git clone git@github.com:GoodThaumiel/ascend310p-rgdr-patch.git
cd ascend310p-rgdr-patch
./setup.sh -- "$MODEL" --tensor-parallel-size 2 --host 127.0.0.1 --port 18083
```

将 `--` 后的参数换成原来的 `vllm serve` 参数，图模式沿用原配置。测试 MTP
投机验证时，在命令前加 `ASCEND310P_RGDR_PATCH_MODE=spec`，或用 `both` 同时替换两条路径。

脚本优先使用当前 shell 配置的 CANN Toolkit，其次搜索常见安装目录。它加载 Toolkit
的 `set_env.sh`，输出 CANN 和 Python 包版本，核对 vLLM Ascend 310P 版本和
插件入口，编译独立扩展并通过 Python entry point 注册 monkey patch。已有
`VLLM_PLUGINS` 名单会保留并补充 Ascend 平台和 RGDR 插件。镜像内已有的
`custom_transformer` vendor 目录会加入 OPP 搜索路径。

只配置时执行 `./setup.sh`。配置完成后，重启服务可以直接使用生成的 launcher。

```bash
./build/rgdr-serve/vllm-serve "$MODEL" --tensor-parallel-size 2 --port 18083
```

迁移到其他 Python 环境或 CANN 安装时重新运行 `setup.sh`。非标准安装可传入
`--python /path/to/python3`、`--cann-root /path/to/toolkit` 和
`--cann-env /path/to/set_env.sh`。需要直接运行 profiling 或 smoke 时，先按脚本输出
`source` 生成的 `env.sh`，用同一个 Python 执行。已 clone 的仓库可先
`git pull --ff-only` 再运行脚本。

## 2. 手动构建并启用算子

仓库包含从 [`c28ba22`](https://github.com/GoodThaumiel/vllm-ascend/commit/c28ba222643929ad63afa51b72fb0f823dc9e5fc)
提取的 packed grouped kernel 和 Torch binding。在目标镜像中编译，
使扩展匹配镜像的 CPU 架构、CANN、Torch 和 torch_npu。CANN 路径请按镜像调整。

```bash
python3 -m pip install --no-deps --upgrade .
export RGDR_BUILD="$PWD/native-src/build-h16"
export RGDR_CANN=/usr/local/Ascend/cann-9.2.1
cmake -S native-src -B "$RGDR_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DSOC_VERSION=Ascend310P3 -DASCEND_CANN_PACKAGE_PATH="$RGDR_CANN" \
  -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$RGDR_BUILD" --target rgdr_v310_cube_fp16_torch -j 8
export ASCEND310P_RGDR_PATCH_LIBRARY="$RGDR_BUILD/librgdr_v310_cube_fp16_torch.so"
```

首次构建使用全新 build 目录，保留主 `.so` 旁边的 `lib/`。默认模式为 `decode`。

如果已有 `VLLM_PLUGINS` 白名单，把 `ascend310p_grouped_rgdr` 追加到原有名单。
启动时应看到两个 TP worker 分别打印 `310P grouped RGDR monkey patch installed`。
关闭特性时清除 `ASCEND310P_RGDR_PATCH_LIBRARY` 并重启服务。需要卸载时运行
`python3 -m pip uninstall ascend310p-rgdr-patch`。

## 3. 上机验证

先用镜像原有 CANN 环境运行 W=1 对照，检查 B1/B32 的输出、所有 state 写回和保留槽 0。
此命令使用 `decode` 或 `both` 模式。

```bash
python3 -m ascend310p_rgdr_patch.smoke --device 0
```

原生对照需要镜像的 `custom_transformer` OPP 包。若仍出现
`The infershape function does not exist`，确认下面的 vendor 目录在镜像中存在，
按实际位置调整，并在新的 Python 进程里重跑。

```bash
rgdr_vendor=/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
if [ -d "$rgdr_vendor" ]; then
  export ASCEND_CUSTOM_OPP_PATH="$rgdr_vendor${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
  python3 -m ascend310p_rgdr_patch.smoke --device 0
fi
```

也可以用 FP32 CPU 参考检查 grouped 算子在 NPU 上的输出和原位 state 更新。
此模式的通过结果仅覆盖 grouped 算子的数值检查。原生 A/B 仍需单独验证。

```bash
python3 -m ascend310p_rgdr_patch.smoke --device 0 --reference cpu
```

图模式部署先运行下面的命令。它在 B1 和 B32 上捕获并重放，
再用同一 B32 图重放 16 个活跃请求，检查槽位变更、padding 输出及 state 写回。

```bash
python3 -m ascend310p_rgdr_patch.smoke --device 0 --graph
python3 -m ascend310p_rgdr_patch.smoke --device 0 --graph --key-heads 16
```

然后用镜像的原图模式和并行参数启动 FP16 cache 的 `vllm serve`。测试普通 decode
时关闭投机解码。在独立的服务 trace 中确认
`rgdr_v310_cube_grouped_packed_fp16`（8/16 head）或
`rgdr_v310_cube_grouped_packed_fp16_h16`（16/32 head）出现在普通 decode。
用相同模型、请求、并发和采样设置做两次压测，分别关闭和开启扩展。若沿用
2k/400、B32 配置，可用以下命令。性能运行时关闭 profiler。

```bash
vllm bench serve --backend openai --endpoint /v1/completions --host 127.0.0.1 --port 18083 \
  --model "$MODEL" --served-model-name "$SERVED_MODEL" --dataset-name random \
  --random-input-len 2048 --random-output-len 400 --random-range-ratio 0 --seed 42 \
  --temperature 0 --ignore-eos --request-rate inf --max-concurrency 32 --num-prompts 64 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed --result-dir "$RESULTS" --result-filename "$VARIANT.json"
```

回传镜像版本、CANN/Torch/torch_npu 版本、W=1 误差、trace 算子名、
输出吞吐与 TPOT/ITL p50/p90。A/B 两侧使用相同的图模式配置。

### 服务仍走原生 V310

Smoke 手动注册插件，并使用 `--key-heads` 指定每 rank 的 Q/K head 数和对应的 V head 数。
服务由 vLLM 加载插件，并按模型实际输入判定。先在同一个 shell 里设置下面的变量，
用原有图模式参数重新启动服务，再发一条输出多 token 的请求。

```bash
export ASCEND310P_RGDR_PATCH_LIBRARY="$PWD/native-src/build-h16/librgdr_v310_cube_fp16_torch.so"
export ASCEND310P_RGDR_PATCH_DIAGNOSTICS=1
```

- `RGDR patch inactive` 表示进程没有收到库路径变量。
- 没有 `RGDR patch active` 或 `inactive` 日志时，检查 `vllm serve` 使用的 Python 环境，
  以及已有的 `VLLM_PLUGINS` 白名单是否包含 `ascend310p_grouped_rgdr`。
- `RGDR patch active` 后仍无 `RGDR dispatch` 时，检查已发出的请求是否进入 decode，
  以及该镜像实际使用的 GDN 调用入口。
- `RGDR dispatch ... route=native` 会给出回退原因及输入元数据。开启 MTP 时，
  默认 `decode` 模式会将验证窗口交给原生算子。要验证替换投机窗口，可显式设为
  `ASCEND310P_RGDR_PATCH_MODE=spec`。
- `RGDR dispatch` 的 `capturing=True` 表示日志来自图捕获。Python 分派记录的是捕获时的选择，
  重放应在同一 rank 的设备 trace 中核对对应的 grouped kernel。

### 单独确认服务分派

临时给服务启动命令追加
`--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/tmp/rgdr-023-trace`，
替换下面的端口和 served model 名称。测试完重启服务，恢复原性能测试参数。

```bash
curl -fsS -X POST http://127.0.0.1:18083/start_profile
curl -fsS http://127.0.0.1:18083/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"rgdr-test","prompt":"Explain matrix multiplication in three sentences.","max_tokens":32,"temperature":0}'
curl -fsS -X POST http://127.0.0.1:18083/stop_profile
```

## 4. Grouped 流水探针

流水探针提供三条独立的试验路径。服务继续使用 `baseline`。在目标镜像拉取代码并
重新安装 Python 包，按第 2 节在新的目录（例如 `native-src/build-probes`）重建
自定义算子，再将 `ASCEND310P_RGDR_PATCH_LIBRARY` 指向新库。

| 变体 | 唯一主要改动 | 要比较的指标 |
|---|---|---|
| `batch-pack` | 分批清零、复制和转置 RHS 与 V，packing 的源码级 V barrier 从 51 降为 9 次/pair/token，额外 UB 3.5 KiB/core | Task Duration、Vector、Scalar |
| `defer-write-wait` | 下一 token 先做 Q/K packing 和 dot 广播，再用相邻的 SetFlag/WaitFlag 等待前一 token 写回，随后才复用 state 和输出 UB | Task Duration、MTE3 与 Vector 的等待时序 |
| `no-state-store` | 只留输出写回和 UB 内部 recurrence，跳过每个 token 的候选 state 写回。仅供独立诊断 | Task Duration、MTE3 |

先检查三个变体在 B32 上的输出与 state。普通 decode 用 W1，MTP3 用 W4，
MTP7 用 W8。W1 没有跨 token 的前一轮写回等待，可作为 `defer-write-wait` 的对照。
默认 8 个 Q/K head、16 个 V head、8 个核。如果实际模型每 rank 为 16/32，
将下面的 `RGDR_KEY_HEADS` 改成 16。

0.1.6 上机时，B32/W4/H8 的 `defer-write-wait` 图重放在候选 state 逐位比对失败。
0.1.7 将跨 token 的事件改回成对发出和等待，保留前移的 dot 广播。重建后先重测
这个场景的 eager 与图重放，并跑一次 baseline 自对照。以下通过后继续矩阵。

```bash
export RGDR_DEVICE=0
export RGDR_KEY_HEADS=8
python3 -m ascend310p_rgdr_patch.profile --check --variant defer-write-wait \
  --window 4 --device "$RGDR_DEVICE" --key-heads "$RGDR_KEY_HEADS"
for variant in baseline defer-write-wait; do
  python3 -m ascend310p_rgdr_patch.profile --check --graph --variant "$variant" \
    --window 4 --device "$RGDR_DEVICE" --key-heads "$RGDR_KEY_HEADS" || exit 1
done
```

如果再次失败，回传含 eager/graph、repeat、首个索引以及 FP16 值和位模式的报错。
每条命令独立启动进程。

```bash
for window in 1 4 8; do
  for variant in batch-pack defer-write-wait no-state-store; do
    python3 -m ascend310p_rgdr_patch.profile --check --graph \
      --variant "$variant" --batch 32 --window "$window" --cores 8 \
      --key-heads "$RGDR_KEY_HEADS" --device "$RGDR_DEVICE" || exit 1
  done
done
```

每组重放三次，与 grouped baseline 做 FP16 位模式比较。`no-state-store` 的输出仍与
baseline 比较，GM state 则必须与调用前逐位一致。上述验证通过后，先抓 B32/W4
的五组 profile，再用 W1 检查固定开销和写回等待，用 W8 检查 token 数增加后的趋势。
原生路径从镜像加载 ACLNN 算子。若 OPP 注册失败，按第 3 节设置
`ASCEND_CUSTOM_OPP_PATH`，并在新的进程中重跑对应组。

```bash
export RGDR_PROFILE_OUT=/tmp/rgdr-310p-pipeline
mkdir -p "$RGDR_PROFILE_OUT"
for window in 4 1 8; do
  for variant in native baseline batch-pack defer-write-wait no-state-store; do
    msprof --output="$RGDR_PROFILE_OUT/$variant-w$window" --ai-core=on --aic-metrics=PipeUtilization \
      python3 -m ascend310p_rgdr_patch.profile --graph \
      --variant "$variant" --batch 32 --window "$window" --cores 8 \
      --key-heads "$RGDR_KEY_HEADS" --device "$RGDR_DEVICE" \
      --warmup 20 --iterations 100 || exit 1
  done
done
```

五组使用同一 RNG seed 和同一组输入、槽号与接受长度。原生收到 FP32 gate，
由 grouped 使用的同一份 FP16 gate 转换。state 在每次重放前通过独立操作恢复，
只统计目标 kernel，按执行顺序取最后 100 条。原生名称含 `RecurrentGatedDeltaRuleV310`，
grouped baseline 匹配 `rgdr_v310_cube_grouped_packed_fp16`，各探针匹配
`rgdr_v310_cube_probe_batch_pack`、`rgdr_v310_cube_probe_defer_write_wait` 或
`rgdr_v310_cube_probe_no_state_store`。16 个 Q/K head 的自定义 kernel 带 `_h16` 后缀。

回传 commit、设备与软件栈、逐组 check 输出、`op_summary_*.csv`；如 profiler 支持，
同时保留逐核 PMU 与流水事件。对比 Task Duration p50/p90、`aicore_time` 和
Scalar、Vector、MTE1/2/3、MAC 的绝对时间。几个流水的时间可重叠。对
`no-state-store` 观察到的差值包含写回传输和相关依赖的共同影响。

## 5. Grouped 7 核与 8 核对照

0.1.9 默认使用 `blockDim=8`，profiling 脚本默认选择 `baseline`。
按第 1 节配置好扩展和环境后运行以下对照。
核内计算和任务编号沿用当前实现，
每个 block 以实际 `GetBlockNum()` 为步长分配 `(request, head-pair)`。

先选实际的每 rank head 数。TP2 用 `RGDR_KEY_HEADS=8`，
attention TP1 + DP2 + EP2 用 `RGDR_KEY_HEADS=16`。

```bash
export RGDR_DEVICE=0
export RGDR_KEY_HEADS=8
python3 -m ascend310p_rgdr_patch.profile --check --variant baseline \
  --cores 8 --reference-cores 7 --batch 32 --window 4 \
  --key-heads "$RGDR_KEY_HEADS" --device "$RGDR_DEVICE"
for window in 1 4 8; do
  python3 -m ascend310p_rgdr_patch.profile --check --graph --variant baseline \
    --cores 8 --reference-cores 7 --batch 32 --window "$window" \
    --key-heads "$RGDR_KEY_HEADS" --device "$RGDR_DEVICE" || exit 1
done
```

通过后，在同卡抓 B32 / MTP3 的 c7/c8，各取最后 100 条 grouped kernel。
记录 Block Dim 和 Task Duration p50/p90，算子符号名与第 4 节相同。
可将 `--window 4` 改成 `--window 8` 补测 MTP7。

```bash
for cores in 7 8; do
  msprof --output="/tmp/rgdr-310p-c$cores-w4" --ai-core=on --aic-metrics=PipeUtilization \
    python3 -m ascend310p_rgdr_patch.profile --graph --variant baseline \
    --cores "$cores" --batch 32 --window 4 \
    --key-heads "$RGDR_KEY_HEADS" --device "$RGDR_DEVICE" \
    --warmup 20 --iterations 100 || exit 1
done
```

服务启动时 worker 日志应显示 `cores=8`，可在设备 trace 中核对 Block Dim。
对照 c7 时，在启动前设置 `ASCEND310P_RGDR_PATCH_CORES=7` 并重新捕图。
第 3 节的普通 decode 图重放 smoke 也读取该变量。

## 6. State gate 与 rank-1 update 消融

使用 `./setup.sh` 编译并安装 0.1.10，然后按其输出加载新 build 的 `env.sh`。
本轮用 B32 / H8 / c8、同卡、同采集配置。对照与所有探针使用同一个新构建，
每组独立抓取。先用 `baseline` 做图重放逐位自对照，再测 W1/W2/W4/W8 的增长曲线。

| 探针 | 执行变化 |
|---|---|
| `no-gate-scale` | 省去 128×128 state 的 gate 乘法。在源码中保留 gate 准备。 |
| `no-rank1` | 省去生成并叠加 rank-1 矩阵的三步。保留输出的 `delta × (k·q)` 修正。 |
| `no-rank1-broadcast` | 省去展开 delta 的 Broadcast。后续乘法读取初始化过的 state 矩阵，保留乘法形状和载入量。 |
| `no-rank1-mul` | 将展开后的 delta 直接用于 state 加法，省去 rank-1 矩阵乘 key 的 Mul。 |
| `no-rank1-add` | 将 rank-1 Mul 的结果直接写至 state UB，省去原 state 与该结果的 Add。 |

五个探针的递推都与 baseline 不同，只通过 profiling 入口运行。每次测量前自动
检查一次输出和 state 有限、保留槽 0 逐位不变。探针继续写回所有候选 state。
编译后的调度及 UB 读写依赖可能随改动变化。用 `no-rank1` 估计整段影响，
用三个单阶段变体定位变化主要发生在哪段，分别与同 W baseline 比较。各差值
不可直接视为指令时长或相加。

```bash
export RGDR_DEVICE=0
export RGDR_KEY_HEADS=8
export RGDR_PROFILE_OUT=/tmp/rgdr-310p-vector-c8
python3 -m ascend310p_rgdr_patch.profile --check --graph --variant baseline \
  --batch 32 --window 4 --key-heads "$RGDR_KEY_HEADS" --cores 8 --device "$RGDR_DEVICE"

profile_rgdr() {
  local variant=$1 window=$2
  msprof --output="$RGDR_PROFILE_OUT/$variant-w$window" --ai-core=on --aic-metrics=PipeUtilization \
    python3 -m ascend310p_rgdr_patch.profile --graph --variant "$variant" \
    --batch 32 --window "$window" --key-heads "$RGDR_KEY_HEADS" --cores 8 \
    --device "$RGDR_DEVICE" --warmup 20 --iterations 100
}
mkdir -p "$RGDR_PROFILE_OUT"
for window in 1 2 4 8; do
  for variant in baseline no-gate-scale no-rank1; do
    profile_rgdr "$variant" "$window" || exit 1
  done
done
for window in 4 8; do
  for variant in no-rank1-broadcast no-rank1-mul no-rank1-add; do
    profile_rgdr "$variant" "$window" || exit 1
  done
done
```

回传脚本打印的请求核数、目标 kernel 的 CSV `Block Num`（应为 8）、
Task Duration p50/p90、Vector、Scalar、MTE1/2/3 和 MAC。目标符号为
`rgdr_v310_cube_grouped_packed_fp16` 或 `rgdr_v310_cube_probe_<variant>`，
其中 variant 名的连字符换成下划线。H16 另外带 `_h16`。按时间排序取目标
kernel 的最后 100 条。若观察到几个微秒的差异，重抓对应的 baseline 和探针
再判断重现性。

## 7. Rank-1 的 16 列广播与双 head 编排

0.1.11 增加三条可与 baseline 逐位比较的路径。每个 delta 仅展开为一个 32B
FP16 数据块。Vector Mul 采用 `{1, 0, 1, 8, 1, 0}`，每个 repeat 计算一行
128 列。delta 在行内复用同一个块，key 在行间复用同一行。仍先将乘积舍入到
FP16，再做 FP16 state Add。

| 变体 | 改动 |
|---|---|
| `rank1-brcb16` | 用 Brcb 在 UB 生成 128×16 的 delta，保留逐 head 顺序。额外显式 UB 4 KiB/core。 |
| `rank1-gather16` | 用预生成的 UB 偏移和 Gather 生成相同的 128×16，保留逐 head 顺序。额外显式 UB 12 KiB/core。 |
| `rank1-gather16-interleave` | 同一 token 先取出并缩放 head 0 的 delta，提交 head 1 的 Mmad，再执行 head 0 的 Gather、Mul 与 Add。延后 head 1 的 CO1 DeQue。 |

Gather 变体在 kernel 初始化时用 Brcb 生成偏移，计算期间的广播只访问 UB。
交错版仍复用单份 L0A/L0C。head 0 的 L0C 已读出，随后才提交 head 1 的
Mmad。交错收益以两个 Gather 变体的差值衡量。

用 `./setup.sh` 重建主扩展，按其输出加载 `env.sh`，在目标设备用上一轮相同的
CANN 环境进行验证。B32/H8/c8、W4/W8 先逐位比对输出和候选 state，包括保留槽。

```bash
export RGDR_DEVICE=2
export RGDR_PROFILE_OUT=/tmp/rgdr-310p-rank1-16
for window in 4 8; do
  for variant in rank1-brcb16 rank1-gather16 rank1-gather16-interleave; do
    python3 -m ascend310p_rgdr_patch.profile --check --graph --variant "$variant" \
      --batch 32 --key-heads 8 --cores 8 --window "$window" --device "$RGDR_DEVICE" || exit 1
  done
done

for window in 4 8; do
  for variant in baseline rank1-brcb16 rank1-gather16 rank1-gather16-interleave; do
    msprof --output="$RGDR_PROFILE_OUT/$variant-w$window" --ai-core=on --aic-metrics=PipeUtilization \
      python3 -m ascend310p_rgdr_patch.profile --graph --variant "$variant" \
      --batch 32 --key-heads 8 --cores 8 --window "$window" --device "$RGDR_DEVICE" \
      --warmup 20 --iterations 100 || exit 1
  done
done
```

每组取最后 100 次目标 kernel，回传 Task Duration p50/p90、Vector、Scalar、
MAC 和 Block Num。变体的 kernel 名用 `rgdr_v310_cube_probe_` 前缀，
将变体名中的连字符换成下划线。也可用 `--key-heads 16` 验证 H16。

### UB 到 L0C 原语探针

310P3 支持矩阵列出了 `broadcast_ub_to_cc` FP16，仓内 GVirt 有该低阶原语的签名。
先用单块 FP16 输入验证 `nBurst=1, lenBurst=1` 的 16×16 读回布局。
这条指令的 CANN 9.1.0-beta.1 编译情况、输出方向和发射流水需要上机证实。
原语探针按独立 CMake target 构建，使用全流水屏障隔离布局测试。

```bash
export RGDR_L0C_BUILD="$PWD/native-src/build-l0c-probe"
cmake -S native-src -B "$RGDR_L0C_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DSOC_VERSION=Ascend310P3 -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DRGDR_ENABLE_L0C_BROADCAST_PROBE=ON -DPython3_EXECUTABLE="$(command -v python3)"
cmake --build "$RGDR_L0C_BUILD" --target rgdr_v310_l0c_probe_torch -j 8
python3 -m ascend310p_rgdr_patch.l0c_probe --device "$RGDR_DEVICE" \
  --library "$RGDR_L0C_BUILD/librgdr_v310_l0c_probe_torch.so"
```

回传首行、首列及 PASS 或完整编译错误。有可读 ISA 时一并回传该指令的流水归属，
用于判断原语能否和另一 head 的 Mmad 在不同 L0C 区域并行。
