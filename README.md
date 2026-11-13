# vLLM Ascend 0.23.0 的 grouped RGDR monkey patch

适用于已有 vLLM 0.23.0、vLLM Ascend 0.23.0 的 Ascend310P3 镜像。目标形状为每 rank
8 个 Q/K head、16 个 value head、K=V=128，使用 FP16 SSM cache。例如该形状的
Qwen 35B-A3B W8A8 TP2。默认只替换普通 GDN decode 的 recurrent 算子。
可选 `ASCEND310P_RGDR_PATCH_MODE=spec` 启用投机路径，或 `both` 启用两条路径。

插件在每个 vLLM 进程中替换 `vllm_ascend._310p.ops.fla.gdn_310` 的一个 Python 函数。
普通 decode 按 W=1 传入原生 block 槽位，每个请求从同一槽位读 state 并原位写回。
Prefill、卷积、采样、调度和 cache 分配沿用镜像原逻辑。形状或 cache layout 超出范围时
沿用原生函数。独立扩展使用全 FP16 递推，原生 gate 输入是 FP32，输出可能有舍入差。

## 1. 安装插件

在目标镜像中执行以下命令。使用启动 vLLM 的同一个 Python 环境。

```bash
git clone git@github.com:GoodThaumiel/ascend310p-rgdr-patch.git
cd ascend310p-rgdr-patch
python3 -m pip install --no-deps .
```

## 2. 构建并启用算子

仓库包含从 [`c28ba22`](https://github.com/GoodThaumiel/vllm-ascend/commit/c28ba222643929ad63afa51b72fb0f823dc9e5fc)
提取的 packed grouped kernel 和 Torch binding。在目标镜像中编译，
使扩展匹配镜像的 CPU 架构、CANN、Torch 和 torch_npu。CANN 路径请按镜像调整。

```bash
export RGDR_BUILD="$PWD/native-src/build"
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

然后用同事镜像的原启动参数启动 TP2 FP16 cache 的 `vllm serve`，去掉
`--speculative-config`，第一轮加 `--enforce-eager`。在独立的服务 trace 中确认
`rgdr_v310_cube_grouped_packed_fp16` 出现在普通 decode，排除 shape 回退。
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
输出吞吐与 TPOT/ITL p50/p90。若生产用图模式，eager 对照通过后，用双方相同的图模式配置复测。

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
