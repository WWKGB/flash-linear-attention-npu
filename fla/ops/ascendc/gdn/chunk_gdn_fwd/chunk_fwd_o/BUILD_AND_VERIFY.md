# chunk_fwd_o A5 编译与验证

本文记录当前开发环境下 `chunk_fwd_o` A5 路径的编译、安装和验证命令。

## 1. 环境准备

```bash
export REPO_ROOT=/home/npu_user6/hey/flash-linear-attention-npu
export CANN_ROOT=/home/wys/cann0707/ascend/cann-9.1.0

source /home/npu_user6/miniforge3/etc/profile.d/conda.sh
conda activate wys_gdn
source "${CANN_ROOT}/set_env.sh"
cd "${REPO_ROOT}"
```

确认环境：

```bash
echo "${CONDA_DEFAULT_ENV}"
command -v python
command -v bisheng
```

预期 conda 环境为 `wys_gdn`，且 `bisheng` 可以被找到。

## 2. 编译算子包

使用文件锁避免多个构建同时清理 `build/` 临时目录：

```bash
cd "${REPO_ROOT}"
flock -w 600 /tmp/chunk_fwd_o_build.lock \
  bash build.sh \
    --soc=ascend950 \
    --pkg \
    --vendor_name=fla_npu \
    --ops=chunk_fwd_o 
```

成功标志：

```text
CPack: - package: .../build/fla-npu-fla_npu_linux-x86_64.run
```

检查安装包：

```bash
ls -l "${REPO_ROOT}/build/fla-npu-fla_npu_linux-x86_64.run"
```



## 3. 安装并加载自定义 OPP

```bash
cd "${REPO_ROOT}"
./build/fla-npu-fla_npu_linux-x86_64.run \
  --quiet \
  --install-path="${REPO_ROOT}/.opp_cann"

source "${REPO_ROOT}/.opp_cann/vendors/fla_npu_transformer/bin/set_env.bash"
```

安装成功会输出：

```text
SUCCESS
```



## 4. Stage3 调用冒烟

当前该用例用于确认 custom op 能返回、NPU 能完成同步且进程正常退出。

```bash
cd "${REPO_ROOT}/torch_custom/fla_npu"
flock -w 180 /tmp/chunk_fwd_o_test.lock \
  timeout --signal=KILL 120s \
  python -u test/test_fwd_o.py \
    1 64 1 1 128 128 0 1 64 0.0883883 bf16 0 0 data 0 float32
```

通过标志：

```text
step 7: after custom op
step 8: after synchronize
step 9: save done
```

命令退出码应为 `0`。

## 5. Stage1 + Stage2 精度回归

```bash
cd "${REPO_ROOT}/torch_custom/fla_npu"
flock -w 180 /tmp/chunk_fwd_o_test.lock \
  timeout --signal=KILL 120s \
  python -u test/test_stage_precision.py
```

通过标志：

```text
=== T=64 exp2 ===
[PASS] mask
[PASS] gate_o
[PASS] gate_A
[PASS] A_raw
[PASS] O_s_raw

=== T=48 tail exp2 ===
[PASS] mask
[PASS] gate_o
[PASS] gate_A
[PASS] A_raw
[PASS] O_s_raw

All stage precision checks passed.
```

注意：当前该脚本只验收 Stage1 和 Stage2 中间量，不代表 Stage3 精度已通过。

## 6. 一次执行完整流程

```bash
export REPO_ROOT=/home/npu_user6/hey/flash-linear-attention-npu
export CANN_ROOT=/home/wys/cann0707/ascend/cann-9.1.0

source /home/npu_user6/miniforge3/etc/profile.d/conda.sh
conda activate wys_gdn
source "${CANN_ROOT}/set_env.sh"

cd "${REPO_ROOT}"
flock -w 600 /tmp/chunk_fwd_o_build.lock \
  bash build.sh --soc=ascend950 --pkg \
    --vendor_name=fla_npu --ops=chunk_fwd_o -j1

./build/fla-npu-fla_npu_linux-x86_64.run \
  --quiet --install-path="${REPO_ROOT}/.opp_cann"
source "${REPO_ROOT}/.opp_cann/vendors/fla_npu_transformer/bin/set_env.bash"

cd "${REPO_ROOT}/torch_custom/fla_npu"
flock -w 180 /tmp/chunk_fwd_o_test.lock \
  timeout --signal=KILL 120s \
  python -u test/test_fwd_o.py \
    1 64 1 1 128 128 0 1 64 0.0883883 bf16 0 0 data 0 float32

flock -w 180 /tmp/chunk_fwd_o_test.lock \
  timeout --signal=KILL 120s \
  python -u test/test_stage_precision.py
```



## 7. 常见问题



### `bisheng compilation tool not found`

重新加载 CANN 环境：

```bash
source /home/wys/cann0707/ascend/cann-9.1.0/set_env.sh
command -v bisheng
```



### `ModuleNotFoundError: No module named 'ml_dtypes'`

测试使用了错误的 Python 环境：

```bash
source /home/npu_user6/miniforge3/etc/profile.d/conda.sh
conda activate wys_gdn
```



### 构建出现临时文件丢失或 `Stop flag detected`

通常是多个构建并发操作同一个 `build/` 目录。等待其他构建结束，并使用本文的
`flock /tmp/chunk_fwd_o_build.lock` 命令重新执行。

### 测试出现数据来自上一轮或 CrossCore 同步异常

确认没有并发运行多个 `chunk_fwd_o` 测试。使用本文的
`flock /tmp/chunk_fwd_o_test.lock` 命令串行测试。

### `507015` / `AICore execution is abnormal`

设置同步启动以获得更准确的报错位置：

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

复现后及时取消：

```bash
unset ASCEND_LAUNCH_BLOCKING
```

