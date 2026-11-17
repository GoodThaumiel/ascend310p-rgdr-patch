#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail

fail() {
    printf 'RGDR setup: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'HELP'
Usage: ./setup.sh [--python PATH] [--cann-root PATH] [--cann-env PATH]
                  [--jobs N] [-- MODEL [vllm serve arguments...]]

Build and install the grouped RGDR plugin in an existing Ascend310P3
vLLM 0.23.0 / vLLM Ascend 0.23.0 or 0.23.0rc1 environment.
With arguments after --, start vLLM serve immediately.
HELP
}

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
rgdr_python=python3
cann_root=
cann_env=
jobs=8
while (($#)); do
    case $1 in
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        --python|--cann-root|--cann-env|--jobs)
            (($# >= 2)) && [[ -n $2 && $2 != --* ]] || fail "$1 requires a value"
            case $1 in
                --python) rgdr_python=$2 ;;
                --cann-root) cann_root=$2 ;;
                --cann-env) cann_env=$2 ;;
                --jobs) jobs=$2 ;;
            esac
            shift 2
            ;;
        *) fail "unknown option $1 (see --help)" ;;
    esac
done
[[ $jobs =~ ^[1-9][0-9]*$ ]] || fail "--jobs must be a positive integer"
[[ $(uname -s) == Linux ]] || fail "run this script inside the target Linux 310P3 image"

has_ascendc() {
    [[ -n $1 && ( -f $1/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake ||
                  -f $1/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake ) ]]
}

if [[ -z $cann_root ]]; then
    for candidate in "${ASCEND_HOME_PATH:-}" "${ASCEND_TOOLKIT_HOME:-}" "${ASCEND_CANN_PACKAGE_PATH:-}" \
        /usr/local/Ascend/cann /usr/local/Ascend/ascend-toolkit/latest \
        "$HOME/Ascend/cann" "$HOME/Ascend/ascend-toolkit/latest"; do
        if has_ascendc "$candidate"; then
            cann_root=$candidate
            break
        fi
    done
fi
if [[ -z $cann_root ]]; then
    while IFS= read -r candidate; do
        if has_ascendc "$candidate"; then
            cann_root=$candidate
            break
        fi
    done < <(printf '%s\n' /usr/local/Ascend/cann-* "$HOME"/Ascend/cann-* | sort -Vr)
fi
has_ascendc "$cann_root" || fail "CANN compiler not found. Pass --cann-root to a Toolkit with ascendc.cmake."
if [[ -z $cann_env ]]; then
    for candidate in "$cann_root/set_env.sh" "$(dirname -- "$cann_root")/set_env.sh"; do
        if [[ -f $candidate ]]; then
            cann_env=$candidate
            break
        fi
    done
fi
[[ -f $cann_env ]] || fail "CANN set_env.sh not found. Pass its path with --cann-env."
cann_env="$(cd -- "$(dirname -- "$cann_env")" && pwd -P)/$(basename -- "$cann_env")"
cann_root=$(cd -- "$cann_root" && pwd -P)
source "$cann_env" || fail "failed to load $cann_env"
[[ ${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}} -ef $cann_root ]] ||
    fail "$cann_env activated a different Toolkit. Pass matching --cann-root and --cann-env."

rgdr_python=$(command -v -- "$rgdr_python") || fail "Python not found. Activate the vLLM environment or pass --python."
# Preserve the interpreter symlink in a venv since it identifies that venv.
rgdr_python=$("$rgdr_python" -c 'import sys; print(sys.executable)')
command -v cmake >/dev/null || fail "CMake 3.19 or newer is required in the target environment"
"$rgdr_python" -m pip --version >/dev/null || fail "pip is required in $rgdr_python"

cann_version=unknown
for info in "$cann_root/compiler/version.info" "$cann_root/version.info" \
    "$cann_root/$(uname -m)-linux/ascend_toolkit_install.info"; do
    if [[ -f $info ]]; then
        version=$(sed -n 's/^[Vv]ersion=//p' "$info" | sed -n '1p')
        if [[ -n $version ]]; then
            cann_version=$version
            break
        fi
    fi
done
printf 'CANN Toolkit: %s (%s)\nPython: %s\n' "$cann_root" "$cann_version" "$rgdr_python"
PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" "$rgdr_python" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

from ascend310p_rgdr_patch import _check_versions, _configured_cores, _configured_mode

try:
    _check_versions()
    _configured_cores()
    _configured_mode()
    for package in ("vllm", "vllm-ascend", "torch", "torch-npu"):
        print(f"{package}: {version(package)}", flush=True)
except (PackageNotFoundError, RuntimeError, ValueError) as error:
    raise SystemExit(f"RGDR setup: {error}. Select the Python environment of the 310P vLLM image.") from None

import torch_npu
from vllm_ascend.utils import is_310p

if not is_310p():
    raise SystemExit("RGDR setup: the installed vLLM Ascend build does not target 310P")
PY

build_root="$repo/build/rgdr-serve"
mkdir -p -- "$build_root"
build_dir=$(mktemp -d "$build_root/build.XXXXXX")
cmake -S "$repo/native-src" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DSOC_VERSION=Ascend310P3 -DASCEND_CANN_PACKAGE_PATH="$cann_root" \
    -DPython3_EXECUTABLE="$rgdr_python"
cmake --build "$build_dir" --target rgdr_v310_cube_fp16_torch --parallel "$jobs"
library="$build_dir/librgdr_v310_cube_fp16_torch.so"
[[ -f $library ]] || fail "build completed without $library"

"$rgdr_python" -m pip install --no-deps --upgrade "$repo"
vendor_dir=$("$rgdr_python" - <<'PY'
from importlib.util import find_spec
from pathlib import Path

ascend = find_spec("vllm_ascend")
vendor = Path(ascend.origin).parent / "_cann_ops_custom" / "vendors" / "custom_transformer"
if vendor.is_dir():
    print(vendor.resolve())
PY
)

# Write a sourceable environment in the build dir, then publish the launcher
# after verifying the installed package and extension in a separate process.
rgdr_env="$build_dir/env.sh"
{
    printf '# Generated by setup.sh. Source in Bash.\n'
    printf 'source %q || return 1\n' "$cann_env"
    printf 'if [[ ! ${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}} -ef %q ]]; then\n' "$cann_root"
    printf '    printf "RGDR: the active CANN Toolkit changed; rerun setup.sh before launch.\\n" >&2\n    return 1\nfi\n'
    printf 'export ASCEND310P_RGDR_PATCH_LIBRARY=%q\n' "$library"
    if [[ -n $vendor_dir ]]; then
        printf 'case :${ASCEND_CUSTOM_OPP_PATH:-}: in\n    *:%q:*) ;;\n' "$vendor_dir"
        printf '    *) export ASCEND_CUSTOM_OPP_PATH=%q${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH} ;;\nesac\n' "$vendor_dir"
    fi
    cat <<'BASH'
export ASCEND310P_RGDR_PATCH_CORES="${ASCEND310P_RGDR_PATCH_CORES:-8}"
export ASCEND310P_RGDR_PATCH_MODE="${ASCEND310P_RGDR_PATCH_MODE:-decode}"
if [[ ${VLLM_PLUGINS+x} ]]; then
    case ",$VLLM_PLUGINS," in
        *,ascend,*) ;;
        *) export VLLM_PLUGINS="${VLLM_PLUGINS:+$VLLM_PLUGINS,}ascend" ;;
    esac
    case ",$VLLM_PLUGINS," in
        *,ascend310p_grouped_rgdr,*) ;;
        *) export VLLM_PLUGINS="$VLLM_PLUGINS,ascend310p_grouped_rgdr" ;;
    esac
fi
BASH
} > "$rgdr_env"

(
    source "$rgdr_env"
    "$rgdr_python" - <<'PY'
from importlib import import_module
from importlib.metadata import entry_points

import torch_npu

from ascend310p_rgdr_patch import _GDN_MODULE, _HELPER, _configured_cores, _configured_mode

plugin, = entry_points(group="vllm.general_plugins", name="ascend310p_grouped_rgdr")
plugin.load()()
helper = getattr(import_module(_GDN_MODULE), _HELPER)
if not getattr(helper, "_ascend310p_grouped_rgdr", False):
    raise SystemExit("RGDR setup: vLLM plugin did not register")
print(f"RGDR loaded: grouped baseline, cores={_configured_cores()}, mode={_configured_mode()}")
PY
)

launcher="$build_root/vllm-serve"
{
    printf '#!/usr/bin/env bash\nset -eo pipefail\nsource %q\n' "$rgdr_env"
    printf 'exec %q -m vllm.entrypoints.cli.main serve "$@"\n' "$rgdr_python"
} > "$build_dir/vllm-serve"
chmod +x "$build_dir/vllm-serve"
mv -f -- "$build_dir/vllm-serve" "$launcher"

printf 'Environment: %s\nRun: %q MODEL [vllm serve arguments...]\n' "$rgdr_env" "$launcher"
if (($#)); then
    exec "$launcher" "$@"
fi
