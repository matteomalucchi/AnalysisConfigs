#!/usr/bin/env bash

cleanup() {
    echo "Cleaning up..."
    [ -f "$new_config_template" ] && rm -f "$new_config_template"
    [ -f "$onnx_exec" ] && rm -f "$onnx_exec"
}

trap 'cleanup; kill -- "$pid"; exit 1' SIGINT SIGTERM

# Required args
config_options=${1%.py}
config_template=$2
run_options=$3
output=$4
shift 4

# Additional args passed directly to pocket-coffea
extra_args=("$@")

# Script location
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Find free run number
i=0
while true; do
    candidate="${SCRIPT_DIR}/../__onnx_executor_${i}__.py"
    if [ ! -f "$candidate" ]; then
        echo "Found free Run number ${i}"
        onnx_exec="$candidate"
        break
    fi
    i=$((i+1))
done

cp "${SCRIPT_DIR}/../onnx_executor_common.py" "$onnx_exec"
new_config_template="__${i}_${config_template}"
cp "$config_template" "$new_config_template"

# Replace placeholder
sed -i "s/__config_file__/${config_options}/g" "$onnx_exec"
sed -i "s/__config_file__/${config_options}/g" "$new_config_template"

# Determine executor
hostname=$(hostname)
case "$hostname" in
    *t3*)
        EXECUTOR="dask@T3_CH_PSI"
        EXECUTOR_CUSTOM_SETUP="--executor-custom-setup ${onnx_exec}"
        ;;
    *lxplus*)
        EXECUTOR="dask@lxplus"
        EXECUTOR_CUSTOM_SETUP=""
        ;;
    *)
        echo "WARNING: Unknown hostname '$hostname', no executor set."
        EXECUTOR=""
        EXECUTOR_CUSTOM_SETUP=""
        ;;
esac

echo "Using executor: $EXECUTOR"

# Build base command
cmd=( pocket-coffea run
      --cfg "$new_config_template"
      --custom-run-options "$run_options"
      -o "$output"
      --process-separately
)

# Add executor if available
if [[ -n "$EXECUTOR" ]]; then
    cmd+=( -e "$EXECUTOR" )
fi

# Add custom setup
if [[ -n "$EXECUTOR_CUSTOM_SETUP" ]]; then
    cmd+=( $EXECUTOR_CUSTOM_SETUP )
fi

# Add extra args (e.g. --test, --debug, etc.)
cmd+=( "${extra_args[@]}" )

# Detect test mode
is_test=false
for arg in "${extra_args[@]}"; do
    [[ "$arg" == "--test" ]] && is_test=true
done

# If test mode, insert --test into command (if not already included)
if $is_test; then
    cmd=( pocket-coffea run
          --cfg "$new_config_template"
          --custom-run-options "$run_options"
          -o "$output"
          --process-separately
          --test
          "${extra_args[@]}"
    )
fi

# Print command exactly as executed
echo "${cmd[@]}"

# Skip execution in debug mode
if printf '%s\n' "${extra_args[@]}" | grep -q -- '--debug'; then
    exit 0
fi

# Run
"${cmd[@]}" &
pid=$!

wait $pid
status=$?

cleanup
exit $status
