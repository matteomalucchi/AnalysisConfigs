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

# All extra args initially
raw_extra_args=("$@")

# Detect and remove --test, --debug, and parse --copy-to
is_test=false
is_debug=false
copy_destination=""
expect_copy_path=false
extra_args=()

for arg in "${raw_extra_args[@]}"; do
    case "$arg" in
        --test)
            is_test=true
            ;;
        --debug)
            is_debug=true
            ;;
        --copy-to)
            expect_copy_path=true
            ;;
        *)
            if [[ "$expect_copy_path" == true ]]; then
                copy_destination="$arg"
                expect_copy_path=false
            else
                extra_args+=("$arg")
            fi
            ;;
    esac
done

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

# Construct command
cmd=( pocket-coffea run
      --cfg "$new_config_template"
      --custom-run-options "$run_options"
      -o "$output"
      --process-separately
)

###################################
# No executor in test mode
###################################
if ! $is_test; then
    [[ -n "$EXECUTOR" ]] && cmd+=( -e "$EXECUTOR" )
    [[ -n "$EXECUTOR_CUSTOM_SETUP" ]] && cmd+=( $EXECUTOR_CUSTOM_SETUP )
fi

###################################
# ✔️ If test mode, add the flag
###################################
$is_test && cmd+=( --test )

# Append cleaned extra args
cmd+=( "${extra_args[@]}" )

# Print the full command exactly as executed
echo "${cmd[@]}"

###################################
# DEBUG MODE → run without trapping
###################################
if $is_debug; then
    # "${cmd[@]}"
    # status=$?
    # cleanup
    exit $status
fi

###################################
# NORMAL MODE → run in background
###################################
"${cmd[@]}" &
pid=$!

wait $pid
status=$?


###################################
# OPTIONAL: COPY OUTPUT DIRECTORY
###################################
if [[ -n "$copy_destination" ]]; then
    echo "Copying output directory with rsync to: $copy_destination"

    mkdir -p "$copy_destination"

    # Actual copy
    rsync -avh --delete "$output"/ "$copy_destination"/
    rsync_status=$?

    if [[ $rsync_status -ne 0 ]]; then
        echo "ERROR: rsync failed. Not deleting original output."
        cleanup
        exit 1
    fi

    echo "Verifying copy using rsync checksum dry-run..."

    rsync -avhc --dry-run "$output"/ "$copy_destination"/ > /tmp/rsync_check_${i}.log
    verify_status=$?

    if [[ $verify_status -ne 0 ]]; then
        echo "ERROR: rsync checksum verification failed!"
        cleanup
        exit 1
    fi

    # Check differences
    if grep -q -v "^sending incremental file list" /tmp/rsync_check_${i}.log; then
        echo "ERROR: Differences detected after checksum!"
        cleanup
        exit 1
    fi

    echo "Checksum OK. Removing original output directory."
    rm -rf "$output"
fi

cleanup
exit $status
