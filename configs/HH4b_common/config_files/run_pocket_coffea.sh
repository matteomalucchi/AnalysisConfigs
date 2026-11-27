#!/usr/bin/env bash

######################################################################
# SLURM JOB OF THE PARQUET MERGING (--merge-parquet*): edit by hand
######################################################################
# The merging of the parquet files (scripts/postprocess_parquet.py) can need a
# lot of memory, so it runs in a slurm job (sbatch --wait: this script waits
# for it before --copy-to). The job inherits the environment (--export=ALL),
# so run this script with the python environment activated. If the output is
# on /pnfs, X509_USER_PROXY must point to a valid proxy on a shared file
# system (e.g. /t3home). The temporary files of the merging are written in
# /scratch/<user> of the worker node (default of postprocess_parquet.py).
MERGE_USE_SLURM=true             # false: run the merging here, without slurm
MERGE_SLURM_PARTITION="standard"
MERGE_SLURM_TIME="08:00:00"
MERGE_SLURM_MEM="200G"
MERGE_SLURM_CPUS=8
# directory of the slurm logs of the merging jobs
MERGE_SLURM_LOG_DIR="${PWD}/slurm_logs_merge_parquet"
######################################################################

cleanup() {
    echo "Cleaning up..."
    [ -f "$new_config_template" ] && rm -f "$new_config_template"
    [ -f "$onnx_exec" ] && rm -f "$onnx_exec"
    [ -n "$merge_jobid_file" ] && [ -f "$merge_jobid_file" ] && rm -f "$merge_jobid_file"
}

# On Ctrl+C / kill / terminal closed: stop pocket-coffea (or cancel the slurm
# job of the merging), wait for it to exit, then remove the temporary files
on_signal() {
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping pocket-coffea (pid $pid)..."
        kill -TERM "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
    fi
    if [ -n "$merge_pid" ] && kill -0 "$merge_pid" 2>/dev/null; then
        merge_jobid=$(head -n 1 "$merge_jobid_file" 2>/dev/null | cut -d ';' -f 1)
        if [ -n "$merge_jobid" ]; then
            echo "Cancelling the slurm job of the merging ($merge_jobid)..."
            scancel "$merge_jobid"
        fi
        kill -TERM "$merge_pid" 2>/dev/null
        wait "$merge_pid" 2>/dev/null
    fi
    cleanup
    exit 1
}

trap on_signal SIGINT SIGTERM SIGHUP

# Required args
config_options=${1%.py}
config_template=$2
run_options=$3
output=$4
shift 4

# All extra args initially
raw_extra_args=("$@")

# Detect and remove --test, --debug, --merge-parquet* and parse --copy-to
is_test=false
is_debug=false
copy_destination=""
expect_copy_path=false
merge_parquet=false
merge_parquet_args=()
expect_merge_args=false
# --skip-parquet-check: do not check that the parquet output directories
# (dump_columns_as_arrays_per_chunk) of the datasets/regions/variations to run
# are empty before running (the check is also skipped with --resubmit-failed)
skip_parquet_check=false
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
        --merge-parquet)
            merge_parquet=true
            ;;
        --merge-parquet-update-config)
            merge_parquet=true
            merge_parquet_args+=( --update-config )
            ;;
        --merge-parquet-remove-original)
            # remove the original parquet files once merged, only if the number
            # of events of each dataset/region/variation is the same in the
            # original and merged files (--remove-original of postprocess_parquet.py)
            # It also updates the config path
            merge_parquet=true
            merge_parquet_args+=( --remove-original )
            merge_parquet_args+=( --update-config )
            ;;
        --merge-parquet-args)
            merge_parquet=true
            expect_merge_args=true
            ;;
        --skip-parquet-check)
            skip_parquet_check=true
            ;;
        *)
            if [[ "$expect_copy_path" == true ]]; then
                copy_destination="$arg"
                expect_copy_path=false
            elif [[ "$expect_merge_args" == true ]]; then
                # e.g. --merge-parquet-args "-j 16 -o <dir>"
                read -r -a _margs <<< "$arg"
                merge_parquet_args+=( "${_margs[@]}" )
                expect_merge_args=false
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
# CHECK THE PARQUET OUTPUT IS EMPTY
###################################
# If the config dumps the columns as parquet chunks (dump_columns_as_arrays_per_chunk,
# set from save_chunk), crash when <dump dir>/<dataset>/<region>/<variation> already
# holds parquet files: the new chunks would be mixed with (and partly overwrite)
# the ones of a previous run. Not done with --resubmit-failed (reuses the same
# directories), --skip-parquet-check or --debug.
if ! $is_debug && ! $skip_parquet_check && [[ ! " ${extra_args[*]} " =~ " --resubmit-failed " ]]; then
    check_cmd=( python "${SCRIPT_DIR}/check_parquet_output.py" --cfg "$new_config_template" )
    # same dataset filters as pocket-coffea run
    for ((k = 0; k < ${#extra_args[@]}; k++)); do
        case "${extra_args[$k]}" in
            --filter-datasets|--filter-samples|--filter-years)
                check_cmd+=( "${extra_args[$k]}" "${extra_args[$((k+1))]}" )
                ;;
            --filter-datasets=*|--filter-samples=*|--filter-years=*)
                check_cmd+=( "${extra_args[$k]}" )
                ;;
        esac
    done
    echo "${check_cmd[@]}"
    if ! "${check_cmd[@]}"; then
        echo "ERROR: parquet output check failed, not running pocket-coffea."
        cleanup
        exit 1
    fi
fi

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
# Run in background + wait so that the trap fires immediately on a signal.
# NB: a background job in a non-interactive shell ignores SIGINT, so on Ctrl+C
# the trap stops pocket-coffea with SIGTERM.
"${cmd[@]}" &
pid=$!
wait $pid
status=$?

###################################
# OPTIONAL: MERGE THE PARQUET FILES
###################################
# Done before the copy of the output directory, so that the copied
# config.json already points to the merged files (--merge-parquet-update-config)
if $merge_parquet; then
    if [[ $status -ne 0 ]]; then
        echo "WARNING: PocketCoffea failed (status $status), not merging the parquet files."
    else
        merge_cmd=( python "${SCRIPT_DIR}/../../../scripts/postprocess_parquet.py"
                    -i "$output" --merge "${merge_parquet_args[@]}" )
        echo "${merge_cmd[@]}"
        if $MERGE_USE_SLURM && command -v sbatch &> /dev/null; then
            mkdir -p "$MERGE_SLURM_LOG_DIR"
            if [[ "$output" == /pnfs* && -z "$X509_USER_PROXY" ]]; then
                echo "WARNING: X509_USER_PROXY not set: the merging job may not be able to write on /pnfs."
            fi
            printf -v merge_wrap '%q ' "${merge_cmd[@]}"
            merge_jobid_file="${MERGE_SLURM_LOG_DIR}/.jobid_${i}_$$"
            sbatch_cmd=( sbatch --wait --parsable --export=ALL
                         --partition "$MERGE_SLURM_PARTITION"
                         --time "$MERGE_SLURM_TIME"
                         --mem "$MERGE_SLURM_MEM"
                         --cpus-per-task "$MERGE_SLURM_CPUS"
                         --job-name "merge_parquet_$(basename "$output")"
                         --output "${MERGE_SLURM_LOG_DIR}/%x_%j.log"
                         --chdir "$PWD"
                         --wrap "$merge_wrap" )
            echo "Submitting the merging as a slurm job (partition $MERGE_SLURM_PARTITION," \
                 "time $MERGE_SLURM_TIME, mem $MERGE_SLURM_MEM, cpus $MERGE_SLURM_CPUS)," \
                 "logs in $MERGE_SLURM_LOG_DIR"
            # in background + wait, so that the trap can cancel the job
            "${sbatch_cmd[@]}" > "$merge_jobid_file" &
            merge_pid=$!
            wait $merge_pid
            merge_status=$?
            merge_jobid=$(head -n 1 "$merge_jobid_file" 2>/dev/null | cut -d ';' -f 1)
            echo "Slurm job $merge_jobid of the merging finished with status $merge_status," \
                 "log: ${MERGE_SLURM_LOG_DIR}/merge_parquet_$(basename "$output")_${merge_jobid}.log"
        else
            $MERGE_USE_SLURM && echo "WARNING: sbatch not found, merging here."
            "${merge_cmd[@]}"
            merge_status=$?
        fi
        if [[ $merge_status -ne 0 ]]; then
            echo "ERROR: merging of the parquet files failed. Not copying the output."
            cleanup
            exit 1
        fi
    fi
fi

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
