#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 (--to-tall | --to-akim) path/to/folder" >&2
    exit 2
}

folder=""
target_host=""
for arg in "$@"; do
    case "$arg" in
        --to-tall)
            [[ -z "$target_host" ]] || usage
            target_host=tallurutit
            source_host=akimiski
            ;;
        --to-akim)
            [[ -z "$target_host" ]] || usage
            target_host=akimiski
            source_host=tallurutit
            ;;
        -*) usage ;;
        *)
            [[ -z "$folder" ]] || usage
            folder="$arg"
            ;;
    esac
done
[[ -n "$folder" && -n "$target_host" ]] || usage
folder=$(realpath --canonicalize-missing --no-symlinks -- "$folder")
folder="${folder%/}/"

current_host=$(hostname -s)
if [[ "$current_host" == "$source_host" ]]; then
    [[ -d "$folder" ]] || { echo "Source folder does not exist: $folder" >&2; exit 2; }
    source="$folder"
    destination="$target_host:$folder"
elif [[ "$current_host" == "$target_host" ]]; then
    source="$source_host:$folder"
    destination="$folder"
else
    echo "Run this script on akimiski or tallurutit." >&2
    exit 2
fi

printf '\n%s -> %s | %s\n\n' "$source_host" "$target_host" "$folder"
rsync -rclnii --protect-args --mkpath --out-format='%i %n' -- "$source" "$destination" | awk '
    function flush(    line, b) {
        if (current == "") return
        line = ""
        for (b = 1; b <= 3; b++) {
            if (!counts[buckets[b]]) continue
            line = line (line == "" ? "" : ", ") buckets[b]
            if (current ~ /\/$/) line = line " (" counts[buckets[b]] ")"
            totals[buckets[b]] += counts[buckets[b]]
        }
        printf "  %-40s %s\n", current, line
        fflush()
        delete counts
    }
    BEGIN { split("match mismatch missing", buckets, " ") }
    /^[<>c][fLDS]\+{9}/ { bucket = "missing" }
    /^\.[fL]/ { bucket = "match" }
    /^[<>]f[^+]|^c[LDS][^+]/ { bucket = "mismatch" }
    !/^[<>c.][fLDS]/ { next }
    {
        path = substr($0, 13)
        slash = index(path, "/")
        name = (slash ? substr(path, 1, slash) : path)
        if (name != current) { flush(); current = name }
        counts[bucket]++
    }
    END {
        flush()
        if (current == "") print "  None"
        printf "\n  Total: match %d, mismatch %d, missing %d\n",
            totals["match"], totals["mismatch"], totals["missing"]
    }
'
