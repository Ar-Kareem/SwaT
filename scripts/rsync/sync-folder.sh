#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 (--to-tall | --to-akim) [--no-dry-run] [--skip-ext EXT]... path/to/folder" >&2
    exit 2
}

dry_run=(--dry-run)
excludes=()
skipped_exts=()
add_skip_ext() {
    local ext="${1#.}"
    [[ -n "$ext" && "$ext" != */* ]] || usage
    excludes+=(--exclude="*.$ext")
    skipped_exts+=(".$ext")
}
folder=""
target_host=""
while [[ $# -gt 0 ]]; do
    arg="$1"
    shift
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
        --no-dry-run) dry_run=() ;;
        --skip-ext)
            [[ $# -gt 0 ]] || usage
            add_skip_ext "$1"
            shift
            ;;
        --skip-ext=*) add_skip_ext "${arg#--skip-ext=}" ;;
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
    if [[ ! -d "$folder" ]]; then
        echo "Source folder does not exist: $folder" >&2
        exit 2
    fi
    source="$folder"
    destination="$target_host:$folder"
elif [[ "$current_host" == "$target_host" ]]; then
    source="$source_host:$folder"
    destination="$folder"
else
    echo "Run this script on akimiski or tallurutit." >&2
    exit 2
fi

format_files() {
    awk '
        function render(dir,    limit, i, entry, block, rows, n, j, total, output) {
            limit = (dir == "" || parents[dir] ? 25 : 5)
            for (i = 1; i <= counts[dir]; i++) {
                entry = entries[dir, i]
                block = (entry in directories ? render(entry) : "  " entry "\n")
                n = split(block, rows, "\n")
                for (j = 1; j < n; j++) {
                    if (++total <= limit) output = output rows[j] "\n"
                }
            }
            if (total > limit) output = output "  " dir "... (" total - limit " more)\n"
            return output
        }
        /^[<>]f|^c[LDS]/ {
            path = substr($0, 13)
            n = split(path, parts, "/")
            dir = ""
            for (i = 1; i < n; i++) {
                child = dir parts[i] "/"
                if (!(child in directories)) {
                    directories[child] = 1
                    entries[dir, ++counts[dir]] = child
                    parents[dir] = 1
                }
                dir = child
            }
            entries[dir, ++counts[dir]] = path
        }
        END {
            if (!counts[""]) print "  None"
            else printf "%s", render("")
        }
    '
}

mode="COPY"
if [[ ${#dry_run[@]} -gt 0 ]]; then
    mode="DRY RUN"
fi
printf '\n%s -> %s | %s\n' "$source_host" "$target_host" "$mode"
if [[ ${#skipped_exts[@]} -gt 0 ]]; then
    printf 'Skipping extensions: %s\n' "${skipped_exts[*]}"
fi
printf '\n'
comparison=$(rsync -rclnii --existing --protect-args --mkpath "${excludes[@]}" \
    --out-format='%i %n' -- "$source" "$destination")
matches=$(printf '%s\n' "$comparison" | awk '/^\.[fL]/ { n++ } END { print n+0 }')
mismatches=$(printf '%s\n' "$comparison" | awk '/^[<>]f|^c[LDS]/ { n++ } END { print n+0 }')
printf 'Matches (already identical): %s\n\n' "$matches"
printf 'Mismatches (skipped): %s\n' "$mismatches"
printf '%s\n' "$comparison" | format_files
missing=$(rsync -ah --ignore-existing --protect-args --mkpath "${excludes[@]}" "${dry_run[@]}" \
    --out-format='%i %n' -- "$source" "$destination")
missing_count=$(printf '%s\n' "$missing" | awk '/^[<>]f|^c[LDS]/ { n++ } END { print n+0 }')
printf '\nMissing files (%s): %s\n' "$mode" "$missing_count"
printf '%s\n' "$missing" | format_files
if [[ ${#dry_run[@]} -gt 0 ]]; then
    printf '\nUse --no-dry-run to copy missing files.\n'
fi
