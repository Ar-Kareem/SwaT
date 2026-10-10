outputs_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/multiagent/outputs_mine"
rsync -avh --ignore-existing --progress "akimiski:$outputs_dir/" "$outputs_dir/"
