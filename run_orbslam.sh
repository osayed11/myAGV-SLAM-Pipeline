#!/usr/bin/env bash
# Convenience runner for ORB-SLAM2 in this coursework repo.
#
# Usage:
#   ./run_orbslam.sh help
#   ./run_orbslam.sh runplaygroundlong
#   ./run_orbslam.sh runplayground
#   ./run_orbslam.sh runclassroom
#   ./run_orbslam.sh runlobby
#   ./run_orbslam.sh run <yaml> <dataset_dir> [output_file]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MONO_TUM="/Users/omarahmed/orb_slam2_build/Install/bin/mono_tum"

# Dynamic dataset evaluation only

die() {
	echo "Error: $*" >&2
	exit 1
}

check_prereqs() {
	[ -x "$MONO_TUM" ] || die "mono_tum not found/executable at: $MONO_TUM"
}

run_orb() {
	local yaml="$1"
	local dataset_dir="$2"
	local output_file="$3"

	[ -f "$yaml" ] || die "YAML not found: $yaml"
	[ -d "$dataset_dir" ] || die "Dataset directory not found: $dataset_dir"
	[ -f "$dataset_dir/rgb.txt" ] || die "Missing rgb.txt in: $dataset_dir"
	[ -e "$dataset_dir/rgb" ] || die "Missing rgb/ in: $dataset_dir"

	mkdir -p "$(dirname "$output_file")"

	local yaml_abs="$(cd "$(dirname "$yaml")" && pwd)/$(basename "$yaml")"
	local dataset_abs="$(cd "$dataset_dir" && pwd)"
	local output_abs="$(cd "$(dirname "$output_file")" && pwd)/$(basename "$output_file")"

	echo "Running ORB-SLAM2 mono_tum"
	echo "  YAML:    $yaml_abs"
	echo "  Dataset: $dataset_abs"
	echo "  Output:  $output_abs"

	cd "$ROOT_DIR"
	"$MONO_TUM" "$yaml_abs" "$dataset_abs" "$output_abs"
}

print_help() {
	cat <<'EOF'
run_orbslam.sh - quick runner for this repo

Commands:
	help
		Show this message.

	run <yaml> <dataset_dir> [output_file]
		Fully custom run.
		If output_file is omitted, uses: <dataset_dir>/orb_slam_results.txt

Examples:
	./question2/run_orbslam.sh runplaygroundlong
	./question2/run_orbslam.sh runplayground
	./question2/run_orbslam.sh runclassroom
	./question2/run_orbslam.sh runlobby
EOF
}

main() {
	check_prereqs

	local cmd="${1:-help}"
	case "$cmd" in
		help|-h|--help)
			print_help
			;;
		run)
			[ "$#" -ge 3 ] || die "Usage: $0 run <yaml> <dataset_dir> [output_file]"
			local yaml="$2"
			local dataset_dir="${3%/}"
			local dname="$(basename "$dataset_dir")"
			local out="${4:-$ROOT_DIR/SLAM/slam_output/$dname/orb_slam_results.txt}"
			run_orb "$yaml" "$dataset_dir" "$out"
			;;
		*)
			die "Unknown command: $cmd (use: $0 help)"
			;;
	esac
}

main "$@"
