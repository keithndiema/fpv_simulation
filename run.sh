#!/usr/bin/env bash
# FPV Sim launcher - one command does everything:
#   ./run.sh                 set up (first time only), build the C++ core, start the simulator
#   ./run.sh --demo          autopilot demo flight
#   ./run.sh --selftest      run the regression tests (no window needed)
#   ./run.sh --fullscreen    start fullscreen
#   ./run.sh --reset         delete the virtual environment + C++ build and start clean
# Any other flag is passed straight to run_sim.py (see ./run.sh --help-sim).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

VENV=".venv"
STAMP="$VENV/.requirements.stamp"
bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '\033[36m[fpv]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[fpv] %s\033[0m\n' "$*" >&2; exit 1; }

if [[ "${1:-}" == "--reset" ]]; then
  info "removing $VENV and build/"
  rm -rf "$VENV" build
  shift
  [[ $# -eq 0 ]] && { info "clean. Run ./run.sh to set up again."; exit 0; }
fi
[[ "${1:-}" == "--help-sim" ]] && { shift; set -- --help; }

# ---- 1. Python -------------------------------------------------------------------------------
PY="$(command -v python3 || command -v python || true)"
[[ -n "$PY" ]] || die "Python 3 not found. Install it:  sudo apt install python3 python3-venv"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || die "Python 3.9+ required (found $($PY -V))."

# ---- 2. C++ toolchain (needed once, to build the physics core) --------------------------------------
if ! command -v cmake >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
  die "No C++ compiler found. Install one:
       Debian/Ubuntu:  sudo apt install build-essential cmake
       Fedora:         sudo dnf install gcc-c++ cmake
       macOS:          xcode-select --install"
fi

# ---- 3. Virtual environment (avoids the 'externally managed' / permission problems of system pip) ----
if [[ ! -x "$VENV/bin/python" ]]; then
  info "creating virtual environment in $VENV"
  "$PY" -m venv "$VENV" 2>/dev/null || die "Could not create a virtual environment. Install it:  sudo apt install python3-venv"
fi
VPY="$VENV/bin/python"

# ---- 4. Dependencies: prebuilt wheels only, so no SDL / compiler headaches ------------------------------
want="$(cksum < requirements.txt)-$("$VPY" -c 'import sys; print(sys.version_info[:2])')"
if [[ ! -f "$STAMP" || "$(cat "$STAMP")" != "$want" ]]; then
  info "installing dependencies (pygame-ce + numpy, prebuilt wheels)"
  "$VPY" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || true
  if ! "$VPY" -m pip install --only-binary=:all: -r requirements.txt; then
    die "Could not install prebuilt wheels for your Python ($("$VPY" -V)).
       - Check your internet connection, or
       - use a slightly older Python (3.11-3.13):  sudo apt install python3.12 python3.12-venv
         then:  ./run.sh --reset   with   PYTHON=python3.12 (edit the PY= line) ."
  fi
  echo "$want" > "$STAMP"
fi

# ---- 5. Display / audio hints ---------------------------------------------------------------------------
if [[ "${1:-}" != "--selftest" && "${1:-}" != "--build" && "${1:-}" != "--bench" ]]; then
  if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" && "$(uname)" == "Linux" ]]; then
    die "No display found (are you in plain SSH?). Run this on the desktop, or use  ssh -X ."
  fi
fi

# ---- 6. Go (the C++ core is built / rebuilt automatically by run_sim.py when needed) -------------------
bold "FPV Sim  -  arm: SPACE   throttle: W/S or left stick   help: F1"
exec "$VPY" run_sim.py "$@"
