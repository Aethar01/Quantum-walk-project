set quiet

alias b := build
alias r := run
alias c := clean

# Variables
venv_dir := ".venv"
venv_python := venv_dir / "bin/python"
pip := venv_dir / "bin/pip"

# Default target
default:
    just --list

# Create virtual environment
make_venv:
    [ -x "$(command -v python)" ] || { echo "Python is not installed"; exit 1; }
    [ -x "$(command -v python -m venv)" ] || { echo "Python venv module is not installed"; exit 1; }
    [ -d {{venv_dir}} ] || python -m venv {{venv_dir}}

# Build the project
build: make_venv
    [ -f {{venv_dir}}/bin/quantum-walk-project ] || {{pip}} install .

# Rebuild the project
rebuild: make_venv
    {{pip}} install --upgrade --force-reinstall .

# Run the project
run *ARGS: build
    {{venv_dir}}/bin/quantum-walk-project {{ARGS}}

# Clean build files and virtual environment
[no-quiet]
clean:
    rm -rf target
    rm -rf {{venv_dir}}
