set quiet

alias b := build
alias r1 := runv1
alias r2 := runv2
alias r3 := runv3
alias r4 := runv4
alias c := clean
alias rb := rebuild

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
    [ -f {{venv_dir}}/bin/walkv1 ] || [ -f {{venv_dir}}/bin/walkv2 ] || {{pip}} install . -q

# Rebuild the project
rebuild: make_venv
    {{pip}} install --upgrade . -q

# Run the project
runv1 *ARGS: build
    {{venv_dir}}/bin/walkv1 {{ARGS}}

runv2 *ARGS: build
    {{venv_dir}}/bin/walkv2 {{ARGS}}

runv3 *ARGS: build
    {{venv_dir}}/bin/walkv3 {{ARGS}}

runv4 *ARGS: build
    {{venv_dir}}/bin/walkv4 {{ARGS}}

# Clean build files and virtual environment
[no-quiet]
clean:
    rm -rf target
    rm -rf {{venv_dir}}
