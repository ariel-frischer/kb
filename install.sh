#!/bin/sh
set -eu

REPO="https://gitlab.com/ariel-frischer/kb.git"

main() {
    echo "Installing kb..."

    # Check for uv
    if ! command -v uv >/dev/null 2>&1; then
        echo "uv not found. Installing uv first..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Install kb as a global tool from git
    uv tool install --from "git+${REPO}" "kb[all]"

    echo ""
    echo "Done! Run 'kb --help' to get started."
}

main
