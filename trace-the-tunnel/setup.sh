#!/bin/bash
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install flask "numpy<2" scipy pandas scikit-learn matplotlib torch
echo ""
echo "Done. To start the server:"
echo "  source .venv/bin/activate"
echo "  python server.py"
