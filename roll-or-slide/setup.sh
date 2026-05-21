#!/bin/bash
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install flask
echo ""
echo "Done. To start the server:"
echo "  source .venv/bin/activate"
echo "  python server.py"
echo ""
echo "Then open http://localhost:5072"
