#!/bin/sh
# Entrypoint wrapper so the image accepts demo flags directly.
#
#   docker run ... inference-audit-demo            -> python inference_audit_demo.py
#   docker run ... inference-audit-demo --sweep    -> python inference_audit_demo.py --sweep
#   docker run ... inference-audit-demo pytest -q  -> pytest -q
#
# If the first argument looks like a flag, treat the whole arg list as demo
# flags; otherwise exec it verbatim (so `pytest`, `python ...`, `sh`, etc. work).
set -e

if [ "${1#-}" != "$1" ]; then
    set -- python inference_audit_demo.py "$@"
fi

exec "$@"
