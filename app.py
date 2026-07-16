"""Deployment entry point (HF Spaces / Railway / any PaaS) — thin wrapper around app/gradio_app.py."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from gradio_app import demo  # noqa: E402

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
