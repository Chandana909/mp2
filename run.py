#!/usr/bin/env python3
"""Run the ML Reliability Platform API server."""

import uvicorn
from platform.config import load_config, get_config

if __name__ == "__main__":
    load_config()
    host = get_config("serving.host") or "0.0.0.0"
    port = int(get_config("serving.port") or 8000)
    uvicorn.run(
        "serving.api:app",
        host=host,
        port=port,
        reload=False,
    )
