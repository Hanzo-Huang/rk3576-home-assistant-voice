#!/usr/bin/env python3
"""Download one RK3576 Piper voice from a direct Hugging Face directory URL."""

import argparse
import json
import shutil
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def download_model(url: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("encoder.onnx", "decoder_rk3576.rknn", "config.json"):
        target = output / name
        temporary = output / f"{name}.part"
        for attempt in range(3):
            try:
                print(f"Downloading {name} (attempt {attempt + 1}/3)", flush=True)
                with urlopen(f"{url.rstrip('/')}/{name}", timeout=120) as response:
                    with temporary.open("wb") as destination:
                        shutil.copyfileobj(response, destination)
                if temporary.stat().st_size == 0:
                    raise ValueError(f"Empty model file: {name}")
                if name == "config.json":
                    with temporary.open(encoding="utf-8") as config_file:
                        json.load(config_file)
                temporary.replace(target)
                print(f"Downloaded {name}: {target.stat().st_size} bytes", flush=True)
                break
            except (OSError, URLError, ValueError):
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Direct model directory URL")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    download_model(args.url, args.output)
