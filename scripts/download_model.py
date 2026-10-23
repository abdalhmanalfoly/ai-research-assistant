"""Intentionally download/cache the embedding model and run a smoke test.

  python -m scripts.download_model            # downloads if missing (needs internet once), then embeds a test text
  python -m scripts.download_model --offline  # fails if the model is not already cached
"""
import argparse
import time

from app.core.config import get_settings
from app.main import build_embedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    st = get_settings()
    emb = build_embedder(st)
    emb.offline = emb.offline or a.offline
    print("cache status before:", emb.availability())
    t = time.time()
    emb.load()
    v = emb.embed_query_sync("smoke test")
    print(f"ok: model={emb.model} device={emb.device} dtype={emb.dtype} dim={len(v)} "
          f"load+embed={time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
