"""Unit tests for the BGE embedder using a FAKE model object (no torch/sentence-transformers, no download).
They verify our wrapper's behaviour, NOT the real model's embedding quality."""
import asyncio
import math
import unittest

from app.core.errors import (EmbeddingDimensionMismatch, EmbeddingUnavailable, InvalidEmbedding,
                             InvalidEmbeddingInput)
from app.services.providers.embeddings import HuggingFaceEmbedder, resolve_device
from tests.fixtures.fakes import FakeSTModel

INSTR = "Represent this sentence for searching relevant passages: "


def make(model=None, **kw):
    model = model or FakeSTModel()
    loads = []

    def loader(name, device, cache, offline, dtype="float32"):
        loads.append((name, device, cache, offline, dtype))
        return model

    kw.setdefault("cuda_available", lambda: False)
    e = HuggingFaceEmbedder("BAAI/bge-large-en-v1.5", 1024, model_loader=loader, **kw)
    return e, model, loads


class EmbedderTests(unittest.TestCase):
    def test_vectors_are_1024_finite_normalised(self):
        e, *_ = make()
        for v in e.embed_documents_sync(["alpha beta", "gamma"]) + [e.embed_query_sync("q")]:
            self.assertEqual(len(v), 1024)
            self.assertTrue(all(isinstance(x, float) and math.isfinite(x) for x in v))
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0, places=5)

    def test_batch_order_preserved_across_batches(self):
        e, m, _ = make(batch_size=3)
        texts = [f"distinct text number{i} word{i}" for i in range(10)]
        out = e.embed_documents_sync(texts)
        self.assertEqual(out, [m.vector(t) for t in texts])
        self.assertEqual([len(c[0]) for c in m.calls], [3, 3, 3, 1])  # bounded batches

    def test_empty_and_invalid_inputs(self):
        e, m, loads = make()
        self.assertEqual(e.embed_documents_sync([]), [])
        self.assertEqual(loads, [])  # nothing to do -> model not loaded
        for bad in (["ok", ""], ["  \n"], [None], [123]):
            with self.assertRaises(InvalidEmbeddingInput):
                e.embed_documents_sync(bad)
        for bad in ("", "   ", None):
            with self.assertRaises(InvalidEmbeddingInput):
                e.embed_query_sync(bad)
        with self.assertRaises(InvalidEmbeddingInput):
            e.embed_documents_sync("not a list")

    def test_nan_inf_wrong_dim_count_and_norm_rejected(self):
        for mode, exc in (("nan", InvalidEmbedding), ("inf", InvalidEmbedding), ("short", EmbeddingDimensionMismatch),
                          ("drop", InvalidEmbedding), ("unnormalised", InvalidEmbedding)):
            e, *_ = make(FakeSTModel(mode=mode))
            with self.assertRaises(exc, msg=mode):
                e.embed_documents_sync(["a b", "c d"])

    def test_unnormalised_allowed_when_normalize_false(self):
        e, *_ = make(FakeSTModel(mode="unnormalised"), normalize=False)
        self.assertEqual(len(e.embed_documents_sync(["a b"])[0]), 1024)

    def test_query_instruction_applied_exactly_once(self):
        e, m, _ = make()
        e.embed_query_sync("what is x?")
        e.embed_query_sync(INSTR + "already prefixed")
        self.assertEqual(m.calls[0][0], [INSTR + "what is x?"])
        self.assertEqual(m.calls[1][0], [INSTR + "already prefixed"])
        self.assertEqual(m.calls[0][0][0].count(INSTR), 1)

    def test_documents_never_get_query_instruction(self):
        e, m, _ = make()
        e.embed_documents_sync(["a passage about retinas"])
        self.assertEqual(m.calls[0][0], ["a passage about retinas"])
        e2, m2, _ = make(document_prefix="doc: ")
        e2.embed_documents_sync(["x y"])
        e2.embed_query_sync("x y")
        self.assertEqual(m2.calls[0][0], ["doc: x y"])
        self.assertTrue(m2.calls[1][0][0].startswith(INSTR))
        self.assertNotIn("doc: ", m2.calls[1][0][0])

    def test_encode_args(self):
        e, m, _ = make(batch_size=7, normalize=True)
        e.embed_documents_sync(["a b"])
        kw = m.calls[0][1]
        self.assertEqual((kw["batch_size"], kw["normalize_embeddings"], kw["show_progress_bar"]), (7, True, False))
        self.assertEqual(m.max_seq_length, 512)
        self.assertTrue(m.evaluated)

    def test_model_loaded_once_and_async_wrappers(self):
        e, m, loads = make()
        asyncio.run(e.embed_query("a b"))
        asyncio.run(e.embed_documents(["c d"]))
        e.embed_documents_sync(["e f"])
        self.assertEqual(len(loads), 1)
        self.assertEqual(e.availability(), "loaded")

    def test_device_resolution(self):
        self.assertEqual(resolve_device("auto", True), "cuda")
        self.assertEqual(resolve_device("auto", False), "cpu")
        self.assertEqual(resolve_device("cpu", True), "cpu")
        self.assertEqual(resolve_device("cuda:1", True), "cuda:1")
        for bad, avail in (("cuda", False), ("cuda:0", False), ("tpu", True), ("", True)):
            with self.assertRaises(EmbeddingUnavailable):
                resolve_device(bad, avail)

    def test_device_passed_to_loader(self):
        e, _, loads = make(device="auto", cuda_available=lambda: True)
        e.load()
        self.assertEqual(loads[0][1], "cuda")
        self.assertEqual(e.device, "cuda")

    def test_float16_cast_only_on_cuda(self):
        e, m, loads = make(device="auto", cuda_available=lambda: True, dtype="float16")
        e.load()
        self.assertTrue(m.half_called)
        self.assertEqual(e.dtype, "float16")
        self.assertEqual(loads[0][4], "float16")  # loader can load directly in fp16

        e2, m2, _ = make(device="cpu", cuda_available=lambda: True, dtype="float16")
        with self.assertLogs("app.services.providers.embeddings", level="WARNING") as logs:
            e2.load()
        self.assertFalse(m2.half_called)
        self.assertIn("only supported on CUDA", "".join(logs.output))

    def test_invalid_dtype_rejected(self):
        with self.assertRaises(ValueError):
            HuggingFaceEmbedder("m", 1024, dtype="float64")

    def test_cuda_requested_but_missing_fails_clearly(self):
        e, *_ = make(device="cuda")
        with self.assertRaises(EmbeddingUnavailable) as ctx:
            e.load()
        self.assertIn("CUDA", str(ctx.exception))

    def test_load_errors_translated(self):
        def boom(*a):
            raise OSError("no network /secret/path")
        e = HuggingFaceEmbedder("m", 1024, model_loader=boom, cuda_available=lambda: False)
        with self.assertRaises(EmbeddingUnavailable) as ctx:
            e.load()
        self.assertNotIn("secret", str(ctx.exception))
        e = HuggingFaceEmbedder("m", 1024, offline=True, model_loader=boom, cuda_available=lambda: False)
        with self.assertRaises(EmbeddingUnavailable) as ctx:
            e.load()
        self.assertIn("offline", str(ctx.exception))

        def missing(*a):
            raise ImportError("sentence_transformers")
        e = HuggingFaceEmbedder("m", 1024, model_loader=missing, cuda_available=lambda: False)
        with self.assertRaises(EmbeddingUnavailable) as ctx:
            e.load()
        self.assertIn("pip install", str(ctx.exception))

    def test_inference_error_translated_without_leaking_text(self):
        e, *_ = make(FakeSTModel(mode="raise"))
        with self.assertLogs("app.services.providers.embeddings", level="ERROR") as logs:
            with self.assertRaises(EmbeddingUnavailable) as ctx:
                e.embed_documents_sync(["confidential paper sentence"])
        self.assertNotIn("confidential", str(ctx.exception) + "".join(logs.output))

    def test_model_dimension_mismatch_detected_at_load(self):
        e, *_ = make(FakeSTModel(dim=768))
        with self.assertRaises(EmbeddingDimensionMismatch):
            e.load()

    def test_availability_without_loading(self):
        e, _, loads = make()
        self.assertIn(e.availability(), {"cached", "not_cached", "unavailable"})
        self.assertEqual(loads, [])


try:
    from app.core.config import Settings
    HAVE_PYDANTIC = True
except ImportError:
    HAVE_PYDANTIC = False


@unittest.skipUnless(HAVE_PYDANTIC, "pydantic-settings not installed")
class SettingsTests(unittest.TestCase):
    def test_defaults_are_bge_1024(self):
        s = Settings(_env_file=None)
        self.assertEqual((s.embedding_provider, s.embedding_model, s.embedding_dim, s.embedding_device),
                         ("huggingface", "BAAI/bge-large-en-v1.5", 1024, "auto"))
        self.assertEqual(s.embedding_dtype, "float32")
        self.assertTrue(s.embedding_normalize)

    def test_cors_wildcard_rejected(self):
        with self.assertRaises(Exception):
            Settings(_env_file=None, cors_origins="*")
        self.assertEqual(Settings(_env_file=None, cors_origins="http://localhost:5500").cors_origins, "http://localhost:5500")

    def test_validation(self):
        for kw in ({"embedding_device": "gpu"}, {"embedding_dtype": "float64"}, {"embedding_dim": 768},
                   {"embedding_provider": "ollama"}, {"embedding_batch_size": 0},
                   {"embedding_max_seq_length": 1024}):
            with self.assertRaises(Exception, msg=str(kw)):
                Settings(_env_file=None, **kw)
        self.assertEqual(Settings(_env_file=None, embedding_device="CUDA:1").embedding_device, "cuda:1")
        self.assertEqual(Settings(_env_file=None, embedding_dtype="FLOAT16").embedding_dtype, "float16")


if __name__ == "__main__":
    unittest.main()
