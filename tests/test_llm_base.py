"""Tests for src/llm/base.py."""

from __future__ import annotations

import json
import unittest

from src.llm.base import chunk_batch_requests


class TestChunkBatchRequests(unittest.TestCase):
    """Batch submissions are capped by payload size, not just by request count."""

    BUDGET = 10_000

    def _req(self, i: int, n_chars: int, char: str = "x") -> dict:
        return {"custom_id": f"r__{i}", "params": {"messages": [{"role": "user", "content": char * n_chars}]}}

    def _chunk_bytes(self, chunk) -> int:
        return sum(len(json.dumps(r).encode()) for r in chunk)

    def test_splits_on_request_count(self):
        chunks = chunk_batch_requests([self._req(i, 10) for i in range(10)], 4, self.BUDGET)
        self.assertEqual([len(c) for c in chunks], [4, 4, 2])

    def test_splits_on_byte_budget_before_count_limit(self):
        # Each request is ~1/3 the budget, so the count limit never binds.
        chunks = chunk_batch_requests([self._req(i, self.BUDGET // 3) for i in range(8)], 1000, self.BUDGET)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(self._chunk_bytes(chunk), self.BUDGET)

    def test_non_ascii_escapes_expand_the_payload(self):
        """An em-dash is one source character but six payload bytes once escaped."""
        # 4 requests of BUDGET/12 source chars each would fit in one chunk if
        # sized off the model output; escaped they are BUDGET/2 apiece, so a
        # correct packer sizes the serialized request and emits at least two.
        chunks = chunk_batch_requests([self._req(i, self.BUDGET // 12, char="—") for i in range(4)], 1000, self.BUDGET)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(self._chunk_bytes(chunk), self.BUDGET)

    def test_every_request_survives_in_order(self):
        requests = [self._req(i, 100) for i in range(25)]
        chunks = chunk_batch_requests(requests, 7, self.BUDGET)
        self.assertEqual([r for c in chunks for r in c], requests)

    def test_empty_input(self):
        self.assertEqual(chunk_batch_requests([], 10, self.BUDGET), [])

    def test_oversized_single_request_gets_its_own_chunk(self):
        requests = [self._req(0, 10), self._req(1, self.BUDGET * 2), self._req(2, 10)]
        chunks = chunk_batch_requests(requests, 100, self.BUDGET)
        self.assertEqual([len(c) for c in chunks], [1, 1, 1])

    def test_budget_is_per_call_not_global(self):
        """Each backend passes its own cap; the same requests chunk differently."""
        requests = [self._req(i, 1_000) for i in range(10)]
        self.assertGreater(
            len(chunk_batch_requests(requests, 1000, 2_000)),
            len(chunk_batch_requests(requests, 1000, 20_000)),
        )


if __name__ == "__main__":
    unittest.main()
