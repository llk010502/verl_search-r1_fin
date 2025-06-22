# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local document indexing utilities for the search tool."""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

from .search_r1_like_utils import _passages2string


def slice_document(document: str, chunk_size: int = 200, stride: int | None = None) -> List[str]:
    """Split a document into overlapping chunks.

    Args:
        document: The full document string.
        chunk_size: Maximum number of tokens per chunk.
        stride: Step size between chunks. Defaults to ``chunk_size`` (no overlap).

    Returns:
        List of document chunks.
    """
    tokens = document.split()
    if stride is None:
        stride = chunk_size
    slices: List[str] = []
    for start in range(0, len(tokens), stride):
        chunk_tokens = tokens[start : start + chunk_size]
        if not chunk_tokens:
            break
        slices.append(" ".join(chunk_tokens))
        if start + chunk_size >= len(tokens):
            break
    return slices


class DocumentIndexer:
    """A lightweight TF-IDF indexer for local document search."""

    def __init__(self, document: str, chunk_size: int = 200, stride: int | None = None) -> None:
        self.chunks = slice_document(document, chunk_size, stride)
        self._build_index()

    def _build_index(self) -> None:
        self._doc_counters = [Counter(c.lower().split()) for c in self.chunks]
        df: Counter[str] = Counter()
        for c in self._doc_counters:
            df.update(c.keys())
        self._idf = {t: math.log((1 + len(self._doc_counters)) / (1 + df[t])) + 1 for t in df}

    def _tfidf(self, counts: Counter[str]) -> Dict[str, float]:
        return {t: f * self._idf.get(t, 0.0) for t, f in counts.items() if t in self._idf}

    @staticmethod
    def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        common = set(v1) & set(v2)
        num = sum(v1[t] * v2[t] for t in common)
        denom1 = math.sqrt(sum(v * v for v in v1.values()))
        denom2 = math.sqrt(sum(v * v for v in v2.values()))
        if denom1 == 0 or denom2 == 0:
            return 0.0
        return num / (denom1 * denom2)

    def _vectorize_query(self, query: str) -> Dict[str, float]:
        return self._tfidf(Counter(query.lower().split()))

    def batch_search(self, query_list: List[str], topk: int = 3) -> Tuple[str, Dict[str, Any]]:
        """Search the indexed document for the given queries.

        Args:
            query_list: Queries to search for.
            topk: Number of top chunks to return for each query.

        Returns:
            ``result_text`` and ``metadata`` mimicking the remote API format.
        """
        all_retrievals: List[List[Dict[str, Any]]] = []
        for query in query_list:
            q_vec = self._vectorize_query(query)
            sims = []
            for doc_counter in self._doc_counters:
                d_vec = self._tfidf(doc_counter)
                sims.append(self._cosine(q_vec, d_vec))
            if sims:
                idxs = np.argsort(sims)[::-1][:topk]
                retrieval = [
                    {"document": {"contents": self.chunks[i]}, "score": float(sims[i])}
                    for i in idxs
                ]
            else:
                retrieval = []
            all_retrievals.append(retrieval)

        total_results = sum(len(r) for r in all_retrievals)
        pretty_results = [
            _passages2string(r) for r in all_retrievals if r
        ]
        final_result = "\n---\n".join(pretty_results) if pretty_results else None

        metadata = {
            "query_count": len(query_list),
            "queries": query_list,
            "api_request_error": None,
            "api_response": None,
            "status": "success" if total_results > 0 else "no_results",
            "total_results": total_results,
            "formatted_result": final_result,
        }

        if final_result:
            result_text = json.dumps({"result": final_result})
        else:
            result_text = json.dumps({"result": "No search results found."})
        return result_text, metadata
