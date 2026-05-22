"""Unit tests for retrieval pipeline — filter extraction only (no vector DB needed)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# Stub out heavy dependencies before importing retrieval
def _stub_embedders():
    mock_text = MagicMock()
    mock_text.embed.return_value = [0.0] * 384
    mock_image = MagicMock()
    mock_image.embed_text_for_image_search.return_value = [0.0] * 512
    return mock_text, mock_image


with patch("src.embeddings.models.get_text_embedder") as mt, \
     patch("src.embeddings.models.get_image_embedder") as mi, \
     patch("src.indexing.vector_store.get_vector_store") as mvs:
    mock_te, mock_ie = _stub_embedders()
    mt.return_value = mock_te
    mi.return_value = mock_ie
    mvs.return_value = MagicMock()
    from src.retrieval.pipeline import RetrievalPipeline


class TestFilterExtraction:
    def setup_method(self):
        with patch("src.embeddings.models.get_text_embedder"), \
             patch("src.embeddings.models.get_image_embedder"), \
             patch("src.indexing.vector_store.get_vector_store"):
            self.pipeline = RetrievalPipeline()

    def test_no_filters_for_generic_query(self):
        filters = self.pipeline._extract_filters("What are the latest lab results?")
        assert filters == {}

    def test_extracts_xray_modality(self):
        filters = self.pipeline._extract_filters("Show me the x-ray for the patient")
        assert filters.get("modality") == "CR"

    def test_extracts_ct_modality(self):
        filters = self.pipeline._extract_filters("Find the CT scan from last week")
        assert filters.get("modality") == "CT"

    def test_extracts_mri_modality(self):
        filters = self.pipeline._extract_filters("MRI report for the brain")
        assert filters.get("modality") == "MR"

    def test_extracts_date_with_dashes(self):
        filters = self.pipeline._extract_filters("blood test from 2024-03-15")
        assert filters.get("study_date") == "20240315"

    def test_extracts_date_without_dashes(self):
        filters = self.pipeline._extract_filters("results dated 20240315")
        assert filters.get("study_date") == "20240315"

    def test_extracts_patient_id(self):
        filters = self.pipeline._extract_filters("patient id P12345 lung infection")
        assert "patient_id" in filters
        assert "p12345" in filters["patient_id"].lower()

    def test_combined_filters(self):
        filters = self.pipeline._extract_filters(
            "Show x-ray for patient P99 on 2023-06-01"
        )
        assert filters.get("modality") == "CR"
        assert filters.get("study_date") == "20230601"


class TestMergeResults:
    def setup_method(self):
        with patch("src.embeddings.models.get_text_embedder"), \
             patch("src.embeddings.models.get_image_embedder"), \
             patch("src.indexing.vector_store.get_vector_store"):
            self.pipeline = RetrievalPipeline()

    def _make_hit(self, asset_id, asset_type, score, page=1):
        return {
            "score": score,
            "id": hash(asset_id),
            "payload": {
                "asset_id": asset_id,
                "document_id": "doc-001",
                "filename": "test.pdf",
                "page_number": page,
                "asset_type": asset_type,
                "snippet": f"Snippet for {asset_id}",
            },
        }

    def test_deduplication(self):
        hit = self._make_hit("a1", "text", 0.9)
        sources = self.pipeline._merge_results([hit], [hit])
        assert len(sources) == 1

    def test_sorted_by_score_descending(self):
        hits_text = [
            self._make_hit("t1", "text", 0.8),
            self._make_hit("t2", "text", 0.5),
        ]
        hits_image = [self._make_hit("i1", "image", 0.95)]
        sources = self.pipeline._merge_results(hits_text, hits_image)
        scores = [s.score for s in sources]
        assert scores == sorted(scores, reverse=True)

    def test_empty_inputs(self):
        sources = self.pipeline._merge_results([], [])
        assert sources == []

    def test_image_and_text_hits_included(self):
        hits_text = [self._make_hit("t1", "text", 0.9)]
        hits_image = [self._make_hit("i1", "image", 0.85)]
        sources = self.pipeline._merge_results(hits_text, hits_image)
        types = {s.type.value for s in sources}
        assert "text" in types
        assert "image" in types
