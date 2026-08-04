"""Tests for modules/language/search.py and rag.py"""

from modules.language.search import search


def test_search_returns_expected_keys():
    results = search("invoice total", top_k=2)
    assert len(results) == 2
    for r in results:
        assert set(r.keys()) == {"chunk_text", "distance", "document_id", "filename", "doc_type"}


def test_search_finds_relevant_document_by_meaning():
    # "how much do I owe" shares no exact words with "Total Due" —
    # this specifically tests semantic (not keyword) search.
    results = search("how much do I owe", top_k=1)
    assert results[0]["doc_type"] == "invoice"


def test_search_respects_top_k():
    results = search("document", top_k=5)
    assert len(results) == 5

def test_search_caps_top_k_to_index_size():
    # Our index currently has 6 chunks total. Requesting more than
    # that should not error or duplicate — it should cap gracefully.
    results = search("documents", top_k=100)
    filenames = [r["filename"] for r in results]
    assert len(filenames) == len(set(filenames)), "search() returned duplicate documents"