import pytest
from app.models.embedding import EmbeddedBlock
from app.models.search import SearchFilter
from app.services.vector_db.chroma_service import ChromaService


@pytest.fixture
def temp_chroma(tmp_path):
    """Fixture providing a fresh ChromaService pointing to an isolated temp directory."""
    db_path = tmp_path / "chroma_test"
    service = ChromaService(persist_dir=db_path, collection_name="test_collection")
    yield service
    # Teardown: clear collection
    service.clear_collection()


def create_sample_block(
    block_id: str,
    text: str,
    embedding: list[float],
    page: int = 1,
    section: str = "Introduction",
    paper_meta: dict | None = None,
) -> EmbeddedBlock:
    return EmbeddedBlock(
        block_id=block_id,
        text=text,
        page_number=page,
        section_title=section,
        embedding=embedding,
        metadata=paper_meta or {"authors": ["Author One", "Author Two"], "year": 2023},
    )


def test_initialization_and_empty_state(temp_chroma):
    assert temp_chroma.total_blocks == 0
    stats = temp_chroma.get_stats()
    assert stats["total_blocks"] == 0
    assert stats["metric"] == "cosine"
    assert stats["collection_name"] == "test_collection"

    # Querying empty DB returns empty list
    results = temp_chroma.query_similar(query_embedding=[0.1] * 4, top_k=3)
    assert results == []


def test_index_and_retrieve_blocks(temp_chroma):
    # Two orthogonal 4-D vectors
    b1 = create_sample_block("b1", "Attention is all you need for sequence models", [1.0, 0.0, 0.0, 0.0], page=1)
    b2 = create_sample_block("b2", "Deep bidirectional transformer for language understanding", [0.0, 1.0, 0.0, 0.0], page=2)

    count = temp_chroma.index_embedded_blocks([b1, b2], paper_id="transformer_paper", paper_title="Transformers")
    assert count == 2
    assert temp_chroma.total_blocks == 2

    # Query matching b1 perfectly
    results = temp_chroma.query_similar(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    top_hit = results[0]
    assert top_hit.block_id == "b1"
    assert top_hit.paper_id == "transformer_paper"
    assert top_hit.page_number == 1
    assert top_hit.section_title == "Introduction"
    assert top_hit.score >= 0.99  # Cosine similarity ~ 1.0
    assert "Author One" in top_hit.metadata.get("authors", "")


def test_metadata_filtering_paper_id(temp_chroma):
    b1 = create_sample_block("b1", "Attention paper chunk", [0.9, 0.1, 0.0, 0.0], page=1)
    b2 = create_sample_block("b2", "BERT paper chunk", [0.95, 0.05, 0.0, 0.0], page=1)

    temp_chroma.index_embedded_blocks([b1], paper_id="attention_paper")
    temp_chroma.index_embedded_blocks([b2], paper_id="bert_paper")

    # Query vector is slightly closer to b2, but filter scopes strictly to attention_paper
    query_vec = [1.0, 0.0, 0.0, 0.0]
    results = temp_chroma.query_similar(
        query_embedding=query_vec,
        top_k=5,
        where={"paper_id": "attention_paper"},
    )

    assert len(results) == 1
    assert results[0].block_id == "b1"
    assert results[0].paper_id == "attention_paper"


def test_structured_search_filter(temp_chroma):
    b1 = create_sample_block("b1", "Page 1 intro", [0.8, 0.2, 0.0, 0.0], page=1)
    b2 = create_sample_block("b2", "Page 3 methodology", [0.85, 0.15, 0.0, 0.0], page=3, section="Methodology")
    b3 = create_sample_block("b3", "Page 7 results", [0.9, 0.1, 0.0, 0.0], page=7, section="Experiments")

    temp_chroma.index_embedded_blocks([b1, b2, b3], paper_id="paper_xyz")

    # Scope to pages 2 through 5
    filt = SearchFilter(paper_id="paper_xyz", min_page=2, max_page=5)
    results = temp_chroma.query_similar(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        search_filter=filt,
    )

    assert len(results) == 1
    assert results[0].block_id == "b2"
    assert results[0].page_number == 3


def test_upsert_idempotency(temp_chroma):
    b1 = create_sample_block("b1", "Initial draft text", [0.5, 0.5, 0.0, 0.0])
    temp_chroma.index_embedded_blocks([b1], paper_id="paper_1")
    assert temp_chroma.total_blocks == 1

    # Re-index with modified text
    b1_updated = create_sample_block("b1", "Final reviewed text", [0.6, 0.4, 0.0, 0.0])
    temp_chroma.index_embedded_blocks([b1_updated], paper_id="paper_1")

    # Count must still be 1 (no duplicate insertion)
    assert temp_chroma.total_blocks == 1

    results = temp_chroma.query_similar(query_embedding=[0.6, 0.4, 0.0, 0.0], top_k=1)
    assert results[0].text == "Final reviewed text"


def test_delete_paper_lifecycle(temp_chroma):
    p1_blocks = [
        create_sample_block("p1_b1", "Paper 1 intro", [1.0, 0.0, 0.0, 0.0]),
        create_sample_block("p1_b2", "Paper 1 methods", [0.9, 0.1, 0.0, 0.0]),
    ]
    p2_blocks = [
        create_sample_block("p2_b1", "Paper 2 intro", [0.0, 1.0, 0.0, 0.0]),
    ]

    temp_chroma.index_embedded_blocks(p1_blocks, paper_id="paper_to_delete")
    temp_chroma.index_embedded_blocks(p2_blocks, paper_id="paper_to_keep")
    assert temp_chroma.total_blocks == 3

    # Delete paper 1
    deleted_count = temp_chroma.delete_paper("paper_to_delete")
    assert deleted_count == 2
    assert temp_chroma.total_blocks == 1

    # Verify querying for paper 1 returns empty
    results = temp_chroma.query_similar(
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        where={"paper_id": "paper_to_delete"},
    )
    assert results == []

    # Remaining paper is intact
    summary = temp_chroma.list_indexed_papers()
    assert len(summary) == 1
    assert summary[0]["paper_id"] == "paper_to_keep"


def test_persistence_across_restarts(tmp_path):
    db_dir = tmp_path / "persistent_chroma"

    # Step 1: Write data with first client instance
    client1 = ChromaService(persist_dir=db_dir, collection_name="persist_col")
    b = create_sample_block("b_persisted", "Persistent chunk", [0.7, 0.7, 0.0, 0.0])
    client1.index_embedded_blocks([b], paper_id="perm_paper")
    assert client1.total_blocks == 1
    del client1  # Simulate app shutdown

    # Step 2: Open second client instance pointing to same directory
    client2 = ChromaService(persist_dir=db_dir, collection_name="persist_col")
    assert client2.total_blocks == 1

    results = client2.query_similar(query_embedding=[0.7, 0.7, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].block_id == "b_persisted"
    assert results[0].text == "Persistent chunk"
