import os
from typing import List, Optional

import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, ConversionResult
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer

# Define available embedding models
embedding_models = {
    # "sentence-mpnet": {
    #     "name": "all-mpnet-base-v2",
    #     "dimensions": 768,
    #     "type": "sentence-transformers",
    #     "description": "MPNet-based model, strong general-purpose embeddings with good performance for semantic search."
    # },
    # "sentence-minilm": {
    #     "name": "all-MiniLM-L6-v2",
    #     "dimensions": 384,
    #     "type": "sentence-transformers",
    #     "description": "Lightweight model with 384 dimensions. Fastest option with decent performance."
    # },
    # "bge-small": {
    #     "name": "BAAI/bge-small-en-v1.5",
    #     "dimensions": 384,
    #     "type": "sentence-transformers",
    #     "description": "Another lightweight and efficient sentence transformer model."
    # },
    # "jina-base": {
    #     "name": "jinaai/jina-embeddings-v2-base-en",
    #     "dimensions": 768,
    #     "type": "sentence-transformers",
    #     "description": "Jina AI's base English embedding model, good balance of performance and efficiency."
    # },
    # "jina-v3": {
    #     "name": "jinaai/jina-embeddings-v3",
    #     "description": "state-of-the-art multilingual text embedding model. It features 570 million parameters and supports input sequences up to 8192 tokens."
    # }
}


# --------------------------------------------------------------
# 1. Prepare document links and filenames
# --------------------------------------------------------------
links = [
    "https://www.galileo.ai/blog/mastering-agents-langgraph-vs-autogen-vs-crew",
    "https://www.linkedin.com/pulse/langgraph-detailed-technical-exploration-ai-workflow-jagadeesan-n9woc",
    "https://towardsdatascience.com/from-basics-to-advanced-exploring-langgraph-e8c1cf4db787",
    "https://medium.com/@hao.l/why-langgraph-stands-out-as-an-exceptional-agent-framework-44806d969cc6",
    "https://pub.towardsai.net/revolutionizing-project-management-with-ai-agents-and-langgraph-ff90951930c1",
    "https://github.com/langchain-ai/langgraph",
]

sources = [
    "galileo_ai",
    "linkedin",
    "towards_data_science",
    "medium",
    "towards_ai",
    "github",
]

# --------------------------------------------------------------
# 2. Convert all docs and chunk each one
# --------------------------------------------------------------
converter = DocumentConverter()
conversion_results: List[ConversionResult] = converter.convert_all(links)

for model_id in list(embedding_models.keys()):
    model_info = embedding_models[model_id]
    model_name = model_info["name"]
    model = SentenceTransformer(model_name)
    print(f"Start Embedding using {model_name} max_tokens = {model.max_seq_length}")
    chunker = HybridChunker(tokenizer=model.tokenizer)

    all_chunks = []

    for index, result in enumerate(conversion_results):
        dl_doc = result.document
        chunk_iter = chunker.chunk(dl_doc=dl_doc)
        chunks = list(chunk_iter)
        if not chunks:
            print(f"⚠️ No chunks produced for doc: {dl_doc.meta.origin.uri} (length={len(dl_doc.text)})")
        for chunk in chunks:
            chunk.meta.origin.uri = sources[index]
        all_chunks.extend(chunks)

    # --------------------------------------------------------------
    # 3. LanceDB setup: one table for all documents
    # --------------------------------------------------------------
    db = lancedb.connect("data/lancedb")

    # Get the appropriate embedding function based on model type
    embedding_func = get_registry().get("sentence-transformers").create(name=model_name, device="cpu")

    # Define the LanceDB schema
    class Chunks(LanceModel):
        filename: str
        source: str
        text: str = embedding_func.SourceField()
        title: Optional[str]
        vector: Vector(model.get_sentence_embedding_dimension()) = embedding_func.VectorField()

    # Create the table with a name reflecting the model
    table_name = f"lang_graph_{model_name.replace('/','_')}"
    if not os.path.isdir(f"data/lancedb/{table_name}.lance"):
        print('creating db')
        table = db.create_table(table_name, schema=Chunks, mode="overwrite")
        # Prepare data for insertion
        processed_chunks = []
        for chunk in all_chunks:
            processed_chunks.append({
                "text": chunk.text,
                "source": chunk.meta.origin.uri,
                "filename": chunk.meta.origin.filename,
                "title": chunk.meta.headings[0] if chunk.meta.headings else None
            })

        if processed_chunks:
            table.add(processed_chunks)
            print(f"✅ Added {len(processed_chunks)} chunks to LanceDB table '{table_name}'")
        else:
            print(f"⚠️ No chunks found for model {model_name}, skipping insertion.")
    else:
        print("Already exists")