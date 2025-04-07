from typing import List, Optional
import lancedb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, ConversionResult
from dotenv import load_dotenv
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from openai import OpenAI
from utils.tokenizer import OpenAITokenizerWrapper
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI and tokenizer
client = OpenAI(api_key=api_key)
open_ai_tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = 8191

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

chunker = HybridChunker(
    tokenizer=open_ai_tokenizer,
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

all_chunks = []

for index, result in enumerate(conversion_results):
    dl_doc = result.document
    chunk_iter = chunker.chunk(dl_doc=dl_doc)
    chunks = list(chunk_iter)
    for chunk in chunks:
        chunk.meta.origin.uri = sources[index]
    all_chunks.extend(chunks)

# --------------------------------------------------------------
# 3. LanceDB setup: one table for all documents
# --------------------------------------------------------------
db = lancedb.connect("data/lancedb")
embedding_func = get_registry().get("openai").create(name="text-embedding-3-large")


class Chunks(LanceModel):
    filename: str
    source: str
    text: str = embedding_func.SourceField()
    title: Optional[str]
    vector: Vector(embedding_func.ndims()) = embedding_func.VectorField()

table = db.create_table("lang_graph_text-embedding-3-small", schema=Chunks, mode="overwrite")

# Modified data preparation
processed_chunks = []
for chunk in all_chunks:
    processed_chunks.append({
        "text": chunk.text,
        "source": chunk.meta.origin.uri,
        "filename": chunk.meta.origin.filename,
        "title": chunk.meta.headings[0] if chunk.meta.headings else None
    })

table.add(processed_chunks)
print("✅ Added", len(processed_chunks), "chunks to LanceDB table 'lang_graph_text-embedding-3-small'")