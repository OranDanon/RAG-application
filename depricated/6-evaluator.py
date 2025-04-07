from typing import List, Dict, Any
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from azure.core.credentials import AzureKeyCredential
from tqdm import tqdm
import lancedb
from lancedb.pydantic import LanceModel, Vector
from openai import AzureOpenAI
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from utils.tokenizer import OpenAITokenizerWrapper
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv
import glob
import utils.my_timer

# Load environment variables
load_dotenv()

# Initialize OpenAI client
# client = OpenAI()
client = AzureOpenAI(
    api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_API_KEY"))
)
open_ai_tokenizer = OpenAITokenizerWrapper()
open_ai_dim = 8191


# --------------------------------------------------------------
# Helper function for transformers embeddings
# --------------------------------------------------------------

def mean_pooling(model_output, attention_mask):
    """Mean pooling for transformers models"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_transformers_embedding(texts, model_name):
    """Get embeddings from transformers models"""
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Tokenize and prepare for the model
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)

    # Move input to GPU if available
    if device == "cuda":
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Get model output
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Mean pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Move back to CPU if needed
    if device == "cuda":
        embeddings = embeddings.cpu()

    return embeddings.numpy()


# --------------------------------------------------------------
# Define embedding models
# --------------------------------------------------------------

embedding_models = {
    # "openai": {
    #     "name": "text-embedding-3-large",
    #     "dimensions": 8191,
    #     "type": "openai",
    #     "description": "OpenAI's high-performance embedding model with 3072 dimensions. Best performance but most expensive."
    # },
    # "openai-small": {
    #     "name": "text-embedding-3-small",
    #     "dimensions": 8191,
    #     "type": "openai",
    #     "description": "OpenAI's smaller embedding model with 1536 dimensions. Good balance of performance and cost."
    # },
    # "sentence-e5": {
    #     "name": "intfloat/e5-large-v2",
    #     "dimensions": 1024,
    #     "type": "sentence-transformers",
    #     "description": "E5 large model, particularly good for retrieval tasks and asymmetric search (query-document)."
    # },
    # "sentence-mpnet": {
    #     "name": "all-mpnet-base-v2",
    #     "dimensions": 768,
    #     "type": "sentence-transformers",
    #     "description": "MPNet-based model, strong general-purpose embeddings with good performance for semantic search."
    # },
    "sentence-minilm": {
        "name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "type": "sentence-transformers",
        "description": "Lightweight model with 384 dimensions. Fastest option with decent performance."
    },
    # "bge-large": {
    #     "name": "BAAI/bge-large-en-v1.5",
    #     "dimensions": 1024,
    #     "type": "sentence-transformers",
    #     "description": "BAAI's BGE large model, state-of-the-art for English semantic search."
    # },
    # "bge-small": {
    #     "name": "BAAI/bge-small-en-v1.5",
    #     "dimensions": 384,
    #     "type": "sentence-transformers",
    #     "description": "Another lightweight and efficient sentence transformer model."
    # },
    # "gte-small": {
    #     "name": "thenlper/gte-small",
    #     "dimensions": 384,
    #     "type": "sentence-transformers",
    #     "description": "Small GTE embedding model, efficient with good performance."
    # },
    # "jina-base": {
    #     "name": "jinaai/jina-embeddings-v2-base-en",
    #     "dimensions": 768,
    #     "type": "sentence-transformers",
    #     "description": "Jina AI's base English embedding model, good balance of performance and efficiency."
    # },
    # "instructor-xl": {
    #     "name": "hkunlp/instructor-xl",
    #     "dimensions": 768,
    #     "type": "sentence-transformers",
    #     "description": "Instruction-tuned embedding model, adaptable to different retrieval tasks."
    # },
    # "bert-base": {
    #     "name": "bert-base-uncased",
    #     "dimensions": 768,
    #     "type": "transformers",
    #     "description": "BERT base model, great for general-purpose NLP tasks."
    # },
    # "distilbert": {
    #     "name": "distilbert-base-uncased",
    #     "dimensions": 768,
    #     "type": "transformers",
    #     "description": "DistilBERT model, a smaller and faster alternative to BERT with good accuracy."
    # }
}

# --------------------------------------------------------------
# Extract the data
# --------------------------------------------------------------

def extract_data(model_info):
    """Extracts data from specified links using DocumentConverter and applies hybrid chunking."""
    converter = DocumentConverter()

    links = [
        "https://www.galileo.ai/blog/mastering-agents-langgraph-vs-autogen-vs-crew",
        "https://www.linkedin.com/pulse/langgraph-detailed-technical-exploration-ai-workflow-jagadeesan-n9woc",
        "https://towardsdatascience.com/from-basics-to-advanced-exploring-langgraph-e8c1cf4db787",
        "https://medium.com/@hao.l/why-langgraph-stands-out-as-an-exceptional-agent-framework-44806d969cc6",
        "https://pub.towardsai.net/revolutionizing-project-management-with-ai-agents-and-langgraph-ff90951930c1",
        "https://github.com/langchain-ai/langgraph",
    ]

    # Extract the filename from the links
    filenames = [
        "galileo_ai.json",
        "linkedin.json",
        "towards_data_science.json",
        "medium.json",
        "towards_ai.json",
        "repo.json",
    ]

    # Convert all links to documents
    conversion_results = converter.convert_all(links)

    model_name = model_info['name']
    if model_info['type'] == 'transformers':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif model_info['type'] == 'sentence-transformers':
        tokenizer = SentenceTransformer(model_name).tokenizer
    elif model_name['type'] == 'openai':
        tokenizer = OpenAITokenizerWrapper()

    # Apply hybrid chunking
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=model_info['dimensions'],
        merge_peers=True,
    )

    all_chunks = []

    for result in conversion_results:
        dl_doc = result.document  # Access the document from the ConversionResult
        chunk_iter = chunker.chunk(dl_doc=dl_doc)
        chunks = list(chunk_iter)

        # Assign filenames to chunks
        for chunk in chunks:
            if not chunk.meta.origin.filename:
                for i, link in enumerate(links):
                    if link in chunk.meta.origin.url:
                        chunk.meta.origin.filename = filenames[i]
                        break

        all_chunks.extend(chunks)
        return all_chunks


# --------------------------------------------------------------
# Create embeddings and index them in LanceDB
# --------------------------------------------------------------

def create_embeddings_and_index(chunks, model_info, db_path="data/lancedb"):
    """Creates embeddings for the given chunks and indexes them in LanceDB."""
    model_name = model_info["name"]
    model_type = model_info["type"]
    dimensions = model_info["dimensions"]
    table_name = f"docling_{model_name.replace('/', '_')}"

    # Create directory if it doesn't exist
    os.makedirs(db_path, exist_ok=True)

    # Create a LanceDB database
    db = lancedb.connect(f"{db_path}/{model_name.replace('/', '_')}")

    # Check if table already exists
    if table_name in db.table_names():
        print(f"Using existing table for {model_name}")
        return db.open_table(table_name)

    # Process chunks
    processed_chunks = []

    for chunk in tqdm(chunks, desc=f"Processing chunks for {model_name}"):
        processed_chunk = {
            "text": chunk.text,
            "metadata": {
                "filename": chunk.meta.origin.filename,
                "page_numbers": [
                                    page_no for page_no in sorted(
                        set(prov.page_no for item in chunk.meta.doc_items for prov in item.prov)
                    )
                                ] or None,
                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
            },
        }
        processed_chunks.append(processed_chunk)

    # Get texts for embedding
    texts = [chunk["text"] for chunk in processed_chunks]

    # Compute embeddings based on model type
    if model_type == "openai":
        # Create vectors using OpenAI's API
        response = client.embeddings.create(
            model=model_name,
            input=texts
        )
        vectors = np.array([embedding.embedding for embedding in response.data])

    elif model_type == "sentence-transformers":
        # Use sentence transformers
        model = SentenceTransformer(model_name)
        vectors = model.encode(texts)

    elif model_type == "transformers":
        # Use transformers
        vectors = get_transformers_embedding(texts, model_name)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Add vectors to processed chunks
    for i, chunk in enumerate(processed_chunks):
        chunk["vector"] = vectors[i]

    # Define the schema
    class ChunkMetadata(LanceModel):
        filename: str | None
        page_numbers: List[int] | None
        title: str | None

    class Chunks(LanceModel):
        text: str
        vector: Vector(dimensions)
        metadata: ChunkMetadata

    # Create and populate the table
    table = db.create_table(table_name, schema=Chunks, mode="overwrite")
    table.add(processed_chunks)

    return db.open_table(table_name)


# --------------------------------------------------------------
# Create queries for evaluation from JSON files
# --------------------------------------------------------------

def create_test_queries():
    """Load test queries from JSON files in /evaluator directory"""
    queries = []

    # Get all JSON files in the /evaluator directory
    json_files = glob.glob("evaluator/*.json")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Process each entry in the JSON file
            for entry in data['questions']:
                if isinstance(entry, dict) and "question" in entry:
                    # Extract file_name correctly
                    file_name = os.path.basename(json_file).split('.')[0]
                    query = {
                        "query": entry["question"],
                        "expected": file_name,
                        "id": entry.get("id", "unknown"),
                        "difficulty": entry.get("difficulty", "unknown"),
                        "question_type": entry.get("question_type", "unknown")
                    }
                    queries.append(query)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"Loaded {len(queries)} test queries from evaluator files")
    return queries


# --------------------------------------------------------------
# Evaluate search performance
# --------------------------------------------------------------

def evaluate_search(table, queries, k_values=(1, 3, 5, 10, 25, 50)):
    """Evaluates search performance of the given table against the provided queries."""
    results = {k: 0 for k in k_values}
    total_queries = len(queries)

    # Track performance by difficulty and question type
    difficulty_results = {}
    question_type_results = {}

    for query_obj in tqdm(queries, desc="Evaluating queries"):
        query = query_obj["query"]
        expected = query_obj["expected"]
        difficulty = query_obj.get("difficulty", "unknown")
        question_type = query_obj.get("question_type", "unknown")

        # Initialize tracking dictionaries if needed
        if difficulty not in difficulty_results:
            difficulty_results[difficulty] = {k: 0 for k in k_values}
        if question_type not in question_type_results:
            question_type_results[question_type] = {k: 0 for k in k_values}

        # Search with the largest k
        max_k = max(k_values)
        search_results = table.search(query=query, query_type="vector").limit(max_k).to_pandas()

        # Check if expected document is in the results for each k
        for k in k_values:
            top_k_results = search_results.iloc[:k]
            filenames = [meta.get('filename') for meta in top_k_results['metadata']]
            if expected in filenames:
                results[k] += 1
                difficulty_results[difficulty][k] += 1
                question_type_results[question_type][k] += 1

    # Convert to recall@k
    recall = {k: results[k] / total_queries for k in k_values}

    # Convert difficulty and question type results to recall@k
    difficulty_counts = {}
    for difficulty in difficulty_results:
        count = sum(1 for q in queries if q.get("difficulty", "unknown") == difficulty)
        difficulty_counts[difficulty] = count
        for k in k_values:
            difficulty_results[difficulty][k] /= count if count > 0 else 1

    question_type_counts = {}
    for qtype in question_type_results:
        count = sum(1 for q in queries if q.get("question_type", "unknown") == qtype)
        question_type_counts[qtype] = count
        for k in k_values:
            question_type_results[qtype][k] /= count if count > 0 else 1

    return {
        "overall": recall,
        "by_difficulty": difficulty_results,
        "by_question_type": question_type_results,
        "difficulty_counts": difficulty_counts,
        "question_type_counts": question_type_counts
    }


# --------------------------------------------------------------
# Plot results
# --------------------------------------------------------------

def plot_results(all_results):
    """Plots the overall recall@k results for all models."""
    plt.figure(figsize=(12, 8))

    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'd', '*', 'x']

    for i, (model_id, result) in enumerate(all_results.items()):
        k_values = sorted(list(result["results"]["overall"].keys()))
        recall_values = [result["results"]["overall"][k] for k in k_values]

        plt.plot(k_values, recall_values, marker=markers[i % len(markers)],
                 color=colors[i % len(colors)], linewidth=2, label=f"{result['name']}")

    plt.title("Comparing Embedding Models (Recall@k)", fontsize=14)
    plt.xlabel("k", fontsize=12)
    plt.ylabel("Recall", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    plt.xticks(k_values, [str(k) for k in k_values])
    plt.ylim(0.0, 1.0)

    plt.savefig("recall@k.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_results_by_category(all_results, category_key, title_prefix):
    """Plot results by a specific category (difficulty or question_type)"""
    # Get all unique categories
    categories = set()
    for model_result in all_results.values():
        categories.update(model_result["results"][category_key].keys())

    # Remove 'unknown' category if it exists and others are available
    if 'unknown' in categories and len(categories) > 1:
        categories.remove('unknown')

    # Plot one graph per category
    for category in categories:
        plt.figure(figsize=(12, 8))

        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'd', '*', 'x']

        for i, (model_id, result) in enumerate(all_results.items()):
            if category in result["results"][category_key]:
                k_values = sorted(list(result["results"][category_key][category].keys()))
                recall_values = [result["results"][category_key][category][k] for k in k_values]

                plt.plot(k_values, recall_values, marker=markers[i % len(markers)],
                         color=colors[i % len(colors)], linewidth=2, label=f"{result['name']}")

        plt.title(f"{title_prefix}: {category.capitalize()}", fontsize=14)
        plt.xlabel("k", fontsize=12)
        plt.ylabel("Recall", fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.xscale('log')
        plt.xticks(k_values, [str(k) for k in k_values])
        plt.ylim(0.0, 1.0)

        plt.savefig(f"embedding_comparison_{category_key}_{category}.png", dpi=300, bbox_inches='tight')
        plt.show()


# --------------------------------------------------------------
# Print model recommendations and insights
# --------------------------------------------------------------

def print_model_insights(all_results):
    """Print insights and recommendations based on evaluation results"""
    print("\n" + "=" * 80)
    print("EMBEDDING MODEL ANALYSIS AND RECOMMENDATIONS")
    print("=" * 80)

    # Calculate overall average performance for each model
    model_avg_performance = {}
    for model_id, result in all_results.items():
        overall_results = result["results"]["overall"]
        avg_performance = sum(overall_results.values()) / len(overall_results)
        model_avg_performance[model_id] = avg_performance

    # Sort models by average performance
    sorted_models = sorted(model_avg_performance.items(), key=lambda x: x[1], reverse=True)

    # Print overall ranking
    print("\nOVERALL MODEL RANKING:")
    for i, (model_id, avg_perf) in enumerate(sorted_models):
        model_info = all_results[model_id]
        print(f"{i + 1}. {model_info['name']} - Average Recall: {avg_perf:.4f}")
        print(f"   Description: {model_info['description']}")

    # Print best model for each k
    print("\nBEST MODEL FOR EACH RECALL@K:")
    k_values = sorted(list(all_results[next(iter(all_results))]["results"]["overall"].keys()))
    for k in k_values:
        best_model_id = max(all_results.keys(), key=lambda m: all_results[m]["results"]["overall"][k])
        best_model = all_results[best_model_id]
        print(f"Recall@{k}: {best_model['name']} ({best_model['results']['overall'][k]:.4f})")

    # Analyze performance by difficulty
    print("\nPERFORMANCE BY DIFFICULTY LEVEL:")
    difficulty_levels = set()
    for result in all_results.values():
        difficulty_levels.update(result["results"].get("by_difficulty", {}).keys())

    for difficulty in sorted(difficulty_levels):
        if difficulty == "unknown":
            continue
        print(f"\n{difficulty.upper()} QUESTIONS:")
        best_model_id = max(
            all_results.keys(),
            key=lambda m: sum(all_results[m]["results"]["by_difficulty"].get(difficulty, {}).values()) /
                          len(all_results[m]["results"]["by_difficulty"].get(difficulty, {}).values() or [1])
        )
        best_model = all_results[best_model_id]
        avg_perf = sum(best_model["results"]["by_difficulty"].get(difficulty, {}).values()) / \
                   len(best_model["results"]["by_difficulty"].get(difficulty, {}).values() or [1])
        print(f"Best model: {best_model['name']} (Avg: {avg_perf:.4f})")

    # Print recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    best_overall_model_id = sorted_models[0][0]
    best_overall_model = all_results[best_overall_model_id]
    print(f"\n1. BEST OVERALL PERFORMANCE: {best_overall_model['name']}")
    print(f"   {best_overall_model['description']}")

    # Find best price/performance ratio
    if "openai" in best_overall_model_id:
        best_non_openai = next((m for m, _ in sorted_models if "openai" not in m), None)
        if best_non_openai:
            print(f"\n2. BEST OPEN-SOURCE ALTERNATIVE: {all_results[best_non_openai]['name']}")
            print(f"   {all_results[best_non_openai]['description']}")
            perf_diff = model_avg_performance[best_overall_model_id] - model_avg_performance[best_non_openai]
            print(
                f"   Performance gap vs. best model: {perf_diff:.4f} ({perf_diff / model_avg_performance[best_overall_model_id] * 100:.1f}% difference)")

    # Find fastest model with decent performance
    smallest_model_id = min(all_results.keys(), key=lambda m: all_results[m].get("dimensions", 1000000))
    smallest_model = all_results[smallest_model_id]
    print(f"\n3. FASTEST OPTION: {smallest_model['name']}")
    print(f"   {smallest_model['description']}")
    perf_diff = model_avg_performance[best_overall_model_id] - model_avg_performance[smallest_model_id]
    print(
        f"   Performance gap vs. best model: {perf_diff:.4f} ({perf_diff / model_avg_performance[best_overall_model_id] * 100:.1f}% difference)")


# --------------------------------------------------------------
# Main evaluation function
# --------------------------------------------------------------
@utils.my_timer.make_timer
def run_evaluation(models_to_evaluate=None):
    """Run the evaluation pipeline for selected models."""
    # Create test queries
    queries = create_test_queries()

    # Results for all models
    all_results = {}

    # If no models specified, use all models
    if models_to_evaluate is None:
        models_to_evaluate = list(embedding_models.keys())

    # Extract data once (only when needed)
    chunks_per_doc = None

    # Evaluate each embedding model
    for model_id in models_to_evaluate:
        model_info = embedding_models[model_id]
        print(f"Evaluating model: {model_info['name']}")

        # Try to use existing table first
        db_path = f"data/lancedb/{model_info['name'].replace('/', '_')}"
        if os.path.exists(db_path):
            try:
                # See if we can open the existing table
                table = create_embeddings_and_index(None, model_info)
            except:
                # If that fails, extract data and create table
                if chunks_per_doc is None:
                    chunks_per_doc = extract_data(model_info)
                table = create_embeddings_and_index(chunks_per_doc, model_info)
        else:
            # Table doesn't exist, extract data and create table
            if chunks_per_doc is None:
                chunks_per_doc = extract_data(model_info)
            table = create_embeddings_and_index(chunks_per_doc, model_info)

        # Evaluate search performance
        results = evaluate_search(table, queries)

        # Store results
        all_results[model_id] = {
            "name": model_info["name"],
            "description": model_info["description"],
            "dimensions": model_info["dimensions"],
            "results": results
        }

        print(f"Overall results for {model_info['name']}: {results['overall']}")

    # Plot results if we have any
    if all_results:
        plot_results(all_results)
        plot_results_by_category(all_results, "by_difficulty", "Difficulty Level")
        plot_results_by_category(all_results, "by_question_type", "Question Type")
        print_model_insights(all_results)

    return all_results


# --------------------------------------------------------------
# Run the evaluation
# --------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate embedding models')
    parser.add_argument('--models', nargs='+', help='Models to evaluate (e.g. openai openai-small)')
    parser.add_argument('--reset', action='store_true', help='Clear existing tables and recompute embeddings')
    args = parser.parse_args()

    if args.reset and os.path.exists("../data/lancedb"):
        import shutil

        shutil.rmtree("../data/lancedb")
        print("Cleared existing LanceDB tables")

    results = run_evaluation()