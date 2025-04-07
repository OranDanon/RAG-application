# --------------------------------------------------------------
# Create queries for evaluation from JSON files
# --------------------------------------------------------------
import glob
import json
import os

from matplotlib import pyplot as plt
from tqdm import tqdm
import lancedb

from utils.my_timer import make_timer


# --------------------------------------------------------------
# Define embedding models
# --------------------------------------------------------------

embedding_models = {
    "openai-small": {
        "name": "text-embedding-3-small",
        "dimensions": 8191,
        "type": "openai",
        "description": "OpenAI's smaller embedding model with 1536 dimensions. Good balance of performance and cost."
    },
    "sentence-mpnet": {
        "name": "all-mpnet-base-v2",
        "dimensions": 768,
        "type": "sentence-transformers",
        "description": "MPNet-based model, strong general-purpose embeddings with good performance for semantic search."
    },
    "sentence-minilm": {
        "name": "all-MiniLM-L6-v2",
        "dimensions": 384,
        "type": "sentence-transformers",
        "description": "Lightweight model with 384 dimensions. Fastest option with decent performance."
    },
    "bge-small": {
        "name": "BAAI/bge-small-en-v1.5",
        "dimensions": 384,
        "type": "sentence-transformers",
        "description": "Another lightweight and efficient sentence transformer model."
    },
}


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
            sources = [source for source in top_k_results['source']]
            if expected in sources:
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
@make_timer
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
        db_path = f"data/lancedb/"
        db = lancedb.connect(db_path)
        table = db.open_table(f"lang_graph_{model_info['name'].replace('/', '_')}")

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

    if args.reset and os.path.exists("data/lancedb"):
        import shutil

        shutil.rmtree("data/lancedb")
        print("Cleared existing LanceDB tables")

    results = run_evaluation()