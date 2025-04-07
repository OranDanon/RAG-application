# LangGraph Information Retrieval System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about LangGraph by retrieving and synthesizing information from multiple sources.

## Overview

This project creates a conversational AI assistant capable of answering technical questions about LangGraph by:

1. Processing and extracting text from multiple document sources
2. Chunking documents using Docling's advanced document understanding capabilities
3. Creating and storing embeddings in a vector database (LanceDB)
4. Retrieving relevant information based on user queries
5. Generating comprehensive answers using retrieved context


![running example.png](running%20example.png)

## Features

- **Advanced Document Processing**: Leverages Docling for intelligent document understanding and chunking
- **Multiple Embedding Model Support**: Compares and evaluates different embedding models
- **Optimized Retrieval Parameters**: Analysis to determine the optimal k value for each embedding model
- **Interactive Chat Interface**: Clean Streamlit UI for conversational interaction

## Data Sources

The system integrates information from:
- GitHub Repository: LangGraph GitHub Repository
- Technical blogs and articles about LangGraph from Galileo, LinkedIn, Medium, and Towards Data Science
- Technical documentation and tutorials

## Architecture

### 1. Document Processing Pipeline

```
Document Sources → Text Extraction → Hybrid Chunking → Embeddings → Vector Storage
```

- **Text Extraction**: Docling converts various document formats (PDF, HTML, Markdown) into a unified format
- **Hybrid Chunking**: Smart chunking that preserves document structure and semantic coherence
- **Embedding Generation**: Multiple embedding models evaluated for optimal performance

### 2. Retrieval System

```
User Query → Query Embedding → Similarity Search → Context Retrieval → Answer Generation
```

- **Vector Search**: Fast similarity search through LanceDB
- **Context Aggregation**: Combines multiple relevant chunks with source metadata
- **Answer Generation**: OpenAI model synthesizes answers using retrieved context

## Getting Started

### Prerequisites

1. Python 3.8+
2. OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/langgraph-rag.git
cd langgraph-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Usage

Run each script in sequence to build and use the system:

1. **Create Embeddings**:
```bash
python open_ai_embed.py   # Create and store embeddings
```

2. **Launch Chat Interface**:
```bash
streamlit run chatbot.py  # Start the Streamlit chat application
```

Then open your browser and navigate to `http://localhost:8501`.

## Evaluation
### Creating synthetic dataset
Using Claude 3.7 Sonnet (Thinks) and an `.md` files I wrote the following prompt to generate the synthetic dataset:

Please help me create a comprehensive Q&A dataset to evaluate a Document-based Conversational AI System for LangGraph Information Retrieval (RAG ChatBot). 
Generate 50 questions per batch (to maintain quality and avoid message size limitations) that follow this specific JSON structure: 
```json
{
  "questions": [
    {
      "id": "unique-question-identifier",
      "section": "Specific document section or topic",
      "question": "User query that requires information from the documents",
      "context": "Exact text excerpt from the document containing the answer",
      "difficulty": "easy/medium/hard",
      "answer_type": "factoid/descriptive/procedural",
      "question_type": "technical/conceptual/comparative"
    }
  ]
}
```
Distribution requirements: 
- Difficulty: 30% easy, 40% medium, 30% hard 
- Answer types: 40% factoid (short, direct answers), 40% descriptive (explanations), 20% procedural (how-to)
- Question types: 50% technical (implementation), 30% conceptual (understanding), 20% comparative (when applicable) 

Question characteristics: 
- Ensure questions span all major LangGraph topics proportionally to their coverage in the documents 
- Make questions increasingly complex across difficulty levels: 
  * Easy: Direct information retrieval from a single paragraph 
  * Medium: Synthesizing information from multiple paragraphs in the same section 
  * Hard: Requiring deeper understanding, inference, or connecting concepts 
- Include questions that test: 
  * Core LangGraph concepts and terminology 
  * Implementation details and code patterns 
  * Best practices and application scenarios 
  * Differences from other frameworks (where documented) 

For each question: 
1. Extract the precise document text in the "context" field that contains the answer 
2. Ensure the question is answerable solely from that context 
3. Vary question formulations (what, how, why, compare, explain, etc.)
4. Assign a unique ID that indicates the topic area and difficulty I'll ask you to generate 2 batches of 50 questions each to reach 100 total questions. For each batch, please focus on different sections.

### Embedding Models

We evaluated several embedding models to find the optimal configuration:

- OpenAI Embeddings (`text-embedding-3-large`)
- Sentence Transformers
- Open Source Alternatives

Our analysis (shown in the graph) demonstrates that `text-embedding-3-large` with a combined cross-encoder re-ranking strategy achieves the best performance across all recall@k metrics.

Key findings:
- OpenAI embeddings consistently outperform open-source alternatives
- Optimal k-value for retrieval is 1-50 depending on the model
- Adding a cross-encoder for re-ranking significantly improves precision
![recall@k.png](recall%40k.png)

| Model                       | 1     | 3     | 5     | 10    | 25    | 50    |
|-----------------------------|-------|-------|-------|-------|-------|-------|
| text-embedding-3-small      | 0.703 | 0.836 | 0.902 | 0.956 | 0.985 | 1.000 |
| all-mpnet-base-v2         | 0.636 | 0.803 | 0.864 | 0.927 | 0.990 | 1.000 |
| all-MiniLM-L6-v2          | 0.683 | 0.841 | 0.886 | 0.932 | 0.985 | 1.000 |
| BAAI/bge-small-en-v1.5     | 0.725 | 0.851 | 0.883 | 0.931 | 0.976 | 1.000 |

```text
BEST MODEL FOR EACH RECALL@K:
Recall@1: BAAI/bge-small-en-v1.5 (0.7254)
Recall@3: BAAI/bge-small-en-v1.5 (0.8508)
Recall@5: text-embedding-3-small (0.9017)
Recall@10: text-embedding-3-small (0.9559)
Recall@25: all-mpnet-base-v2 (0.9898)
Recall@50: text-embedding-3-small (1.0000)
```

## Document Chunking Strategy

The system uses Docling's HybridChunker which:

1. Preserves document structure (headings, paragraphs, tables)
2. Maintains semantic coherence within chunks
3. Optimizes chunk size for the specific embedding model
4. Retains metadata and hierarchical relationships

## Future Improvements
- **Figure Interpretation**: Use LLMs to translate figures and charts into textual descriptions
- Implement cross-encoder re-ranking for improved retrieval precision
  - Implement contextual_rag
  - Graph RAG, Light RAG, Path RAG (for more advanced relations: cross documents Q&A)
- Enhance evaluation with the 100 Q&A dataset to measure system performance: E.g. use rubrics to test generation

## Evaluation

The system was evaluated using a custom dataset of 100 questions covering:
- Different difficulty levels (easy, medium, hard)
- Various answer types (factoid, descriptive, procedural)
- Different question types (technical, conceptual, comparative)

Performance metrics include:
- Relevance of retrieved contexts
- Answer accuracy and comprehensiveness
- Response time and efficiency

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Docling for document processing capabilities
- LanceDB for vector storage
- OpenAI for embedding and language models
