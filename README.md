# Knowledge Graph-based Retrieval Augmented Generation

A collection of implementations for Knowledge Graph-based RAG  approaches and baseline methods for comparison.

## Overview

The repository implements several RAG approaches:

1. **Baseline approaches**:
   - **Standard RAG**: Traditional retrieval-based approach using vector similarity
   - **Chain-of-Thought RAG**: Enhanced retrieval with explicit reasoning steps

2. **KG-RAG approaches**:
   - **Entity-based approach**: Uses embedding-based entity matching and beam search to find relevant information in the knowledge graph
   - **Cypher-based approach**: Uses Cypher queries to retrieve information from a Neo4j graph database
   - **GraphRAG-based approach**: Implements a community detection and hierarchical search strategy

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) for dependency.

```bash
# Clone the repository
git clone https://github.com/yourusername/kg-rag.git
cd kg-rag

# Install uv if you don't have it
curl -sSf https://astral.sh/uv/install.sh | bash

uv sync
source .venv/bin/activate
```

For development, you can install the dev dependencies:

```bash
uv sync --dev
source .venv/bin/activate
```


## Environment Variables

Export the following environment variables:

```
OPENAI_API_KEY=your_openai_api_key
```

For the Cypher-based approach, also add:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Data

The latest evaluation dataset lives under `new_data/`:

- `new_data/researcher/`: JSON source documents used to build entity and GraphRAG artifacts
- `new_data/questions_main.csv`: question/answer pairs for evaluation

Artifacts are written under `data/researcher/`, and evaluation outputs go under
`evaluation_results/questions_main` (or a separate directory if you want to
separate GraphRAG runs).

## Usage

### Questions_main dataset (new_data)

Build entity artifacts + document chunks (JSON):

```bash
python scripts/build_entity_graph.py \
    --docs-dir new_data/researcher \
    --output-dir data/researcher \
    --file-extensions .json
```

Build GraphRAG artifacts:

```bash
python scripts/build_graphrag.py \
    --docs-dir new_data/researcher \
    --output-dir data/researcher \
    --file-extensions .json
```

Run evaluations (exact match is already set in `kg_rag/configs/researcher-kgrag.json`):

```bash
python kg_rag/evaluation/run_evaluation.py \
    --data-path new_data/questions_main.csv \
    --config-path kg_rag/configs/researcher-kgrag.json \
    --method entity \
    --output-dir evaluation_results/questions_main
```

```bash
python kg_rag/evaluation/run_evaluation.py \
    --data-path new_data/questions_main.csv \
    --config-path kg_rag/configs/researcher-kgrag.json \
    --method graphrag \
    --output-dir evaluation_results/questions_main_graph
```

If you also want cypher-based results from Neo4j:

```bash
python kg_rag/evaluation/run_evaluation.py \
    --data-path new_data/questions_main.csv \
    --config-path kg_rag/configs/researcher-kgrag.json \
    --method cypher \
    --output-dir evaluation_results/questions_main
```

### Running Scripts (new_data)

### 1. Building Entity Graph Artifacts

Build the entity graph artifacts used by the entity-based KG-RAG methods:

```bash
python scripts/build_entity_graph.py \
    --docs-dir new_data/researcher \
    --output-dir data/researcher \
    --file-extensions .json
```

### 2. Building GraphRAG Artifacts

Build GraphRAG artifacts and vector stores:

```bash
python scripts/build_graphrag.py \
    --docs-dir new_data/researcher \
    --output-dir data/researcher \
    --file-extensions .json
```

### 3. Running Interactive Query Mode (Entity KG-RAG)

```bash
python scripts/run_entity_rag.py \
    --graph-documents-pkl-path data/researcher/entity_graph_documents.pkl \
    --documents-pkl-path data/researcher/documents.pkl \
    --documents-path new_data/researcher
```

### 4. Running Evaluation

```bash
python kg_rag/evaluation/run_evaluation.py \
    --data-path new_data/questions_main.csv \
    --config-path kg_rag/configs/researcher-kgrag.json \
    --method entity \
    --output-dir evaluation_results/questions_main
```

```bash
python kg_rag/evaluation/run_evaluation.py \
    --data-path new_data/questions_main.csv \
    --config-path kg_rag/configs/researcher-kgrag.json \
    --method graphrag \
    --output-dir evaluation_results/questions_main_graph
```

```bash
python kg_rag/evaluation/run_evaluation.py \
    --data-path new_data/questions_main.csv \
    --config-path kg_rag/configs/researcher-kgrag.json \
    --method cypher \
    --output-dir evaluation_results/questions_main
```

### 5. Running Hyperparameter Search

```bash
python kg_rag/evaluation/hyperparameter_search.py \
    --data-path new_data/questions_main.csv \
    --graph-path data/researcher/entity_graph.pkl \
    --method entity \
    --configs-path kg_rag/evaluation/hyperparameter_configs.json \
    --output-dir hyperparameter_search \
    --max-samples 10 \
    --verbose
```

```bash
python kg_rag/evaluation/hyperparameter_search.py \
    --data-path new_data/questions_main.csv \
    --method graphrag \
    --graphrag-artifacts data/researcher/graphrag_artifacts.pkl \
    --vector-store-dir data/researcher/vector_stores \
    --configs-path kg_rag/evaluation/hyperparameter_configs.json \
    --output-dir hyperparameter_search \
    --max-samples 10 \
    --verbose
```

## Development

### Pre-commit hooks

This project uses pre-commit hooks to ensure code quality:

```bash
# Run pre-commit hooks on all files
pre-commit run --all-files
```

### Running tests

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=kg_rag tests/
```
