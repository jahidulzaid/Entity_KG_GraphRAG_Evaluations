#!/usr/bin/env python
"""Script to run evaluation across different KG-RAG methods."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_rag.methods.baseline_rag.kg_rag import BaselineRAG
from kg_rag.methods.cypher_based.kg_rag import CypherBasedKGRAG

# Import RAG systems
from kg_rag.methods.entity_based.kg_rag import EntityBasedKGRAG
from kg_rag.methods.graphrag_based.kg_rag import (
    create_graphrag_system,
)
from kg_rag.utils.document_loader import load_documents, load_graph_documents
from kg_rag.utils.evaluator import Evaluator
from kg_rag.utils.graph_utils import create_graph_from_graph_documents


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate KG-RAG methods")
    parser.add_argument(
        "--data-path", type=str, required=True, help="Path to evaluation dataset CSV"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["all", "entity", "cypher", "graphrag", "baseline"],
        default="all",
        help="KG-RAG method to evaluate",
    )
    parser.add_argument(
        "--config-path", type=str, default=None, help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--use-cot", action="store_true", help="Use Chain-of-Thought prompting"
    )
    parser.add_argument(
        "--numerical-answer",
        action="store_true",
        help="Format answer as numerical value only",
    )
    parser.add_argument(
        "--normalize-answers",
        action="store_true",
        help="Normalize answers as numbers before comparison",
    )
    parser.add_argument(
        "--exact-match",
        action="store_true",
        help="Use exact string matching for answers",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--question-col",
        type=str,
        default="New Question",
        help="Column name for questions in the dataset",
    )
    parser.add_argument(
        "--answer-col",
        type=str,
        default="New Answer",
        help="Column name for answers in the dataset",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    # New argument for question index
    parser.add_argument(
        "--question-index",
        type=int,
        default=None,
        help="Index of a specific question to evaluate (0-based)",
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from a JSON file."""
    if not config_path:
        return {}

    with open(config_path) as f:
        return json.load(f)


def create_entity_rag(config, use_cot=False, numerical_answer=False, verbose=False):
    """Create an entity-based KG-RAG system."""
    llm = ChatOpenAI(temperature=0, model_name=config.get("model_name", "gpt-4o"))

    # Load the documents
    print(f"Loading documents from {config.get('documents_pkl_path')}")
    documents = load_documents(
        directory_path=config.get("documents_path"),
        pickle_path=config.get("documents_pkl_path"),
    )

    # Load the graph
    print(
        f"Converting graph documents from {config.get('graph_documents_pkl_path')} to graph..."
    )
    graph_documents = load_graph_documents(config.get("graph_documents_pkl_path"))

    graph = create_graph_from_graph_documents(graph_documents)

    return EntityBasedKGRAG(
        graph=graph,
        graph_documents=graph_documents,
        document_chunks=documents,
        llm=llm,
        top_k_nodes=config.get("top_k_nodes", 10),
        top_k_chunks=config.get("top_k_chunks", 5),
        max_hops=config.get("max_hops", 1),
        similarity_threshold=config.get("similarity_threshold", 0.7),
        node_freq_weight=config.get("node_freq_weight", 0.4),
        node_sim_weight=config.get("node_sim_weight", 0.6),
        use_cot=use_cot,
        numerical_answer=numerical_answer,
        verbose=verbose,
    )


def create_baseline_rag(config, use_cot=False, numerical_answer=False, verbose=False):
    """Create a standard baseline RAG system."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    persist_dir = (
        config.get("persist_dir")
        or config.get("chroma_persist_dir")
        or "chroma_db"
    )

    return BaselineRAG(
        collection_name=config.get("collection_name", "sec_10q"),
        chroma_persist_dir=persist_dir,
        model_name=config.get("model_name", "gpt-4o"),
        embedding_model=config.get("embedding_model", "text-embedding-3-small"),
        top_k=config.get("top_k", 5),
        use_cot=use_cot,
        numerical_answer=numerical_answer,
        verbose=verbose,
    )


def create_cypher_rag(config, use_cot=False, numerical_answer=False, verbose=False):
    """Create a Cypher-based KG-RAG system."""
    from langchain_neo4j import Neo4jGraph

    # Create Neo4j connection
    neo4j_uri = (
        config.get("neo4j_uri")
        or os.getenv("NEO4J_URI")
        or "bolt://localhost:7687"
    )
    neo4j_user = (
        config.get("neo4j_user")
        or os.getenv("NEO4J_USER")
        or os.getenv("NEO4J_USERNAME")
        or "neo4j"
    )
    neo4j_password = (
        config.get("neo4j_password")
        or os.getenv("NEO4J_PASSWORD")
        or "password"
    )
    neo4j_database = config.get("neo4j_database") or os.getenv("NEO4J_DATABASE")

    import inspect

    graph_kwargs = {
        "url": neo4j_uri,
        "username": neo4j_user,
        "password": neo4j_password,
    }
    if (
        neo4j_database
        and "database" in inspect.signature(Neo4jGraph.__init__).parameters
    ):
        graph_kwargs["database"] = neo4j_database

    neo4j_graph = Neo4jGraph(**graph_kwargs)

    # Create LLM
    llm = ChatOpenAI(temperature=0, model_name=config.get("model_name", "gpt-4o"))

    return CypherBasedKGRAG(
        graph=neo4j_graph,
        llm=llm,
        max_depth=config.get("max_depth", 2),
        max_hops=config.get("max_hops", 3),
        use_cot=use_cot,
        numerical_answer=numerical_answer,
        verbose=verbose,
    )


def create_graphrag_rag(config, use_cot=False, numerical_answer=False, verbose=False):
    """Create a GraphRAG-based KG-RAG system."""
    # Get configuration
    artifacts_path = config.get("artifacts_path")
    if not artifacts_path:
        raise ValueError("artifacts_path is required for GraphRAG-based method")

    vector_store_dir = config.get("vector_store_dir", "vector_stores")
    llm_model = config.get("model_name", "gpt-4o")
    search_strategy = config.get("search_strategy", "local")
    community_level = config.get("community_level", 2)

    return create_graphrag_system(
        artifacts_path=artifacts_path,
        vector_store_dir=vector_store_dir,
        llm_model=llm_model,
        search_strategy=search_strategy,
        community_level=community_level,
        use_cot=use_cot,
        numerical_answer=numerical_answer,
        verbose=verbose,
    )


def check_question_index(index, df):
    """Check if the question index is valid."""
    if index < 0 or index >= len(df):
        raise ValueError(
            f"Question index {index} is out of bounds. Dataset has {len(df)} entries."
        )
    df = df.iloc[[index]].copy()
    print(f"Evaluating only question at index {index}")
    return df


def check_method(method):
    """Check if the method is valid and return a list of methods."""
    if method == "all":
        methods = ["entity", "baseline", "cypher", "graphrag"]
    else:
        methods = [method]
    return methods


def normalize_column_name(name: str) -> str:
    """Normalize column names for case-insensitive matching."""
    return str(name).replace("\ufeff", "").strip().lower()


def resolve_columns(
    df: pd.DataFrame, question_col: str, answer_col: str
) -> tuple[str, str]:
    """Resolve column names, falling back to common alternatives."""
    column_map = {normalize_column_name(col): col for col in df.columns}

    def pick_column(primary: str, candidates: list[str]) -> str | None:
        if primary in df.columns:
            return primary
        primary_key = normalize_column_name(primary)
        if primary_key in column_map:
            return column_map[primary_key]
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
            candidate_key = normalize_column_name(candidate)
            if candidate_key in column_map:
                return column_map[candidate_key]
        return None

    resolved_question = pick_column(
        question_col,
        ["question", "query", "prompt", "new question"],
    )
    resolved_answer = pick_column(
        answer_col,
        ["ground_truth", "ground truth", "answer", "gold_answer", "gold", "new answer"],
    )

    if not resolved_question or not resolved_answer:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            "Could not find required columns in dataset. "
            f"Available columns: {available}"
        )

    return resolved_question, resolved_answer


def main():
    """Run evaluation across different KG-RAG methods."""
    # Load environment variables
    load_dotenv()

    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(args.config_path) or {}

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    df = pd.read_csv(args.data_path)

    if args.question_index is not None:
        df = check_question_index(args.question_index, df)

    question_col, answer_col = resolve_columns(
        df, args.question_col, args.answer_col
    )

    # Determine which methods to evaluate
    methods = []
    methods = check_method(args.method)

    # Filter out methods missing required components
    if "graphrag" in methods and not config.get("artifacts_path"):
        print("Warning: GraphRAG-based method requires artifacts_path in config")
        methods.remove("graphrag")

    # Suffix for CoT evaluation
    cot_suffix = "-cot" if args.use_cot else ""

    normalize_answers = config.get("normalize_answers", True)
    if args.normalize_answers and args.exact_match:
        raise ValueError("Use only one of --normalize-answers or --exact-match")
    if args.normalize_answers:
        normalize_answers = True
    if args.exact_match:
        normalize_answers = False

    results = {}

    # Evaluate each method
    for method in methods:
        print(f"\nEvaluating {method}-based KG-RAG{cot_suffix}...")
        try:
            # Create RAG system
            if method == "entity":
                rag_system = create_entity_rag(
                    config, args.use_cot, args.numerical_answer, args.verbose
                )
            elif method == "baseline":
                rag_system = create_baseline_rag(
                    config, args.use_cot, args.numerical_answer, args.verbose
                )
            elif method == "cypher":
                rag_system = create_cypher_rag(
                    config, args.use_cot, args.numerical_answer, args.verbose
                )
            elif method == "graphrag":
                rag_system = create_graphrag_rag(
                    config, args.use_cot, args.numerical_answer, args.verbose
                )

            # Create evaluator
            method_name = f"{method}{cot_suffix}"
            evaluator = Evaluator(
                rag_system=rag_system,
                config=config,
                output_dir=output_dir,
                experiment_name=f"{method_name}_rag",
                verbose=args.verbose,
            )

            # Run evaluation
            method_results = evaluator.evaluate(
                data_path=df,
                question_col=question_col,
                answer_col=answer_col,
                max_samples=args.max_samples,
                normalize_answers=normalize_answers,
            )

            results[method_name] = method_results

        except Exception as e:
            print(f"Error evaluating {method}-based KG-RAG: {str(e)}")

    # Print summary
    print("\nEvaluation Results Summary:")
    for method_name in results:
        print(f"{method_name} KG-RAG: {results[method_name]['accuracy']:.2%} accuracy")

    print(f"\nDetailed results saved to {output_dir}")

    # If specific question, print more details
    if args.question_index is not None:
        print("\nDetailed results for the specified question:")
        for method_name, method_results in results.items():
            print(f"\n{method_name} KG-RAG:")
            if (
                "predictions" in method_results
                and len(method_results["predictions"]) > 0
            ):
                prediction = method_results["predictions"][0]
                print(f"Question: {prediction.get('question', 'N/A')}")
                print(f"Reference Answer: {prediction.get('reference', 'N/A')}")
                print(f"Predicted Answer: {prediction.get('prediction', 'N/A')}")
                print(f"Correct: {prediction.get('is_correct', False)}")


if __name__ == "__main__":
    main()
