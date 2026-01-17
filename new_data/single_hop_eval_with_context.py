from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from app import tkg  # noqa: E402
from chatbot import TkgDependencies, tkg_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-hop evaluation over a CSV of questions with context extraction for RAGAS."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default=str(SCRIPT_DIR / "output_with_context_v1.csv"),
        help="Input CSV with columns question and ground truth (or gold_answer).",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "output_with_context_v1.csv"),
        help="Output CSV path (default: output_with_context_v1.csv).",
    )
    return parser.parse_args()


def resolve_ground_truth_field(fieldnames: Iterable[str]) -> str:
    candidates = ("ground truth", "ground_truth", "gold_answer", "gold")
    for name in candidates:
        if name in fieldnames:
            return name
    raise ValueError(
        "Input CSV must include a ground truth column "
        "(ground truth, ground_truth, or gold_answer)."
    )


def normalize_header(name: str) -> str:
    cleaned = name.replace("\ufeff", "").strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def message_to_text(msg) -> str:
    """Extract text content from a message object."""
    if hasattr(msg, "parts"):
        return "".join(str(part) for part in msg.parts)
    if hasattr(msg, "content"):
        return str(msg.content)
    if hasattr(msg, "text"):
        return str(msg.text)
    if hasattr(msg, "value"):
        return str(msg.value)
    return ""


def safe_messages(fetch) -> list:
    """Safely fetch messages from a result object."""
    try:
        return fetch() or []
    except AttributeError:
        return []


def extract_response_text(result) -> str:
    """Extract the final response text from the agent result."""
    response_text = getattr(result, "output", "") or ""
    if not response_text:
        for msg in reversed(safe_messages(getattr(result, "new_messages", lambda: []))):
            candidate = message_to_text(msg)
            if candidate:
                response_text = candidate
                break
    if not response_text:
        for msg in reversed(safe_messages(getattr(result, "all_messages", lambda: []))):
            if getattr(msg, "role", None) in ("assistant", "assistant_message", None):
                candidate = message_to_text(msg)
                if candidate:
                    response_text = candidate
                    break
    return response_text or "No response generated."


def extract_context_from_messages(result) -> list[str]:
    """
    Extract retrieved context from tool call responses in the agent result.
    Returns a list of context strings (facts) retrieved from the TKG search.
    """
    contexts = []
    
    try:
        # Get all messages from the result
        all_msgs = safe_messages(getattr(result, "all_messages", lambda: []))
        
        for msg in all_msgs:
            # Check if this is a tool return message (contains search results)
            # Pydantic AI tool returns are typically stored in parts or content
            msg_role = getattr(msg, "role", None)
            
            # Look for tool return messages or messages with parts containing tool results
            if msg_role == "tool":
                # Try to extract content from the message
                content = None
                if hasattr(msg, "content"):
                    content = msg.content
                elif hasattr(msg, "parts"):
                    # Parts might contain the tool return
                    for part in msg.parts:
                        if hasattr(part, "content"):
                            content = part.content
                            break
                        elif isinstance(part, (dict, str)):
                            content = part
                            break
                
                # Parse the content to extract facts
                if content:
                    # Content might be a JSON string or a list of dicts
                    try:
                        if isinstance(content, str):
                            # Try to parse as JSON
                            parsed = json.loads(content)
                        else:
                            parsed = content
                        
                        # If it's a list of results (from search_tkg)
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and "fact" in item:
                                    contexts.append(item["fact"])
                                elif hasattr(item, "fact"):
                                    contexts.append(item.fact)
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        # If parsing fails, try to extract as string
                        content_str = str(content)
                        if content_str and content_str != "None":
                            contexts.append(content_str)
            
            # Also check for parts that might contain tool results in assistant messages
            elif hasattr(msg, "parts"):
                for part in msg.parts:
                    # Check if part has tool_return or similar attributes
                    if hasattr(part, "part_kind") and "tool" in str(getattr(part, "part_kind", "")).lower():
                        if hasattr(part, "content"):
                            content = part.content
                            try:
                                if isinstance(content, str):
                                    parsed = json.loads(content)
                                else:
                                    parsed = content
                                
                                if isinstance(parsed, list):
                                    for item in parsed:
                                        if isinstance(item, dict) and "fact" in item:
                                            contexts.append(item["fact"])
                                        elif hasattr(item, "fact"):
                                            contexts.append(item.fact)
                            except (json.JSONDecodeError, TypeError, AttributeError):
                                pass
    
    except Exception as e:
        print(f"Warning: Error extracting context: {e}")
    
    return contexts


def format_context(contexts: list[str]) -> str:
    """Format the context list into a single string for CSV output."""
    if not contexts:
        return "No context retrieved"
    
    # Join contexts with a separator that's easy to parse later
    # Using double newline as separator
    return "\n\n".join(contexts)


async def answer_question_with_context(deps: TkgDependencies, question: str) -> tuple[str, str]:
    """
    Answer a question and return both the answer and the retrieved context.
    
    Returns:
        tuple: (answer, context) where context is formatted as a string
    """
    result = await tkg_agent.run(question, message_history=[], deps=deps)
    
    # Extract the answer
    answer = extract_response_text(result)
    
    # Extract the context
    contexts = extract_context_from_messages(result)
    context = format_context(contexts)
    
    return answer, context


async def run_eval(input_csv: str, output_csv: str) -> None:
    """Run evaluation with context extraction for RAGAS metrics."""
    load_dotenv(ROOT_DIR / ".env")

    neo4j_uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")

    tkg_client = tkg(neo4j_uri, neo4j_user, neo4j_password)
    deps = TkgDependencies(tkg_client=tkg_client)

    try:
        with open(input_csv, "r", encoding="utf-8-sig", newline="") as infile:
            reader = csv.DictReader(infile)
            if not reader.fieldnames:
                raise ValueError("Input CSV must include a header row.")

            field_map = {
                normalize_header(name): name
                for name in reader.fieldnames
                if name is not None
            }
            question_field = field_map.get("question")
            if not question_field:
                available = ", ".join(reader.fieldnames)
                raise ValueError(
                    "Input CSV must include a 'question' column. "
                    f"Found: {available}"
                )

            ground_truth_field = resolve_ground_truth_field(field_map.keys())
            ground_truth_field = field_map[ground_truth_field]
            rows = list(reader)
            total = len(rows)
            print(f"Loaded {total} questions from {input_csv}.")

            with open(output_csv, "w", encoding="utf-8-sig", newline="") as outfile:
                writer = csv.DictWriter(
                    outfile, fieldnames=["question", "ground_truth", "context", "answer"]
                )
                writer.writeheader()

                for index, row in enumerate(rows, start=1):
                    question = (row.get(question_field) or "").strip()
                    ground_truth = (row.get(ground_truth_field) or "").strip()
                    if not question:
                        continue

                    preview = question if len(question) <= 80 else question[:77] + "..."
                    print(f"[{index}/{total}] {preview}")

                    try:
                        answer, context = await answer_question_with_context(deps, question)
                    except Exception as exc:
                        answer = f"Error: {exc}"
                        context = "Error retrieving context"

                    writer.writerow(
                        {
                            "question": question,
                            "ground_truth": ground_truth,
                            "context": context,
                            "answer": answer,
                        }
                    )

                print(f"\nEvaluation complete!")
                print(f"Wrote results to {output_csv}.")
                print(f"Output includes: question, ground_truth, context, answer")
                print(f"This format is compatible with RAGAS metrics evaluation.")
    finally:
        await tkg_client.close()


def main() -> None:
    args = parse_args()
    asyncio.run(run_eval(args.input_csv, args.output))


if __name__ == "__main__":
    main()
