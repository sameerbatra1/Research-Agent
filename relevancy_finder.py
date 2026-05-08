"""
Agent 3: Relevancy Finder

Given the user's research requirements and a CSV of paper abstracts,
decide whether each paper is relevant to the user's goals.

Inputs:
- Requirements text (read from requirements_summary.txt if present, otherwise prompted)
- Abstracts CSV (defaults to extracted_abstracts.csv; can be adjusted below)

Outputs:
- CSV with columns: title, abstract, relevant (bool), score (0-5), reason
"""

from __future__ import annotations
import os
import json
from typing import List, Dict, Optional
import re
import csv

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()

AZURE_ENDPOINT = os.getenv("Azure_OpenAI_Endpoint")
AZURE_API_KEY = os.getenv("Azure_OpenAI_Key")
AZURE_API_VERSION = "2024-10-21"
DEPLOYMENT_NAME = "gpt-4o-mini"

if not AZURE_ENDPOINT or not AZURE_API_KEY:
    raise RuntimeError("Azure OpenAI endpoint/key not set. Define AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT, 
    api_key=AZURE_API_KEY, 
    api_version=AZURE_API_VERSION
)


def azure_complete(prompt: str, max_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    resp = client.chat.completions.create(model=DEPLOYMENT_NAME, messages=messages, max_tokens=max_tokens)
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


def load_abstracts_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Abstracts CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize expected columns
    expected_cols = {"title", "abstract"}
    lower_map = {c.lower(): c for c in df.columns}
    if not expected_cols.issubset(set(lower_map.keys())):
        raise ValueError(
            f"CSV must contain columns 'title' and 'abstract'. Found: {list(df.columns)}"
        )
    # If actual columns are different cased, rename to standard
    rename_map = {}
    for std in expected_cols:
        actual = lower_map.get(std)
        if actual and actual != std:
            rename_map[actual] = std
    if rename_map:
        df = df.rename(columns=rename_map)
    return df[["title", "abstract"]]


def build_relevancy_prompt(requirements: str, paper_title: str, paper_abstract: str) -> str:
    return (
        "You are an expert research assistant. Evaluate whether the given paper aligns with the user's research goals.\n"
        "Return STRICT JSON only (no markdown, no extra text) with this schema: "
        "{\"relevant\": boolean, \"score\": integer (0-5), \"reason\": string}.\n\n"
        "User Requirements:\n"
        "------------------\n"
        f"{requirements}\n\n"
        "Paper:\n"
        "------\n"
        f"Title: {paper_title}\n"
        f"Abstract: {paper_abstract}\n\n"
        "Criteria:\n"
        "- Topical alignment with the user's goals\n"
        "- Methodological fit (e.g., tasks, datasets, evaluation methods)\n"
        "- Desired outputs/insights match\n"
        "- Recency/coverage as implied by requirements\n"
        "Important: Respond with STRICT JSON only (no code fences). Keep reason concise and to the point. "
        "Score scale: 0 = not relevant at all, 5 = highly relevant."
    )


def _strip_code_fences(possible_json: str) -> str:
    text = possible_json.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", text).strip()
    return text


def _extract_json_object(possible_json: str) -> Optional[str]:
    # Try to find a JSON object within text
    match = re.search(r"\{[\s\S]*\}", possible_json)
    if match:
        return match.group(0)
    return None


def parse_relevancy_response(raw_text: str) -> Dict[str, object]:
    text = raw_text.strip()
    cleaned = _strip_code_fences(text)
    candidate = _extract_json_object(cleaned) or cleaned
    # Attempt strict JSON parse first
    try:
        data = json.loads(candidate)
        relevant = bool(data.get("relevant", False))
        # parse score and clamp to 0-5
        score_raw = data.get("score", 0)
        try:
            score = int(score_raw)
        except Exception:
            score = 0
        if score < 0:
            score = 0
        if score > 5:
            score = 5
        reason = str(data.get("reason", "")).strip()
        return {"relevant": relevant, "score": score, "reason": reason}
    except Exception:
        # Fallback: keep full text as reason (no hard truncation) and infer relevance via regex
        is_true = bool(re.search(r'"?relevant"?\s*:\s*true', cleaned, flags=re.IGNORECASE))
        m = re.search(r'"?score"?\s*:\s*(\d+)', cleaned)
        score = int(m.group(1)) if m and m.group(1).isdigit() else (5 if is_true else 0)
        if score < 0:
            score = 0
        if score > 5:
            score = 5
        return {"relevant": is_true, "score": score, "reason": cleaned}


def evaluate_relevancy(
    requirements_text: str,
    abstracts_df: pd.DataFrame,
) -> pd.DataFrame:
    results: List[Dict[str, object]] = []
    total = len(abstracts_df)
    print(f"Found {total} paper(s) to evaluate.")
    for idx, (_, row) in enumerate(abstracts_df.iterrows(), start=1):
        title_for_log = str(row.get("title", "")).strip()
        print(f"[{idx}/{total}] Evaluating: {title_for_log}")
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        if not abstract:
            results.append({
                "title": title,
                "abstract": abstract,
                "relevant": False,
                "score": 0,
                "reason": "No abstract available",
            })
            continue

        try:
            prompt = build_relevancy_prompt(requirements_text, title, abstract)
            raw = azure_complete(prompt, max_tokens=512)
            parsed = parse_relevancy_response(raw)
        except Exception as error:
            parsed = {"relevant": False, "score": 0, "reason": f"Error: {error}"}

        results.append({
            "title": title,
            "abstract": abstract,
            "relevant": bool(parsed.get("relevant", False)),
            "score": int(parsed.get("score", 0)),
            "reason": str(parsed.get("reason", "")),
        })

    print(f"Completed evaluating {total} paper(s).")
    return pd.DataFrame(results, columns=["title", "abstract", "relevant", "score", "reason"])


def try_default_csv_path() -> Optional[str]:
    candidates = [
        "extracted_abstracts.csv",
        "extracted abstracts.csv",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


if __name__ == "__main__":
    # Load requirements
    requirements_path = r"F:\Python\SAIM\Research Assistant\requirements_summary.txt"
    if os.path.isfile(requirements_path):
        with open(requirements_path, "r", encoding="utf-8", errors="ignore") as f:
            requirements = f.read().strip()
    else:
        requirements = input("Enter a short summary of your research goals: ").strip()

    if not requirements:
        print("Error: requirements text is empty.")
        raise SystemExit(1)

    # Load abstracts CSV
    csv_path = try_default_csv_path()
    if not csv_path:
        csv_path = r"F:\Python\SAIM\Research Assistant\extracted_abstracts.csv"

    try:
        abstracts = load_abstracts_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise SystemExit(1)

    # Evaluate
    df_results = evaluate_relevancy(requirements, abstracts)
    df_results.to_csv("relevancy_results.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    total = len(df_results)
    num_relevant = int(df_results["relevant"].sum())
    print(f"Evaluated {total} papers. Relevant: {num_relevant}. Saved to relevancy_results.csv")