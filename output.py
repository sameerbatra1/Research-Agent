"""
Agent 4: Structured extraction for relevant papers

Steps:
- Read relevant papers from relevancy_results.csv
- Locate corresponding PDF files in the papers folder
- Extract full text from each PDF
- Call Gemini with a strict JSON schema prompt to extract fields
- Save a CSV with: title, abstract, Authors, Objective, Methodology, Algorithms Used,
  Dataset Used, Example of Working Methodology, Results, Limitations, Summary
"""

from __future__ import annotations

import os
import re
import json
from typing import List, Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader
import time
import google.generativeai as genai


load_dotenv()


DEFAULT_PAPERS_FOLDER = r"F:\\Python\\SAIM\\Research Assistant\\papers\\Skills Over Time"
RELEVANCY_CSV_PATH = "relevancy_results.csv"
OUTPUT_CSV_PATH = "papers_extracted_fields.csv"


def configure_api(model_name: str = "gemini-2.5-flash") -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API environment variable is not set. Ensure .env has GEMINI_API=..."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def load_relevant_papers(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Relevancy CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Normalize column names
    lower_map = {c.lower(): c for c in df.columns}
    required = {"title", "abstract", "relevant"}
    if not required.issubset(set(lower_map.keys())):
        raise ValueError(
            f"CSV must have columns {required}. Found: {list(df.columns)}"
        )
    # Rename to standard
    for col in list(required):
        actual = lower_map.get(col)
        if actual and actual != col:
            df = df.rename(columns={actual: col})
    # Filter relevant rows
    def to_bool(val) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        s = str(val).strip().lower()
        return s in {"true", "yes", "1", "y"}

    df["relevant"] = df["relevant"].apply(to_bool)
    return df[df["relevant"] == True][["title", "abstract"]].reset_index(drop=True)


def normalize_text_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def find_pdf_for_title(title: str, folder_path: str) -> Optional[str]:
    if not os.path.isdir(folder_path):
        return None
    norm_title = normalize_text_for_match(title)
    best_path = None
    best_score = 0.0

    for name in os.listdir(folder_path):
        if not name.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(folder_path, name)
        stem = os.path.splitext(name)[0]
        norm_stem = normalize_text_for_match(stem)

        # Simple token overlap score
        title_tokens = set(norm_title.split())
        stem_tokens = set(norm_stem.split())
        if not title_tokens or not stem_tokens:
            continue
        overlap = len(title_tokens & stem_tokens) / max(1, len(title_tokens))

        # Bonus if stem contains a large part of the title as substring
        substring_bonus = 0.2 if norm_title[:50] in norm_stem else 0.0
        score = overlap + substring_bonus

        # Try metadata title to improve score
        try:
            reader = PdfReader(pdf_path)
            meta = getattr(reader, "metadata", None)
            meta_title = ""
            if meta:
                mt = getattr(meta, "title", None)
                if isinstance(mt, str):
                    meta_title = mt
                else:
                    mt2 = None
                    try:
                        mt2 = meta.get("/Title")  # type: ignore[attr-defined]
                    except Exception:
                        mt2 = None
                    if isinstance(mt2, str):
                        meta_title = mt2
            if meta_title:
                norm_meta = normalize_text_for_match(meta_title)
                meta_tokens = set(norm_meta.split())
                overlap_meta = len(title_tokens & meta_tokens) / max(1, len(title_tokens))
                score = max(score, overlap_meta + 0.1)
        except Exception:
            pass

        if score > best_score:
            best_score = score
            best_path = pdf_path

    # Require minimal score to avoid bad matches
    if best_score < 0.2:
        return None
    return best_path


def read_full_text(pdf_path: str, max_chars: int = 250_000) -> str:
    reader = PdfReader(pdf_path)
    texts: List[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            texts.append(t)
        if sum(len(s) for s in texts) > max_chars:
            break
    combined = "\n\n".join(texts).strip()
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined


EXTRACTION_PROMPT_TEMPLATE = (
    "You are an expert research assistant. Given the full text of a research paper, extract the following fields **in the exact order** listed below.  \n\n"
    "For each field, provide a concise and accurate answer based only on the text. If a field is not mentioned or cannot be found, respond with \"Not mentioned\".\n\n"
    "Fields to extract:\n\n"
    "1. Authors: List all author names exactly as they appear.\n"
    "2. Objective: Summarize the main objective or goal of the paper in 1-2 sentences.\n"
    "3. Methodology: Describe the methodology or approach used in the study.\n"
    "4. Algorithms Used: List any specific algorithms, models, or techniques employed.\n"
    "5. Dataset Used: Name the dataset(s) used for experiments or evaluation.\n"
    "6. Example of Working Methodology: Provide a concrete example or description illustrating how the methodology works.\n"
    "7. Results: Summarize the key results or findings of the paper.\n"
    "8. Limitations: Note any limitations, drawbacks, or challenges mentioned by the authors.\n"
    "9. Summary: Provide a brief overall summary of the paper (3-4 sentences).\n\n"
    "Return your answer as a JSON object with the exact field names above as keys.\n\n"
    "Example output format:\n"
    "{\n"
    "  \"Authors\": \"...\",\n"
    "  \"Objective\": \"...\",\n"
    "  \"Methodology\": \"...\",\n"
    "  \"Algorithms Used\": \"...\",\n"
    "  \"Dataset Used\": \"...\",\n"
    "  \"Example of Working Methodology\": \"...\",\n"
    "  \"Results\": \"...\",\n"
    "  \"Limitations\": \"...\",\n"
    "  \"Summary\": \"...\"\n"
    "}\n\n"
    "Paper full text below:\n"
    "-----------------------------\n"
    "{paper_text}\n"
    "-----------------------------\n"
)


def _strip_code_fences(possible_json: str) -> str:
    text = possible_json.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", text).strip()
    return text


def _extract_json_object(possible_json: str) -> Optional[str]:
    match = re.search(r"\{[\s\S]*\}", possible_json)
    if match:
        return match.group(0)
    return None


FIELD_NAMES = [
    "Authors",
    "Objective",
    "Methodology",
    "Algorithms Used",
    "Dataset Used",
    "Example of Working Methodology",
    "Results",
    "Limitations",
    "Summary",
]


def extract_fields_from_text(model: genai.GenerativeModel, paper_text: str) -> Dict[str, str]:
    prompt = EXTRACTION_PROMPT_TEMPLATE.replace("{paper_text}", paper_text)
    response = model.generate_content(prompt)
    raw = getattr(response, "text", "") or ""
    cleaned = _strip_code_fences(raw)
    candidate = _extract_json_object(cleaned) or cleaned
    try:
        data = json.loads(candidate)
        result = {name: str(data.get(name, "Not mentioned")).strip() for name in FIELD_NAMES}
        return result
    except Exception:
        # Fallback: fill all as Not mentioned, keep raw in Summary
        return {name: (cleaned if name == "Summary" else "Not mentioned") for name in FIELD_NAMES}


def process_papers(
    papers_folder: str,
    relevancy_csv_path: str = RELEVANCY_CSV_PATH,
    output_csv_path: str = OUTPUT_CSV_PATH,
) -> pd.DataFrame:
    model = configure_api()
    relevant_df = load_relevant_papers(relevancy_csv_path)
    total = len(relevant_df)
    print(f"Found {total} relevant paper(s) to process.")

    rows: List[Dict[str, str]] = []
    api_calls_made = 0
    for idx, (_, row) in enumerate(relevant_df.iterrows(), start=1):
        title = str(row["title"]).strip()
        print(f"[{idx}/{total}] Processing: {title}")
        abstract = str(row["abstract"]).strip()
        pdf_path = find_pdf_for_title(title, papers_folder)

        if not pdf_path:
            extracted = {name: "Not mentioned" for name in FIELD_NAMES}
            extracted["Summary"] = "PDF not found for title"
        else:
            paper_text = read_full_text(pdf_path)
            extracted = extract_fields_from_text(model, paper_text)
            api_calls_made += 1
            if api_calls_made % 9 == 0:
                print("Hit 9 API calls. Sleeping 60 seconds to respect rate limits...")
                time.sleep(60)

        output_row = {
            "title": title,
            "abstract": abstract,
        }
        output_row.update(extracted)
        rows.append(output_row)

    print(f"Completed processing {total} relevant paper(s).")
    out_df = pd.DataFrame(rows, columns=["title", "abstract"] + FIELD_NAMES)
    out_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    return out_df


if __name__ == "__main__":
    folder = DEFAULT_PAPERS_FOLDER
    try:
        df = process_papers(folder)
        print(f"Processed {len(df)} relevant papers. Saved to {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"Error: {e}")
