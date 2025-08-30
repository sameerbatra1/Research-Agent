"""
Extract abstracts from PDFs using the first two pages and Gemini.

- Reads PDFs from a user-provided folder
- Extracts text from the first two pages of each PDF
- Calls Gemini (same pattern as requirement_agent) to extract or infer the abstract
- Returns a pandas DataFrame with columns: title, abstract
"""

from __future__ import annotations
import os
from typing import List, Dict, Optional
import time

import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader
import google.generativeai as genai


load_dotenv()


def configure_api(model_name: str = "gemini-2.5-flash") -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API environment variable is not set. Ensure .env has GEMINI_API=..."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def read_first_two_pages_text(pdf_path: str, max_chars: int = 40000) -> str:
    """Extract text from the first two pages of the given PDF.

    max_chars prevents sending overly long prompts.
    """
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    pages_to_read = min(2, num_pages)
    collected_text_parts: List[str] = []
    for page_index in range(pages_to_read):
        page = reader.pages[page_index]
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        collected_text_parts.append(page_text)

    combined_text = "\n\n".join(collected_text_parts).strip()
    if len(combined_text) > max_chars:
        combined_text = combined_text[:max_chars]
    return combined_text


def get_pdf_title(pdf_path: str) -> str:
    """Best-effort title extraction: metadata title or filename stem."""
    try:
        reader = PdfReader(pdf_path)
        metadata = getattr(reader, "metadata", None)
        if metadata:
            # Try attribute access first (pypdf), then key access (PyPDF2)
            title_attr = getattr(metadata, "title", None)
            if title_attr and isinstance(title_attr, str) and title_attr.strip():
                return title_attr.strip()
            try:
                title_key = metadata.get("/Title")  # type: ignore[attr-defined]
                if title_key and isinstance(title_key, str) and title_key.strip():
                    return title_key.strip()
            except Exception:
                pass
    except Exception:
        pass
    # Fallback to filename without extension
    return os.path.splitext(os.path.basename(pdf_path))[0]


def build_abstract_prompt(first_two_pages_text: str) -> str:
    """Create a prompt instructing Gemini to extract or infer the abstract."""
    return (
        "You are an expert research assistant. You will receive the first two pages of an academic paper.\n"
        "Your task: Extract the paper's Abstract.\n"
        "- If an explicit 'Abstract' section exists, return it verbatim (without adding any headers).\n"
        "- If no explicit Abstract is present, infer a abstract with objective, methods, data, key findings, and contributions.\n"
        "- Return only the abstract text. Do not include any explanations, headers, or labels.\n\n"
        "Paper first two pages below:\n"
        "-----------------------------\n"
        f"{first_two_pages_text}\n"
        "-----------------------------\n"
    )


def extract_abstracts_from_pdfs(folder_path: str, model_name: str = "gemini-2.5-flash", save_csv_path: Optional[str] = None,) -> pd.DataFrame:
    """Process all PDFs in a folder, extract abstracts using Gemini, and return a DataFrame.
    - folder_path: directory containing PDF files
    - model_name: Gemini model to use (defaults to same as requirement_agent)
    - save_csv_path: optional path to save the resulting DataFrame as CSV
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    model = configure_api(model_name)

    results: List[Dict[str, str]] = []
    pdf_filenames = [
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if name.lower().endswith(".pdf")
    ]

    total = len(pdf_filenames)
    print(f"Found {total} PDF(s) to process.")

    api_calls_made = 0
    for idx, pdf_path in enumerate(pdf_filenames, start=1):
        print(f"[{idx}/{total}] Processing: {os.path.basename(pdf_path)}")
        paper_title = get_pdf_title(pdf_path)
        try:
            first_two_pages = read_first_two_pages_text(pdf_path)
            if not first_two_pages:
                abstract_text = ""
            else:
                prompt = build_abstract_prompt(first_two_pages)
                response = model.generate_content(prompt)
                abstract_text = getattr(response, "text", "") or ""
                api_calls_made += 1
                if api_calls_made % 9 == 0:
                    print("Hit 9 API calls. Sleeping 80 seconds to respect rate limits...")
                    time.sleep(80)
        except Exception as error:
            abstract_text = f"Error extracting abstract: {error}"

        results.append({
            "title": paper_title,
            "abstract": abstract_text,
        })

    print(f"Completed processing {total} PDF(s).")

    df = pd.DataFrame(results, columns=["title", "abstract"])
    if save_csv_path:
        df.to_csv(save_csv_path, index=False)
    return df


if __name__ == "__main__":
    folder = r"F:\\Python\\SAIM\\Research Assistant\\papers\\Skill Extraction Tool" 
    try:
        dataframe = extract_abstracts_from_pdfs(
            folder_path=folder,
            model_name="gemini-2.5-flash",
            save_csv_path="extracted_abstracts.csv",
        )
        print(f"Processed {len(dataframe)} PDFs. Saved to extracted_abstracts.csv")
    except Exception as main_error:
        print(f"Error: {main_error}")
