"""
Convert papers_extracted_fields.csv rows into formatted single-page PDFs.

Steps:
- Read papers_extracted_fields.csv 
- For each row, use Gemini 2.5 Flash to format the data into a well-structured single-page text
- Convert the formatted text to PDF using reportlab
- Save each paper as a separate PDF file
"""

from __future__ import annotations

import os
import re
import time
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch


load_dotenv()

DEFAULT_CSV_PATH = "papers_extracted_fields.csv"
OUTPUT_FOLDER = "generated_pdfs"


def configure_api(model_name: str = "gemini-2.5-flash") -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API environment variable is not set. Ensure .env has GEMINI_API=..."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def build_formatting_prompt(row_data: Dict[str, str]) -> str:
    """Create a prompt for Gemini to format the paper data into a well-structured single page."""
    title = row_data.get('title', 'Not provided')
    abstract = row_data.get('abstract', 'Not provided')
    authors = row_data.get('Authors', 'Not provided')
    objective = row_data.get('Objective', 'Not provided')
    methodology = row_data.get('Methodology', 'Not provided')
    algorithms_used = row_data.get('Algorithms Used', 'Not provided')
    dataset_used = row_data.get('Dataset Used', 'Not provided')
    example_methodology = row_data.get('Example of Working Methodology', 'Not provided')
    results = row_data.get('Results', 'Not provided')
    limitations = row_data.get('Limitations', 'Not provided')
    summary = row_data.get('Summary', 'Not provided')
    
    return f"""I will give you the following details of a research paper:

Title: {title}
Abstract: {abstract}
Authors: {authors}
Objective: {objective}
Methodology: {methodology}
Algorithms Used: {algorithms_used}
Dataset Used: {dataset_used}
Example of Working Methodology: {example_methodology}
Results: {results}
Limitations: {limitations}
Summary: {summary}

Using these, create a one-page summary in the following format:
Title of paper  
Author(s)  
[Plain text summary here...]

Instructions for writing:
- Do not include section headings (like Objective, Methodology, etc.).
- Write in clear, simple, and non-technical language so that anyone can understand, not just researchers.
- Keep the flow natural, like telling the story of the research from start to end.
- Make sure it reads smoothly as a single piece of text.
- Do not use bullet points.
- Do not add extra commentary beyond the given details."""


def format_paper_with_gemini(model: genai.GenerativeModel, row_data: Dict[str, str]) -> str:
    """Use Gemini to format the paper data into a well-structured text."""
    prompt = build_formatting_prompt(row_data)
    try:
        response = model.generate_content(prompt)
        formatted_text = getattr(response, "text", "") or ""
        return formatted_text.strip()
    except Exception as e:
        # Fallback formatting if Gemini fails
        title = row_data.get('title', 'Untitled Paper')
        return f"""RESEARCH PAPER SUMMARY

{title}

AUTHORS: {row_data.get('Authors', 'Not provided')}

ABSTRACT
{row_data.get('abstract', 'Not provided')}

OBJECTIVE
{row_data.get('Objective', 'Not provided')}

METHODOLOGY
{row_data.get('Methodology', 'Not provided')}

ALGORITHMS USED
{row_data.get('Algorithms Used', 'Not provided')}

DATASET USED
{row_data.get('Dataset Used', 'Not provided')}

WORKING METHODOLOGY EXAMPLE
{row_data.get('Example of Working Methodology', 'Not provided')}

RESULTS
{row_data.get('Results', 'Not provided')}

LIMITATIONS
{row_data.get('Limitations', 'Not provided')}

SUMMARY
{row_data.get('Summary', 'Not provided')}

Note: This document was generated automatically due to formatting service unavailability."""


def clean_filename(title: str) -> str:
    """Clean title for use as filename."""
    # Remove invalid characters for filenames
    cleaned = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace spaces and other characters with underscores
    cleaned = re.sub(r'[\s\-\[\](){},.;:!@#$%^&*+=]', '_', cleaned)
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    # Limit length
    if len(cleaned) > 100:
        cleaned = cleaned[:100]
    # Remove trailing underscores
    cleaned = cleaned.strip('_')
    return cleaned or 'untitled_paper'


def create_combined_pdf(formatted_texts: List[str], output_path: str) -> None:
    """Create a single PDF from multiple formatted texts using reportlab."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=14,
        spaceAfter=12,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=12,
        spaceAfter=6,
        textColor='black'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        alignment=0  # Left alignment
    )
    
    separator_style = ParagraphStyle(
        'Separator',
        parent=styles['Normal'],
        fontSize=12,
        spaceBefore=20,
        spaceAfter=20,
        alignment=1  # Center alignment
    )
    
    # Create content for all papers
    story = []
    
    for i, text in enumerate(formatted_texts):
        # Add separator between papers (except for the first one)
        if i > 0:
            story.append(Spacer(1, 20))
            story.append(Paragraph("─" * 50, separator_style))
            story.append(Spacer(1, 20))
        
        # Parse text into paragraphs and add to story
        lines = text.split('\n')
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_paragraph:
                    story.append(Paragraph(' '.join(current_paragraph), body_style))
                    current_paragraph = []
                story.append(Spacer(1, 6))
            elif line.isupper() and len(line.split()) <= 6:  # Likely a heading
                if current_paragraph:
                    story.append(Paragraph(' '.join(current_paragraph), body_style))
                    current_paragraph = []
                story.append(Paragraph(line, heading_style))
            elif line.startswith('RESEARCH PAPER') or line.endswith('SUMMARY'):
                if current_paragraph:
                    story.append(Paragraph(' '.join(current_paragraph), body_style))
                    current_paragraph = []
                story.append(Paragraph(line, title_style))
            else:
                current_paragraph.append(line)
        
        # Add any remaining paragraph
        if current_paragraph:
            story.append(Paragraph(' '.join(current_paragraph), body_style))
    
    # Build PDF
    doc.build(story)


def process_csv_to_single_pdf(
    csv_path: str = DEFAULT_CSV_PATH,
    output_filename: str = "all_papers_summary.pdf"
) -> None:
    """Process the CSV file and generate a single PDF containing all papers."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    total = len(df)
    print(f"Found {total} paper(s) to convert to single PDF.")
    
    # Configure Gemini
    model = configure_api()
    
    api_calls_made = 0
    formatted_texts = []
    successful_formats = 0
    
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        title = str(row.get('title', f'Paper_{idx}')).strip()
        print(f"[{idx}/{total}] Processing: {title}")
        
        # Convert row to dictionary
        row_data = {col: str(row.get(col, 'Not provided')).strip() for col in df.columns}
        
        try:
            # Format with Gemini
            formatted_text = format_paper_with_gemini(model, row_data)
            api_calls_made += 1
            formatted_texts.append(formatted_text)
            successful_formats += 1
            print(f"    ✓ Formatted successfully")
            
            # Rate limiting - process 9 rows then sleep for 60 seconds
            if api_calls_made % 9 == 0:
                print("Hit 9 API calls. Sleeping 60 seconds to respect rate limits...")
                time.sleep(60)
                
        except Exception as e:
            print(f"    ✗ Error processing {title}: {e}")
            # Add a fallback formatted text for failed papers
            fallback_text = f"{title}\n{row_data.get('Authors', 'Unknown Authors')}\n\nThis paper could not be processed due to an error: {str(e)}"
            formatted_texts.append(fallback_text)
            continue
    
    # Create single PDF with all papers
    if formatted_texts:
        print(f"\nCreating combined PDF with {len(formatted_texts)} papers...")
        try:
            create_combined_pdf(formatted_texts, output_filename)
            print(f"✅ Successfully created combined PDF: {output_filename}")
            print(f"Successfully formatted {successful_formats} out of {total} papers.")
        except Exception as e:
            print(f"❌ Error creating combined PDF: {e}")
    else:
        print("No papers were successfully formatted. No PDF created.")


if __name__ == "__main__":
    try:
        process_csv_to_single_pdf()
    except Exception as main_error:
        print(f"Error: {main_error}")