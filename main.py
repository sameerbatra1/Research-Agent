import os
import sys
import subprocess

def main():
    print("========================================")
    print("          Research Agent Pipeline       ")
    print("========================================")
    
    # 1. Ask user for PDF folder
    pdf_folder = input("\nEnter the absolute path to the folder containing your PDFs: ").strip()
    # Remove quotes if user dragged and dropped the folder into the terminal
    pdf_folder = pdf_folder.strip("'").strip('"')
    if not os.path.isdir(pdf_folder):
        print(f"Error: Directory '{pdf_folder}' does not exist.")
        sys.exit(1)

    # 2. Get Research Requirements
    print("\n========================================")
    print("--- Step 1: Requirements Gathering ---")
    print("========================================")
    use_existing = False
    if os.path.isfile("requirements_summary.txt"):
        ans = input("Found existing 'requirements_summary.txt'. Do you want to use it? (y/n): ").strip().lower()
        if ans == 'y':
            use_existing = True
            
    if not use_existing:
        print("Starting Brainstorming Agent...\n")
        # Run brainstorming_agent.py as a subprocess because it uses interactive inputs in __main__
        result = subprocess.run([sys.executable, "brainstorming_agent.py"])
        if result.returncode != 0:
            print("Brainstorming Agent failed or was interrupted. Exiting.")
            sys.exit(result.returncode)
        
        if not os.path.isfile("requirements_summary.txt"):
            print("Failed to create 'requirements_summary.txt'. Exiting.")
            sys.exit(1)

    # 3. Extract Abstracts
    print("\n========================================")
    print("--- Step 2: Extracting Abstracts from PDFs ---")
    print("========================================")
    from abstract_extrator import extract_abstracts_from_pdfs
    extract_abstracts_from_pdfs(
        folder_path=pdf_folder,
        model_name="gemini-2.5-flash",
        save_csv_path="extracted_abstracts.csv"
    )

    # 4. Evaluate Relevancy
    print("\n========================================")
    print("--- Step 3: Evaluating Relevancy ---")
    print("========================================")
    from relevancy_finder import evaluate_relevancy, load_abstracts_csv
    with open("requirements_summary.txt", "r", encoding="utf-8") as f:
        requirements = f.read().strip()
    
    abstracts_df = load_abstracts_csv("extracted_abstracts.csv")
    relevancy_df = evaluate_relevancy(requirements, abstracts_df)
    relevancy_df.to_csv("relevancy_results.csv", index=False, encoding="utf-8-sig")
    
    num_relevant = int(relevancy_df["relevant"].sum())
    print(f"Evaluated {len(relevancy_df)} papers. Relevant: {num_relevant}. Saved to 'relevancy_results.csv'")

    if num_relevant == 0:
        print("No relevant papers found based on your requirements. Exiting pipeline.")
        sys.exit(0)

    # 5. Extract Structured Fields
    print("\n========================================")
    print("--- Step 4: Extracting Structured Fields from Relevant PDFs ---")
    print("========================================")
    from output import process_papers
    extracted_df = process_papers(
        papers_folder=pdf_folder,
        relevancy_csv_path="relevancy_results.csv",
        output_csv_path="papers_extracted_fields.csv"
    )
    print(f"Extracted fields for {len(extracted_df)} papers. Saved to 'papers_extracted_fields.csv'")

    # 6. Generate Summary PDF
    print("\n========================================")
    print("--- Step 5: Generating PDF Summary ---")
    print("========================================")
    from to_pdf import process_csv_to_single_pdf
    process_csv_to_single_pdf(
        csv_path="papers_extracted_fields.csv",
        output_filename="all_papers_summary.pdf"
    )

    print("\n=======================================================================")
    print(" Pipeline Complete! Check 'all_papers_summary.pdf' for your final output.")
    print("=======================================================================\n")

if __name__ == "__main__":
    main()
