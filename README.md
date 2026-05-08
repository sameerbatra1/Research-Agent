# Research Agent

An AI-powered pipeline to automate your research literature review process. The Research Agent intelligently extracts, evaluates, and summarizes academic papers based on your custom research requirements.

## 🚀 Features

- **Interactive Requirements Gathering**: Chat with an AI assistant to articulate your research goals.
- **Automated Abstract Extraction**: Processes a local directory of PDF papers to extract their titles and abstracts.
- **Relevancy Filtering**: Automatically evaluates each paper against your specific research requirements to find the most relevant literature.
- **Deep Information Extraction**: Extracts structured fields (Authors, Objective, Methodology, Results, Limitations, etc.) from the full text of relevant papers.
- **PDF Summary Generation**: Compiles all relevant paper summaries into a beautifully formatted, easy-to-read PDF document.

## 🏗️ Architecture Pipeline

1. **`brainstorming_agent.py`**: Interactive CLI powered by Azure DeepSeek-R1 to determine research requirements. Saves output to `requirements_summary.txt`.
2. **`abstract_extrator.py`**: Uses Gemini 2.5 Flash to parse PDFs and extract abstracts. Outputs `extracted_abstracts.csv`.
3. **`relevancy_finder.py`**: Evaluates abstract relevance using Azure OpenAI (GPT-4o-mini). Outputs `relevancy_results.csv`.
4. **`output.py`**: Extracts detailed structured fields from full text using Gemini. Outputs `papers_extracted_fields.csv`.
5. **`to_pdf.py`**: Formats the extracted data and converts it into `all_papers_summary.pdf` using ReportLab.
6. **`main.py`**: The orchestrator script that ties all modules together into a seamless end-to-end pipeline.

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- API keys for Gemini, Azure OpenAI, and Azure AI Inference (DeepSeek)

### 1. Clone & Install Dependencies
Clone this repository, then install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory of the project and populate it with your API keys:

```ini
# Gemini Configuration (for extraction and formatting)
GEMINI_API=your_gemini_api_key_here

# Azure OpenAI Configuration (for relevancy finding)
Azure_OpenAI_Endpoint=your_azure_openai_endpoint_here
Azure_OpenAI_Key=your_azure_openai_key_here

# Azure Inference Configuration (for the DeepSeek brainstorming agent)
AZURE_INFERENCE_ENDPOINT=your_azure_inference_endpoint_here
AZURE_INFERENCE_DEPLOYMENT=DeepSeek-R1
AZURE_KEY_DEEPSEEK=your_azure_deepseek_key_here
```

## 📖 Usage

To run the full end-to-end pipeline, simply execute the orchestrator script:

```bash
python main.py
```

**Step-by-Step Workflow:**
1. You will be prompted to enter the absolute path to your folder containing the PDF papers.
2. If you don't already have a `requirements_summary.txt`, the interactive Brainstorming Agent will launch to gather your research goals.
3. Sit back as the agent parses the PDFs, extracts abstracts, finds relevant papers, structures their data, and generates a final `all_papers_summary.pdf`!

## 📂 Project Structure

- `main.py` - Orchestrator script for the whole pipeline.
- `requirements.txt` - Python dependencies.
- `brainstorming_agent.py` - Requirement gathering agent (DeepSeek).
- `requirement_agent.py` - Simpler version of requirement gathering.
- `abstract_extrator.py` - Extracts abstracts from PDF (Gemini).
- `relevancy_finder.py` - Evaluates relevancy (Azure OpenAI).
- `output.py` - Structured data extraction (Gemini).
- `to_pdf.py` - Compiles output to a final PDF.