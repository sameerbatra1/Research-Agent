import os
import re
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

endpoint = os.getenv("AZURE_INFERENCE_ENDPOINT", "https://llmapikeys.services.ai.azure.com/models")
model_name = os.getenv("AZURE_INFERENCE_DEPLOYMENT", "DeepSeek-R1")
deep_seek_api = os.getenv("AZURE_KEY_DEEPSEEK")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(deep_seek_api),
    api_version="2024-05-01-preview",
)

def complete_text(messages) -> str:
    response = client.complete(
        messages=messages,
        model=model_name,
    )
    # Try to extract text robustly
    try:
        choice = response.choices[0]
        msg = getattr(choice, "message", None)
        if msg and getattr(msg, "content", None):
            # content may be a list of items with .text
            content = msg.content
            if isinstance(content, list):
                return "".join(getattr(part, "text", "") for part in content)
            return str(content)
    except Exception:
        pass
    # Fallbacks used by some SDK builds
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    return ""


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from the model output."""
    if not text:
        return text
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)

def requirement_agent_initial_questions() -> str:
    messages = [
        SystemMessage(content="You are a helpful research assistant."),
        UserMessage(content=(
            """
            You are a requirements brainstorming helper. Your goal is to start by asking the user broad, open-ended questions about their research project.
            Instructions:
            1. Begin the session by explaining that you will help gather requirements step by step.
            2. Ask broad, high-level questions to get an overview of the project.
            3. Avoid making assumptions or giving recommendations yet.
            
            Cover areas such as:
            Overall research goals
            Motivation / why the project matters
            Target audience or beneficiaries
            Expected outcomes or deliverables
            General research domain (e.g., AI, healthcare, economics)

            Tone:
            Curious, supportive, and professional. Encourage the user to think aloud if unsure.
"""
        )),
    ]
    return complete_text(messages)

if __name__ == "__main__":
    print("Requirements Brainstorming Assistant")
    try:
        questions_raw = requirement_agent_initial_questions()
        questions = strip_think_tags(questions_raw).strip()
        print(questions)
        print("==="*50)
        conversation = [
            SystemMessage(content="You are a helpful brainstorming assistant."),
            AssistantMessage(content=questions),
        ]
        user_answer = input("Answer: ")
        conversation.append(UserMessage(content=user_answer))
        followup_1 = complete_text(conversation)
        # First cycle complete

        # Second Cycle Start
        prompt_2 = (
            """
            Role:
                You are continuing as a requirements brainstorming helper. Now that the user has shared broad details, your task is to ask clarifying and detailed follow-up questions.

                Instructions:
                Analyze the user’s previous responses.
                Identify areas that need more detail, clarification, or specificity.
                Ask targeted, probing questions to uncover details such as:
                Research methods or approaches they are considering
                Specific subdomains, theories, or techniques of interest
                Timeframe for relevant literature (e.g., last 5 years, historical perspective)
                Types of publications they care about (journals, conferences, preprints)
                Inclusion or exclusion criteria (e.g., empirical studies only, exclude surveys)
                Constraints (budget, data availability, computational resources)
                Measures of success (what would make this research useful/valid)
                
                Tone:
                Inquisitive but analytical, like a consultant drilling into specifics.
            """
        )
        
        conversation.append(SystemMessage(content=prompt_2))

        followup_2_raw = complete_text(conversation)
        followup_2 = strip_think_tags(followup_2_raw).strip()
        print(followup_2)

        conversation.append(AssistantMessage(content=followup_2))

        user_answer2 = input("Answer: ")
        conversation.append(UserMessage(content=user_answer2))

        # Second follow-up complete

        # Third Cycle Start
        prompt_3 = (
            """
            Role:
            You are a requirements analyst who has now collected broad and detailed inputs from the user. Your goal is to give recommendations and suggestions for refining their requirements.

            Instructions:
            Review all the user’s answers so far.
            Identify gaps, ambiguities, or areas that could use refinement.
            Suggest additional criteria they may want to consider (e.g., quality of papers, regional focus, open-access requirement, ethical considerations).
            Offer recommendations that strengthen the requirements and make them more practical for evaluation.
            Present recommendations in a clear list, then explicitly ask the user if they want to adopt any of them.

            Tone:
            Advisory, constructive, and professional. Frame suggestions as helpful, not corrective.
            """
        )

        conversation.append(SystemMessage(content=prompt_3))

        followup_3_raw = complete_text(conversation)
        followup_3 = strip_think_tags(followup_3_raw).strip()
        print(followup_3)

        conversation.append(AssistantMessage(content=followup_3))

        user_answer_3 = input("Answer: ")
        conversation.append(UserMessage(content=user_answer_3))

        # Third follow-up complete

        # Fourth Cycle Start - Requirements Summary
        prompt_4 = (
            """
            Role:
                You are a requirements summarizer. Your only job is to summarize the user’s inputs and decisions from the conversation in a structured way.

            Instructions:
                Do not add new ideas, recommendations, or assumptions.
                Only reformat and organize the requirements that have already been discussed.
                Present the requirements as clear bullet points, grouped under meaningful categories such as:
                Project Goals
                Scope & Focus
                Constraints & Criteria
                Exclusions
                Success Measures
                If any category has no information, simply omit it — do not invent content.
                The output must be concise, professional, and ready to save to a .txt file.

            Tone:
                Neutral, structured, and factual.
            """
        )

        conversation.append(SystemMessage(content=prompt_4))

        summary = complete_text(conversation)
        summary = strip_think_tags(summary).strip()
        print(summary)

        conversation.append(AssistantMessage(content=summary))

        # Save summary to requirements_summary.txt in this script's directory
        # summary_text_clean = strip_think_tags(summary).strip()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        summary_path = os.path.join(script_dir, "requirements_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Saved requirements summary to: {summary_path}")
    except Exception as e:
        print(f"Error gathering requirements: {str(e)}")
        print("Make sure your Azure endpoint, model deployment, and API key are valid and set in the .env file")