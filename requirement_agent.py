# This agent gathers requirements for a research project using Azure DeepSeek (R1) chat completions.
import os
import re
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
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

def complete_text(messages, max_tokens: int = 1024) -> str:
    response = client.complete(
        messages=messages,
        max_tokens=max_tokens,
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
            "Assume you are the best research requirements agent out there. "
            "I am planning on working on a research project. Ask me 5 small questions to gather requirements for the project. "
            "We are not aiming for every detail right now. These requirements are for analysing which papers in my database are closest to my research so no need to draft complete plans."
        )),
    ]
    return complete_text(messages, max_tokens=1024)


if __name__ == "__main__":
    print("Requirement Agent")
    try:
        # Example usage
        requirements = requirement_agent_initial_questions()
        print(requirements)
        print("==="*50)
        content = "--"
        conversation = [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content=requirements),
        ]

        # First user answer and follow-up
        user_answer = input("Answer: ")
        conversation.append(UserMessage(content=user_answer))
        conversation.append(UserMessage(content="Ask me follow up questions based on the previous conversation."))
        followup_1 = complete_text(conversation, max_tokens=512)
        print(followup_1)

        # Second user answer
        user_answer = input("Answer: ")
        conversation.append(UserMessage(content=user_answer))

        print("==="*50)
        # Summarize requirements based on the conversation
        conversation.append(UserMessage(content=(
            "Using the previous conversation about requirements gathering, summarize the key points and requirements for the research project."
        )))
        summary_text = complete_text(conversation, max_tokens=512)
        summary_text_clean = strip_think_tags(summary_text).strip()
        print("Summary of Requirements:", summary_text_clean)

        # Save summary to requirements_summary.txt in this script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        summary_path = os.path.join(script_dir, "requirements_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text_clean)
        print(f"Saved requirements summary to: {summary_path}")
        
    except Exception as e:
        print(f"Error gathering requirements: {str(e)}")
        print("Make sure your Azure endpoint, model deployment, and API key are valid and set in the .env file")