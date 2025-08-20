import subprocess
import shlex
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import chat_agent_executor

load_dotenv()

# Explicitly get the API key from the environment
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Error: GOOGLE_API_KEY not found. Make sure it's in your .env file.")

@tool(description="Get staged git changes (diff).")
def git_diff() -> str:
  try:
    return subprocess.check_output(shlex.split("git diff --staged")).decode("utf-8", "ignore")
  except subprocess.CalledProcessError:
    return ""

@tool(description="Ask user if intent is unclear.")
def ask_user(prompt: str) -> str:
  return input((prompt or "The intent of the changes is unclear. Could you provide a brief summary?").strip() + " ")

# Pass the API key directly to the model during initialization
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=api_key)

agent = chat_agent_executor.create_tool_calling_executor(model, [git_diff, ask_user])

result = agent.invoke({
    "messages": [
        ("system", "You are a great commit message assistant. Use git_diff to see staged changes; stop if no diffs. "
                   "If the changes are obvious, write a conventional commit message. "
                   "If not, call ask_user once to get more context. "
                   "Always output one single-line conventional commit message and nothing else."),
        ("user", "Write the commit message for my staged changes.")
    ]
})

if result.get("messages") and len(result["messages"]) > 1:
    print(result["messages"][-1].content.strip())
else:
    print("Could not generate a commit message. Please check for staged changes.")