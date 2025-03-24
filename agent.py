import json
from typing import List, Dict
import os
from functools import partial
from langchain.tools import Tool
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate
from canvas import CanvasManager
from langchain_deepseek import ChatDeepSeek

# Load DeepSeek API key
deepseek_api_key = ""
try:
    with open("keys/deepseek.txt", "r") as file:
        deepseek_api_key = file.read().strip()
except Exception as e:
    print("Error reading DeepSeek API key:", e)

# Load Canvas API key
canvas_api_key = ""
API_URL = "https://canvas.nus.edu.sg/"
try:
    with open("keys/canvas.txt", "r") as file:
        canvas_api_key = file.read().strip()
except Exception as e:
    print("Error reading Canvas API key:", e)

# Initialize Canvas Manager
manager = CanvasManager(API_URL, canvas_api_key)

# Define tools
tools = [
    Tool(
        name="Retrieve Lecture Slides",
        func=partial(manager.retrieve_lecture_slides_by_topic),
        description="Retrieve lecture slides related to a topic. Inputs: topic (str), filter_terms (optional)."
    ),
    Tool(
        name="Get Timetable",
        func=partial(manager.get_timetable),
        description="Retrieve the timetable for a course. Inputs: full_time (bool), intake (str), course (str)."
    ),
    Tool(
        name="List Upcoming Assignments",
        func=partial(manager.list_upcoming_assignments),
        description="List upcoming assignments. Input: hide_older_than (int, days)."
    ),
    Tool(
        name="Get Assignment Detail",
        func=partial(manager.get_assignment_detail),
        description="Retrieve details of a specific assignment. Inputs: assignment_name (str), threshold (optional, default=0.8)."
    ),
    Tool(
        name="List Announcements",
        func=partial(manager.list_announcements),
        description="List recent announcements. Inputs: hide_older_than (int, days), only_unread (optional, default=False)."
    ),
    Tool(
        name="Get Announcement Detail",
        func=partial(manager.get_announcement_detail),
        description="Retrieve details of a specific announcement. Inputs: announcement_title (str), threshold (optional, default=0.8)."
    )
]

# Define the prompt template.
# The prompt instructs the assistant to output a JSON object with "action" and "action_input" keys
# when a tool call is needed.
prompt = ChatPromptTemplate.from_template("""
You are a helpful Canvas Q&A assistant.

When a tool call is needed, output a JSON object with exactly two keys:
  - "action": one of the tool names below.
  - "action_input": a dictionary of inputs for the tool.

If no tool call is needed, simply output your answer as plain text.

You can access the following tools:
{tools}

Question: {input}
{chat_history}
""")

llm = ChatDeepSeek(
    model_name="deepseek-reasoner",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=deepseek_api_key
)

# Conversation memory now expects a single input key "input"
memory = ConversationSummaryMemory(llm=llm, return_messages=True)


def process_agent_response(response_text: str):
    # Debug: print raw response received from the agent
    print("Debug: Raw agent response:", response_text)

    # Remove markdown code fences if present.
    if response_text.startswith("```"):
        lines = response_text.splitlines()
        json_lines = [line for line in lines if not line.strip().startswith("```")]
        response_text = "\n".join(json_lines).strip()

    try:
        response_json = json.loads(response_text)
        # Debug: print parsed JSON for tool call decision
        print("Debug: Parsed JSON:", response_json)
    except json.JSONDecodeError:
        # No JSON tool call; return the plain text response.
        print("Debug: Response is not a JSON tool call. Returning plain text.")
        return response_text

    action = response_json.get("action")
    action_input = response_json.get("action_input", {})

    # Debug: print the tool action and its input
    print("Debug: Action to call:", action)
    print("Debug: Action input:", action_input)

    # Map action names to their corresponding functions.
    tool_mapping = {
        "Retrieve Lecture Slides": manager.retrieve_lecture_slides_by_topic,
        "Get Timetable": manager.get_timetable,
        "List Upcoming Assignments": manager.list_upcoming_assignments,
        "Get Assignment Detail": manager.get_assignment_detail,
        "Retrieve details of a specific assignment": manager.get_assignment_detail,
        "List Announcements": manager.list_announcements,
        "Get Announcement Detail": manager.get_announcement_detail,
        "Retrieve details of a specific announcement": manager.get_announcement_detail,
    }

    if action in tool_mapping:
        tool_func = tool_mapping[action]
        try:
            tool_result = tool_func(**action_input)
            return tool_result
        except Exception as e:
            return f"Error while executing tool '{action}': {e}"
    else:
        return f"Unknown action: {action}"


def ask_canvas_agent(query: str):
    # Load conversation history from memory (if any)
    chat_history = memory.load_memory_variables({})
    full_prompt = prompt.format(
        tools=[tool.description for tool in tools],
        input=query,
        chat_history=chat_history
    )

    # Debug: print the full prompt sent to the LLM
    print("Debug: Full prompt sent to agent:\n", full_prompt)

    response = llm.invoke([{"role": "user", "content": full_prompt}])
    response_text = response.content if hasattr(response, "content") else str(response)

    # Debug: print the raw response received
    print("Debug: Raw agent response:\n", response_text)

    memory.save_context({"input": query}, {"output": response_text})
    final_result = process_agent_response(response_text)
    return final_result


if __name__ == "__main__":
    print("Welcome to the Canvas Q&A Agent!")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        result = ask_canvas_agent(query)
        print(f"Agent: {result}")
