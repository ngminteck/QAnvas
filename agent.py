from typing import List, Dict
import requests
import os
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import Tool
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate
from canvas import CanvasManager
from langchain_community.chat_models import ChatDeepSeekAI
from functools import partial

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
        description="Retrieve details of a specific assignment. Inputs: assignment_name (str), threshold (optional, default=0.7)."
    ),
    Tool(
        name="List Announcements",
        func=partial(manager.list_announcements),
        description="List recent announcements. Inputs: hide_older_than (int, days), only_unread (optional, default=False)."
    ),
    Tool(
        name="Get Announcement Detail",
        func=partial(manager.get_announcement_detail),
        description="Retrieve details of a specific announcement. Inputs: announcement_title (str), threshold (optional, default=0.7)."
    )
]

# Define the prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful Canvas Q&A assistant.

You can access the following tools:
{tools}

Question: {input}
{chat_history}
""")

# Initialize DeepSeek LLM
llm = ChatDeepSeekAI(
    model_name="deepseek-reasoner",
    temperature=0.7,
    api_key=deepseek_api_key
)

# Memory using the new API
memory = ConversationSummaryMemory(llm=llm, return_messages=True)


# Create a function that processes user queries
def ask_canvas_agent(query: str):
    context = {"input": query, "chat_history": memory.load_memory_variables({})}

    # Ensure pipeline execution
    full_prompt = prompt.format(tools=[tool.description for tool in tools], input=query,
                                chat_history=context["chat_history"])
    response = llm.invoke([{"role": "user", "content": full_prompt}])

    memory.save_context(context, {"output": response})
    return response


# Example usage
if __name__ == "__main__":
    print("Welcome to the Canvas Q&A Agent!")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = ask_canvas_agent(query)
        print(f"Agent: {response}")
