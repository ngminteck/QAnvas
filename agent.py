import json
import difflib
from functools import partial
from typing import Optional, List

from pydantic import BaseModel

from langchain.tools import StructuredTool  # New structured tool integration.
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from canvas import CanvasManager
import re
import os

# Define Pydantic models for tool input schemas.
class RetrieveLectureSlidesInput(BaseModel):
    query: str
    subjects: Optional[List[str]] = None


class GetTimetableInput(BaseModel):
    full_time: bool
    intake_year: int
    course: str


class GetAssignmentDetailInput(BaseModel):
    assignment_name: str


class GetAnnouncementDetailInput(BaseModel):
    announcement_title: str


class OtherInput(BaseModel):
    query: str


# For tools that require no input, we can simply use an empty model.
class EmptyInput(BaseModel):
    pass


class CanvasAgent:
    """
    A Canvas Q&A agent that detects multiple intents and synthesizes a final answer
    by leveraging various tools related to Canvas tasks using OpenAI's language models.

    The agent’s process is roughly:
      1. Use an LLM prompt to generate tool calls (intents) with required inputs.
      2. Validate those inputs using structured schemas.
      3. (Optionally, ask follow-up questions if inputs are missing.)
      4. Execute the tools and then synthesize a final answer.
    """

    def __init__(self):

        self.openai_api_key = ""

        with open("keys/openai.txt", "r") as file:
            self.openai_api_key = file.read().strip()

        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        # Initialize the Canvas Manager.
        self.manager = CanvasManager()

        # Consolidated tool info with structured input schemas.
        self.tool_info = {
            "Retrieve Lecture Slides": {
                "function": self.manager.retrieve_lecture_slides_by_topic,
                "description": ("Retrieve lecture slides related to a query. "
                                "Provide a 'query' string and (optionally) a list of 'subjects'."),
                "schema": RetrieveLectureSlidesInput,
            },
            "Get Timetable Or Exam Schedule": {
                "function": self.manager.get_timetable_or_exam_date,
                "description": ("Retrieve the timetable for a course. "
                                "Provide 'full_time' (bool), 'intake_year' (int), and 'course' (str)."),
                "schema": GetTimetableInput,
            },
            "List Upcoming Assignments": {
                "function": self.manager.list_upcoming_assignments,
                "description": "List upcoming assignments. (No input required.)",
                "schema": EmptyInput,
            },
            "Get Assignment Detail": {
                "function": self.manager.get_assignment_detail,
                "description": ("Retrieve details of a specific assignment. "
                                "Provide 'assignment_name' (str)."),
                "schema": GetAssignmentDetailInput,
            },
            "List Announcements": {
                "function": self.manager.list_announcements,
                "description": "List recent announcements. (No input required.)",
                "schema": EmptyInput,
            },
            "Get Announcement Detail": {
                "function": self.manager.get_announcement_detail,
                "description": ("Retrieve details of a specific announcement. "
                                "Provide 'announcement_title' (str)."),
                "schema": GetAnnouncementDetailInput,
            },
            "Other": {
                "function": self.manager.other,
                "description": ("Other intent no in the list. "
                                "Provide a 'query' string"),
                "schema": OtherInput,
            },
        }

        # Build tool integrations using StructuredTool.
        self.tools = [
            StructuredTool(
                name=name,
                func=info["function"],
                description=info["description"],
                args_schema=info["schema"]
            ) for name, info in self.tool_info.items()
        ]

        # Build separate mappings if needed.
        self.tool_mapping = {name: info["function"] for name, info in self.tool_info.items()}

        # Prepare the tool schemas string with escaped curly braces.
        tool_schemas = "\n".join([
            f"{name} expects: " + json.dumps(schema.model_json_schema(), indent=2).replace("{", "{{").replace("}", "}}")
            for name, info in self.tool_info.items()
            for schema in [info["schema"]]
        ])

        # Set up the multi-intent prompt template.
        # Use double curly braces for "input" and "chat_history" to leave them for later substitution.
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful Canvas Q&A assistant.

Analyze the user's query and detect all relevant intents related to Canvas tasks (e.g. lecture slides, timetables, assignments, or announcements). For each intent, output a JSON object with exactly two keys:
  - "action": one of the following tool names: {tool_names}.
  - "action_input": a dictionary representing the tool's input.

Follow the schema for each tool as follows:
{tool_schemas}

If no tool call is needed, simply output your answer as plain text.

After you have generated the required tool calls and they are executed, you will receive the tool responses.
Finally, synthesize a clear and well-organized final answer that integrates all the responses with your analysis.

Question: {{input}}
{{chat_history}}
""".format(
                tool_names=", ".join(self.tool_info.keys()),
                tool_schemas=tool_schemas
            )
        )

        # Initialize the OpenAI Chat Model & Conversation Memory.
        self.llm = ChatOpenAI(model_name="gpt-4",
                              temperature=0,
                              openai_api_key=self.openai_api_key)
        self.memory = ConversationSummaryMemory(llm=self.llm, return_messages=True)

    @staticmethod
    def _load_key(filepath: str, key_name: str) -> str:
        try:
            with open(filepath, "r") as file:
                return file.read().strip()
        except Exception as e:
            print(f"Error reading {key_name} key from {filepath}: {e}")
            return ""

    @staticmethod
    def get_best_tool_mapping(action: str, mapping: dict, threshold: float = 0.8):
        best_match = None
        best_score = 0.0
        for key in mapping:
            score = difflib.SequenceMatcher(None, action.lower(), key.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = key
        if best_score >= threshold:
            return best_match, best_score
        return None, best_score

    def detect_intents(self, response_text: str) -> list:
        """
        Parse the LLM response for intents. First try valid JSON, then Python literals, then split on blank lines.
        Returns a list of intent dicts.
        """
        text = response_text.strip()
        # 1) Try JSON array/object
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                print(f"[DEBUG] Parsed entire response as JSON list with {len(parsed)} intents")
                return parsed
            if isinstance(parsed, dict):
                print("[DEBUG] Parsed entire response as JSON object")
                return [parsed]
        except Exception:
            pass

        # 2) Try Python literal parsing
        try:
            import ast
            parsed_py = ast.literal_eval(text)
            if isinstance(parsed_py, list):
                print(f"[DEBUG] Parsed entire response as Python literal list with {len(parsed_py)} intents")
                return parsed_py
            if isinstance(parsed_py, dict):
                print("[DEBUG] Parsed entire response as Python literal dict")
                return [parsed_py]
        except Exception:
            pass

        # 3) Fallback: split on blank lines and parse each chunk as JSON
        # Normalize CRLF to LF
        text_normalized = text.replace('\r\n', '\n')
        raw_chunks = [
            c.strip()
            for c in re.split(r'\n{2,}', text_normalized)
            if c.strip()
        ]
        print(f"[DEBUG] Found {len(raw_chunks)} raw chunks after blank-line split")

        intents = []
        for idx, chunk in enumerate(raw_chunks, start=1):
            try:
                intent = json.loads(chunk)
                print(f"[DEBUG] Chunk {idx} → parsed intent via JSON: {intent}")
                intents.append(intent)
            except json.JSONDecodeError:
                try:
                    import ast
                    intent = ast.literal_eval(chunk)
                    print(f"[DEBUG] Chunk {idx} → parsed intent via Python literal: {intent}")
                    intents.append(intent)
                except Exception as e:
                    print(f"[DEBUG] Chunk {idx} failed to parse: {e}")
        return intents

    def process_agent_response(self, response_text: str, original_query: str) -> str:
        """
        Process the agent's raw LLM output:
          1. Detect intents.
          2. (Optionally, you could further validate or fill missing parameters here using structured input.)
          3. Synthesize a final answer that integrates the original query with tool responses.
        """
        intents = self.detect_intents(response_text)
        # Here we simply assume the LLM output adheres to the schemas.
        # In a more advanced setup, you could use the Pydantic models to validate each action_input.
        tool_results = {}
        for intent in intents:
            mapped_tool, _ = CanvasAgent.get_best_tool_mapping(intent.get("action", ""), self.tool_mapping)
            if mapped_tool:
                # For structured tools, validation is automatically handled when calling the schema.
                try:
                    # Convert the action_input into a Pydantic model to validate.
                    model_cls = self.tool_info[mapped_tool]["schema"]
                    validated_params = model_cls(**intent.get("action_input", {})).model_dump()
                    result = self.tool_mapping[mapped_tool](**validated_params)
                    tool_results[mapped_tool] = result
                except Exception as e:
                    tool_results[mapped_tool] = f"Error executing tool '{mapped_tool}': {e}"
            else:
                tool_results[intent.get("action", "")] = "Tool not found."

        print("Tool Results.\n")
        print(tool_results)
        synthesis_prompt = (
            "You are a helpful Canvas Q&A assistant.\n\n"
            f"The original query was:\n{original_query}\n\n"
            "The following tool responses (or error messages) were obtained:\n"
            f"{json.dumps(tool_results, indent=2)}\n\n"
            "Please synthesize a final, well-organized answer that integrates the above information. "
            "If any tool requires additional information, indicate which details are needed."
        )
        final_response = self.llm.invoke([{"role": "user", "content": synthesis_prompt}])
        final_text = final_response.content.strip() if hasattr(final_response, "content") else str(final_response).strip()
        return final_text

    def ask(self, query: str) -> str:
        """
        Processes the entire query:
          - Retrieves conversation history.
          - Builds and sends the multi-intent prompt.
          - Saves conversation history.
          - Processes the LLM response via structured tool integration.
          - Returns the final synthesized answer.
        """
        chat_history = self.memory.load_memory_variables({})
        full_prompt = self.prompt.format(
            tools="\n".join([tool.description for tool in self.tools]),
            input=query,
            chat_history=chat_history
        )
        print("Debug: Full prompt sent to agent:\n", full_prompt)
        response = self.llm.invoke([{"role": "user", "content": full_prompt}])
        response_text = response.content if hasattr(response, "content") else str(response)
        print("Debug: Raw agent response:\n", response_text)
        self.memory.save_context({"input": query}, {"output": response_text})
        final_result = self.process_agent_response(response_text, query)
        return final_result


if __name__ == "__main__":
    agent = CanvasAgent()
    print("Welcome to the Canvas Q&A Agent!")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        answer = agent.ask(user_query)
        print(f"\nAgent: {answer}")
