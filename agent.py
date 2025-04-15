import json
import difflib
from functools import partial

from langchain.tools import Tool
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from canvas import CanvasManager


class CanvasAgent:
    """
    A Canvas Q&A agent that detects multiple intents and synthesizes a final answer
    by leveraging various tools related to Canvas tasks (lecture slides, timetables,
    assignments, announcements) using OpenAI's language models.

    The tool-calling process is divided into four steps:
      1. Detect intents from the LLM response.
      2. Compare each intent against available tools.
      3. Validate the input parameters for each tool (check for presence and type, and convert when possible).
      4. Clean up and synthesize the final answer by integrating the original query with the tool responses.
    """

    def __init__(self,
                 canvas_api_key_path: str = "keys/canvas.txt",
                 openai_api_key_path: str = "keys/openai.txt",
                 api_url: str = "https://canvas.nus.edu.sg/"):
        # Load API keys.
        self.canvas_api_key = self._load_key(canvas_api_key_path, "Canvas API")
        self.api_url = api_url
        self.openai_api_key = self._load_key(openai_api_key_path, "OpenAI API")

        # Initialize the Canvas Manager.
        self.manager = CanvasManager(self.api_url, self.canvas_api_key)

        # Set up the tools.
        self.tools = [
            Tool(
                name="Retrieve Lecture Slides",
                func=partial(self.manager.retrieve_lecture_slides_by_topic),
                description=("Retrieve lecture slides related to a query. "
                             "Inputs: query (str), subjects (optional, list of str).")
            ),
            Tool(
                name="Get Timetable Or Exam Schedule",
                func=partial(self.manager.get_timetable_or_exam_date),
                description=("Retrieve the timetable for a course. "
                             "Inputs: full_time (bool), intake_year (int), course (str).")
            ),
            Tool(
                name="List Upcoming Assignments",
                func=partial(self.manager.list_upcoming_assignments),
                description="List upcoming assignments. (No input parameters required.)"
            ),
            Tool(
                name="Get Assignment Detail",
                func=partial(self.manager.get_assignment_detail),
                description=("Retrieve details of a specific assignment. "
                             "Inputs: assignment_name (str).")
            ),
            Tool(
                name="List Announcements",
                func=partial(self.manager.list_announcements),
                description="List recent announcements. (No input parameters required.)"
            ),
            Tool(
                name="Get Announcement Detail",
                func=partial(self.manager.get_announcement_detail),
                description=("Retrieve details of a specific announcement. "
                             "Inputs: announcement_title (str).")
            )
        ]

        # Set up the multi-intent prompt template.
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful Canvas Q&A assistant.

Analyze the user's query and detect all relevant intents related to Canvas tasks (e.g. lecture slides, timetables, assignments, or announcements). For each intent, output a JSON object with exactly two keys:
  - "action": one of the following tool names: "Retrieve Lecture Slides", "Get Timetable Or Exam Schedule", "List Upcoming Assignments", "Get Assignment Detail", "List Announcements", "Get Announcement Detail".
  - "action_input": a dictionary of inputs for the tool.

If no tool call is needed, simply output your answer as plain text.

After you have generated the required tool calls and they are executed, you will receive the tool responses.
Finally, synthesize a clear and well-organized final answer that integrates all the responses with your analysis.

Question: {input}
{chat_history}
"""
        )

        # Initialize the OpenAI Chat Model & Conversation Memory.
        self.llm = ChatOpenAI(model_name="gpt-4",
                              temperature=0,
                              openai_api_key=self.openai_api_key)
        self.memory = ConversationSummaryMemory(llm=self.llm, return_messages=True)

        # Mapping between tool names and CanvasManager functions.
        self.tool_mapping = {
            "Retrieve Lecture Slides": self.manager.retrieve_lecture_slides_by_topic,
            "Get Timetable Or Exam Schedule": self.manager.get_timetable_or_exam_date,
            "List Upcoming Assignments": self.manager.list_upcoming_assignments,
            "Get Assignment Detail": self.manager.get_assignment_detail,
            "List Announcements": self.manager.list_announcements,
            "Get Announcement Detail": self.manager.get_announcement_detail,
        }

        # Define the required parameters for each tool.
        self.tool_required_params = {
            "Retrieve Lecture Slides": ["query"],
            "Get Timetable Or Exam Schedule": ["full_time", "intake_year", "course"],
            "List Upcoming Assignments": [],  # No parameters required.
            "Get Assignment Detail": ["assignment_name"],
            "List Announcements": [],  # No parameters required.
            "Get Announcement Detail": ["announcement_title"],
        }

        # Define expected types for parameters.
        self.tool_param_types = {
            "Retrieve Lecture Slides": {"query": str, "subjects": list},
            "Get Timetable Or Exam Schedule": {"full_time": bool, "intake_year": int, "course": str},
            "List Upcoming Assignments": {},
            "Get Assignment Detail": {"assignment_name": str},
            "List Announcements": {},
            "Get Announcement Detail": {"announcement_title": str},
        }

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

    @staticmethod
    def convert_param(value, expected_type):
        """
        Attempt to convert 'value' to the expected_type. If conversion fails,
        return (None, error message). Otherwise, return (converted_value, None).
        """
        try:
            if expected_type is bool:
                if isinstance(value, bool):
                    return value, None
                elif isinstance(value, str):
                    lower_val = value.lower()
                    if lower_val in ['true', 'yes', '1']:
                        return True, None
                    elif lower_val in ['false', 'no', '0']:
                        return False, None
                    else:
                        raise ValueError("Invalid boolean string")
                else:
                    return bool(value), None
            elif expected_type is int:
                if isinstance(value, int):
                    return value, None
                else:
                    return int(value), None
            elif expected_type is list:
                if isinstance(value, list):
                    return value, None
                elif isinstance(value, str):
                    # Assume comma-separated list
                    return [item.strip() for item in value.split(',') if item.strip()], None
                else:
                    raise ValueError("Cannot convert to list")
            else:
                if isinstance(value, expected_type):
                    return value, None
                return expected_type(value), None
        except Exception as e:
            return None, f"Expected type {expected_type.__name__}, got {type(value).__name__}"

    def detect_intents(self, response_text: str) -> list:
        """
        Step 1: Remove extraneous markdown and parse the JSON output from the LLM.
        Returns a list of intents (each a dict with 'action' and 'action_input').
        """
        if response_text.startswith("```"):
            lines = response_text.splitlines()
            json_lines = [line for line in lines if not line.strip().startswith("```")]
            response_text = "\n".join(json_lines).strip()
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict):
                return [parsed]
            else:
                return []
        except json.JSONDecodeError:
            return []  # If output isn’t valid JSON, treat it as final answer elsewhere.

    def validate_intent(self, intent: dict, original_query: str) -> (str, dict, list):
        """
        Steps 2 & 3: Compare the intent with available tools via fuzzy matching.
        Then check that all required parameters exist and are of the proper type.
        Returns a tuple: (mapped_tool_name, validated_parameters, missing_or_invalid_parameters)
        """
        action_name = intent.get("action", "")
        action_input = intent.get("action_input", {}).copy()

        # For "Retrieve Lecture Slides", enforce using the original query.
        if action_name == "Retrieve Lecture Slides":
            action_input["query"] = original_query

        mapped_tool, _ = CanvasAgent.get_best_tool_mapping(action_name, self.tool_mapping)
        if not mapped_tool:
            return action_name, action_input, ["Tool not found"]

        # Check for missing required parameters.
        required_params = self.tool_required_params.get(mapped_tool, [])
        missing = [param for param in required_params if param not in action_input or not action_input[param]]

        # Now validate types for parameters if provided.
        type_errors = []
        expected_types = self.tool_param_types.get(mapped_tool, {})
        for param, expected_type in expected_types.items():
            if param in action_input and action_input[param]:
                converted_value, error = CanvasAgent.convert_param(action_input[param], expected_type)
                if error:
                    type_errors.append(f"{param}: {error}")
                else:
                    # Replace with the converted value.
                    action_input[param] = converted_value

        missing_or_invalid = missing + type_errors
        return mapped_tool, action_input, missing_or_invalid

    def execute_tool(self, tool_name: str, params: dict):
        """
        Execute the tool function using the provided parameters.
        """
        tool_func = self.tool_mapping.get(tool_name)
        if tool_func:
            return tool_func(**params)
        else:
            raise ValueError("No valid tool found.")

    def process_intents(self, intents: list, original_query: str) -> dict:
        """
        For each detected intent, validate and either execute the tool or note missing/invalid parameters.
        Returns a dictionary mapping tool names (or intent actions) to their result or message.
        """
        tool_results = {}
        for intent in intents:
            mapped_tool, validated_params, missing = self.validate_intent(intent, original_query)
            if missing:
                message = f"Missing or invalid parameter(s): {', '.join(missing)}."
                tool_results[mapped_tool if mapped_tool != "" else intent.get("action", "")] = message
                print(f"Debug: For action '{intent.get('action')}', missing/invalid parameters: {missing}")
            else:
                try:
                    result = self.execute_tool(mapped_tool, validated_params)
                    tool_results[mapped_tool] = result
                except Exception as e:
                    tool_results[mapped_tool] = f"Error while executing tool '{mapped_tool}': {e}"
        return tool_results

    def process_agent_response(self, response_text: str, original_query: str) -> str:
        """
        Complete processing by:
          1. Detecting intents.
          2. Validating and executing (or reporting missing/invalid parameters).
          3. Synthesizing a final answer from the original query and tool responses.
        """
        # Step 1: Detect intents.
        intents = self.detect_intents(response_text)
        if not intents:
            return response_text

        # Step 2 & 3: Validate and process each intent.
        tool_results = self.process_intents(intents, original_query)

        # Step 4: Synthesize a final answer.
        synthesis_prompt = (
            "You are a helpful Canvas Q&A assistant.\n\n"
            f"The original query was:\n{original_query}\n\n"
            "The following tool responses (or missing/invalid parameter messages) were obtained:\n"
            f"{json.dumps(tool_results, indent=2)}\n\n"
            "Please synthesize a final, well-organized answer that integrates the above information. "
            "If any tool requires additional information, indicate which details are needed."
        )
        final_response = self.llm.invoke([{"role": "user", "content": synthesis_prompt}])
        final_text = final_response.content.strip() if hasattr(final_response, "content") else str(
            final_response).strip()
        return final_text

    def ask(self, query: str) -> str:
        """
        Processes the entire query:
          - Retrieves conversation history.
          - Builds and sends the multi-intent prompt.
          - Saves conversation history.
          - Processes the LLM response through the multi-step tool calling process.
          - Returns the final answer.
        """
        # Retrieve conversation history.
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

        # Save the conversation.
        self.memory.save_context({"input": query}, {"output": response_text})

        # Process the tool calls and synthesize the final answer.
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
