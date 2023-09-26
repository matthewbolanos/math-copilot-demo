import asyncio

from plugins.math_plugin.Math import Math

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.planning.sequential_planner import SequentialPlanner
from semantic_kernel.planning.sequential_planner.sequential_planner_config import SequentialPlannerConfig
from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)

from promptflow import tool
from promptflow.connections import AzureOpenAIConnection


import logging

class PlannerLogger(logging.Handler):
    def __init__(self, *args, **kwargs):
        super(PlannerLogger, self).__init__(*args, **kwargs)
        self.log_records = []

    def emit(self, record):
        self.log_records.append(self.format(record))


@tool
def chat(
    intent: str,
    math_problem: str,
    deployment_name: str,
    connection: AzureOpenAIConnection,
):
    print("Intent: " + intent)
    print(intent == "PerformMath")
    if intent == "PerformMath":
        logger = logging.getLogger(__name__)
        handler = PlannerLogger()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # Create kernel for math plugin
        math_kernel = sk.Kernel(log=logger)

        # Add the chat service
        math_kernel.add_chat_service(
            "chat_completion",
            AzureChatCompletion(
                deployment_name,
                connection.api_base,
                connection.api_key,
            ),
        )

        # Import the math plugin
        math_kernel.import_skill(Math(), "Math")

        # Create the planner to solve the math problem
        planner = SequentialPlanner(kernel=math_kernel)

        # Create a plan to solve the math problem
        request = "Solve this math problem: " + math_problem
        request += "\n\n"
        request += "- When using the functions, do not pass any units or symbols like $; only send valid floats.\n"
        request += "- Only use the necessary functions needed to answer the word problem; it's ok to only use one function.\n"

        plan = asyncio.run(planner.create_plan_async(request))

        # Get the result of the math problem
        math_answer = asyncio.run(math_kernel.run_async(plan)).result

        # Print the debug log
        print("Debug log: " + "\n\n".join(handler.log_records))

        # Print the raw plan
        print("Plan JSON: " + plan.json())

        # Print summary of the plan
        for index, step in enumerate(plan._steps):
            print("Function: " + step.skill_name + "." + step._function.name)
            print("Input vars: " + str(step.parameters.variables))
            print("Output vars: " + str(step._outputs))

        # Print the result of the math problem
        print("Result: " + str(math_answer))

        # Add the answer of the math problem to the context
        return "The bot should respond with this answer: " + math_answer
    
    return ""
