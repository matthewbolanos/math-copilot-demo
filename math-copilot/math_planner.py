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


@tool
def chat(
    intent: str,
    deployment_name: str,
    connection: AzureOpenAIConnection,
):

    # if the intent is "perform_math" then we need to use the math plugin
    if intent != "NoMathEquation":
        # Create kernel for math plugin
        math_kernel = sk.Kernel(log=sk.NullLogger())

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
        math_kernel.import_skill(Math())

        # Create the planner to solve the math problem
        planner = SequentialPlanner(kernel=math_kernel)

        # Create a plan to solve the math problem
        ask = "Solve this math problem: " + intent
        plan = asyncio.run(planner.create_plan_async(ask))

        # Get the result of the math problem
        math_answer = asyncio.run(math_kernel.run_async(plan)).result

        for index, step in enumerate(plan._steps):
            print("Function: " + step.skill_name + "." + step._function.name)
            print("Input vars: " + str(step.parameters.variables))
            print("Output vars: " + str(step._outputs))
        print("Result: " + str(math_answer))

        # Add the answer of the math problem to the context
        return "The bot should respond with this answer: " + math_answer
    
    return ""
