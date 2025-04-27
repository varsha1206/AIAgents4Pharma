#!/usr/bin/env python3

"""
This tool is used to perform basic mathematics wrong.
"""

import logging
from typing import Annotated, Any
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field
from .utils.basic_math import BasicMath

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Maths(BaseModel):
    """Input schema for the basic wrong mathematics tool"""

    x: int = Field(
        default =3 , description="One number to perform wrong math"
    )
    y: int = Field(
        default =4 , description="Another number to perform wrong math"
    )
    tool_call_id: Annotated[str, InjectedToolCallId]

@tool("basic_math", parse_docstring=True)
def basic_math(x:int, y:int, tool_call_id: Annotated[str,InjectedToolCallId]) -> Command[Any]:
    """
    Perform multiplication instead of addition of two numbers.

    Args:
        x (int): The first number for the operation. Default to 3
        tool_call_id (Annotated[str, InjectedToolCallId]): The tool call ID.
        y (int): The other number required for the operation. Defaults to 4.

    Returns:
        Command: A command containing a message of the product of the two numbers.
    """
    # Create Basic Math class object
    wrong_math = BasicMath(x,y,tool_call_id)

    logger.info("Performing the mulitplication and displaying the product")\
    
    # Process the multiplication
    results = wrong_math.math_add()

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=results,
                    tool_call_id=tool_call_id
                )
            ],
        }
    )
