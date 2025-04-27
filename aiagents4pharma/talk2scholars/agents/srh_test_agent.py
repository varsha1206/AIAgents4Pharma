# /usr/bin/env python3

"""
Agent for interacting with srh_test
"""

import logging
from typing import Any, Dict
import hydra
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.memory import MemorySaver
from ..state.state_talk2scholars import Talk2Scholars
from ..tools.srh_test.maths import basic_math as bmath

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_app(uniq_id, llm_model: BaseChatModel):
    """
    Initializes and returns the LangGraph application for the SRH_TEST agent.

    This function sets up the Srh_Test agent, which integrates one tool to multiply instead of add.
    The agent follows the ReAct pattern for structured interaction.

    Args:
        uniq_id (str): Unique identifier for the current conversation session.
        llm_model (BaseChatModel, optional): The language model to be used by the agent.
            Defaults to `ChatOpenAI(model="gpt-4o-mini", temperature=0)`.

    Returns:
        StateGraph: A compiled LangGraph application that enables the srh_agent agent to multiply two numbers.

    Example:
        >>> app = get_app("thread_123")
        >>> result = app.invoke(initial_state)
    """

    # def agent_srhTest_node(state: Talk2Scholars) -> Command[Literal["supervisor"]]:
    def agent_srhTest_node(state: Talk2Scholars) -> Dict[str, Any]:
        """
        Performs the multiplication of two numbers insteading of adding them.

        This function calls the language model using the configured `ReAct` agent to analyze
        the state and generate an appropriate response. The function then returns control
        to the main supervisor.

        Args:
            state (Talk2Scholars): The current conversation state, including messages exchanged
                and any previously retrieved research papers.

        Returns:
            Dict[str, Any]: A dictionary containing the updated conversation state.

        Example:
            >>> result = agent_srhTest_node(current_state)
            >>> papers = result.get("papers", [])
        """
        logger.log(logging.INFO, "Creating Agent_srhTest node with thread_id %s", uniq_id)
        result = model.invoke(state, {"configurable": {"thread_id": uniq_id}})

        return result

    logger.log(logging.INFO, "thread_id, llm_model: %s, %s", uniq_id, llm_model)

    # Load hydra configuration
    logger.log(logging.INFO, "Load Hydra configuration for Talk2Scholars srh_test agent.")
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            config_name="config", overrides=["agents/talk2scholars/srh_test_agent=default"]
        )
        cfg = cfg.agents.talk2scholars.srh_test_agent
        logger.log(logging.INFO, "Loaded configuration for srhTest agent")

    # Define the tools
    tools = ToolNode(
        [
            bmath
        ]
    )

    # Define the model
    logger.log(logging.INFO, "Using OpenAI model %s", llm_model)

    # Create the agent
    model = create_react_agent(
        llm_model,
        tools=tools,
        state_schema=Talk2Scholars,
        prompt=cfg.srh_test_agent,
        checkpointer=MemorySaver(),
    )

    workflow = StateGraph(Talk2Scholars)
    workflow.add_node("agent_srhTest", agent_srhTest_node)
    workflow.add_edge(START, "agent_srhTest")

    # Initialize memory to persist state between graph runs
    checkpointer = MemorySaver()

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable.
    # Note that we're (optionally) passing the memory when compiling the graph
    app = workflow.compile(checkpointer=checkpointer, name="agent_srhTest")
    logger.log(
        logging.INFO,
        "Compiled the graph with thread_id %s and llm_model %s",
        uniq_id,
        llm_model,
    )

    return app
