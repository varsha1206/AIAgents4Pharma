#!/usr/bin/env python3

"""
Tool for asking a question about the simulation results.
"""

from typing import Type, Optional
import streamlit as st
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

class AskQuestionInput(BaseModel):
    """
    Input schema for the AskQuestion tool.
    """
    question: str = Field(description="question about the simulation results")
    st_session_key: str = Field(description="Streamlit session key")

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class AskQuestionTool(BaseTool):
    """
    Tool for calculating the product of two numbers.
    """
    name: str = "ask_question"
    description: str = "A tool to ask question about the simulation results."
    args_schema: Type[BaseModel] = AskQuestionInput
    return_direct: bool = True

    def _run(self,
             question: str,
             st_session_key: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Run the tool.

        Args:
            question (str): The question to ask about the simulation results.
            st_session_key (str): The Streamlit session key.
            run_manager (Optional[CallbackManagerForToolRun]): The CallbackManagerForToolRun object.

        Returns:
            str: The answer to the question.
        """
        if st_session_key not in st.session_state:
            return f"Session key {st_session_key} not found in Streamlit session state."
        model_object = st.session_state[st_session_key]
        # check if model_object is None
        if model_object is None:
            return "Please run the simulation first before asking a question."
        if model_object.simulation_results is None:
            return "Please run the simulation first before asking a question."
        df = model_object.simulation_results
        st.text(f"Simulation Results of the model {model_object.model_id}")
        st.dataframe(df, use_container_width = False, width = 650)
        # Check if run_manager's metadata has the key 'prompt_content'
        if 'prompt' in run_manager.metadata:
            prompt_content = run_manager.metadata['prompt']
        else:
            prompt_content = None
        # Create a pandas dataframe agent with OpenAI
        df_agent = create_pandas_dataframe_agent(ChatOpenAI(model="gpt-3.5-turbo"),
                                                  allow_dangerous_code=True,
                                                  agent_type=AgentType.OPENAI_FUNCTIONS,
                                                  df=df,
                                                  prefix=prompt_content)
        llm_result = df_agent.invoke(question)
        return llm_result["output"]

    def run(self,
            question: str,
            st_session_key: str,
            run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Run the tool.

        Args:
            question (str): The question to ask about the simulation results.
            st_session_key (str): The Streamlit session key.
            run_manager (Optional[CallbackManagerForToolRun]): The CallbackManagerForToolRun object.

        Returns:
            str: The answer to the question.
        """
        return self._run(question=question, st_session_key=st_session_key, run_manager=run_manager)

    def get_metadata(self):
        """
        Get metadata for the tool.

        Returns:
            dict: The metadata for the tool.
        """
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema.schema(),
            "return_direct": self.return_direct,
        }
