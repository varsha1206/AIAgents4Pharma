#!/usr/bin/env python3

"""
Tool for asking a question about the model description.
"""

from typing import Type, Optional
from pydantic import BaseModel, Field
import streamlit as st
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class ModelDescriptionInput(BaseModel):
    """
    Input schema for the ModelDescription tool.
    """
    question: str = Field(description="question about the model description")
    st_session_key: str = Field(description="Streamlit session key")


# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class ModelDescriptionTool(BaseTool):
    """
    Tool for returning the description of the specified BioModel.
    """
    name: str = "model_description"
    description: str = "A tool to ask about the description of the model."
    args_schema: Type[BaseModel] = ModelDescriptionInput
    return_direct: bool = True

    def _run(self,
             question: str,
             st_session_key: str,
             run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Run the tool.

        Args:
            question (str): The question to ask about the model description.
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
        description = model_object.description
        if description is None:
            return "No description found for the model."
        # Append the BioModel ID of the model to the description
        description = f"{description} (BioModel ID: {model_object.model_id})"
        llm = ChatOpenAI(model="gpt-4")
        # Check if run_manager's metadata has the key 'prompt_content'
        if 'prompt' in run_manager.metadata:
            prompt_content = run_manager.metadata['prompt']
        else:
            prompt_content = '''Given the description {description},
                            answer the question {question}.'''
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", prompt_content),
             ("user", "{description} {question}")]
        )
        parser = StrOutputParser()
        chain = prompt_template | llm | parser
        # return st.html(description)
        return chain.invoke({"description": description,
                             "question": question})

    def run(self,
            question: str,
            st_session_key: str,
            run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Run the tool.

        Args:
            question (str): The question to ask about the model description.
            st_session_key (str): The Streamlit session key.
            run_manager (Optional[CallbackManagerForToolRun]): The CallbackManagerForToolRun object.
        
        Returns:
            str: The answer to the question
        """
        return self._run(question=question,
                         st_session_key=st_session_key,
                         run_manager=run_manager)

    def get_metadata(self):
        """
        Get metadata for the tool.

        Returns:
            dict: The metadata for the tool.
        """
        return {
            "name": self.name,
            "description": self.description,
            "return_direct": self.return_direct,
        }
