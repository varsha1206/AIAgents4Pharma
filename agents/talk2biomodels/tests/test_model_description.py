'''
Test cases for model_description.py
'''

import pytest
import streamlit as st
from langchain_core.callbacks import CallbackManagerForToolRun
from ..tools.model_description import ModelDescriptionTool, ModelDescriptionInput
from ..models.copasimodel import CopasiModel

@pytest.fixture(name="model_description_tool")
def model_description_tool_fixture():
    '''
    Fixture for creating an instance of ModelDescriptionTool.
    '''
    return ModelDescriptionTool()

def test_run_with_missing_session_key(model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with a missing session key.
    '''
    input_data = ModelDescriptionInput(
        question="What is the description of the model?",
        st_session_key="missing_key"
    )
    result = model_description_tool.call_run(**input_data.model_dump())
    assert result == "Session key missing_key not found in Streamlit session state."

def test_run_with_valid_key_but_model_data(model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with a valid session key.
    '''
    input_data = ModelDescriptionInput(
        question="What is the description of the model?",
        st_session_key="test_key"
    )
    st.session_state["test_key"] = None
    result = model_description_tool.call_run(**input_data.model_dump())
    assert result == "Please run the simulation first before asking a question."

def test_run_with_valid_key_and_model_data_but_no_description(model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with a valid session key and model data.
    '''
    input_data = ModelDescriptionInput(
        question="What is the description of the model?",
        st_session_key="test_key"
    )
    st.session_state["test_key"] = CopasiModel(model_id=64)
    st.session_state["test_key"].description = None
    result = model_description_tool.call_run(**input_data.model_dump())
    assert result == "No description found for the model."

def test_run_with_valid_key_model_data_description(model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with a valid session key and model data.
    '''
    input_data = ModelDescriptionInput(
        question="What is the description of the model?",
        st_session_key="test_key"
    )
    st.session_state["test_key"] = CopasiModel(model_id=64)
    run_manager = CallbackManagerForToolRun(run_id=1, handlers=[], inheritable_handlers=False)
    result = model_description_tool.call_run(**input_data.model_dump(), run_manager=run_manager)
    assert result is not None
    run_manager = CallbackManagerForToolRun(run_id=1,
                                            handlers=[],
                                            inheritable_handlers=False,
                                            metadata={"prompt": "Answer the question carefully."})
    result = model_description_tool.call_run(**input_data.model_dump(), run_manager=run_manager)
    assert result is not None

def test_get_metadata(model_description_tool):
    '''
    Test the get_metadata method of the ModelDescriptionTool class.
    '''
    metadata = model_description_tool.get_metadata()
    assert metadata["name"] == "model_description"
    assert metadata["description"] == "A tool to ask about the description of the model."
    assert metadata["return_direct"] == model_description_tool.return_direct
