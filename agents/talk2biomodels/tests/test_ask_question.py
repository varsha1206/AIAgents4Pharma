'''
Test cases for ask_question.py
'''

import pytest
import streamlit as st
from langchain_core.callbacks import CallbackManagerForToolRun
from ..tools.ask_question import AskQuestionTool, AskQuestionInput
from ..models.copasimodel import CopasiModel

@pytest.fixture(name="ask_question_tool")
def ask_question_tool_fixture():
    '''
    Fixture for creating an instance of AskQuestionTool.
    '''
    return AskQuestionTool()

def test_run_with_missing_session_key(ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class with a missing session key.
    '''
    input_data = AskQuestionInput(
        question="What is the concentration of species1 at time 10?",
        st_session_key="missing_key"
    )
    result = ask_question_tool.run(**input_data.model_dump())
    assert result == "Session key missing_key not found in Streamlit session state."

def test_run_with_valid_key_but_model_data(ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class with a valid session key.
    '''
    input_data = AskQuestionInput(
        question="What is the concentration of species1 at time 10?",
        st_session_key="test_key"
    )
    st.session_state["test_key"] = None
    result = ask_question_tool.run(**input_data.model_dump())
    assert result == "Please run the simulation first before asking a question."

def test_run_with_valid_key_and_model_data_but_no_simulation(ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class with a valid session key and model data.
    '''
    input_data = AskQuestionInput(
        question="What is the concentration of species1 at time 10?",
        st_session_key="test_key"
    )
    st.session_state["test_key"] = CopasiModel(model_id=64)
    result = ask_question_tool.run(**input_data.model_dump())
    assert result == "Please run the simulation first before asking a question."

def test_run_with_valid_key_model_data_simulation(ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class 
    with a valid session key, model data, and simulation.
    '''
    input_data = AskQuestionInput(
        question="What is the concentration of species1 at time 10?",
        st_session_key="test_key"
    )
    st.session_state["test_key"] = CopasiModel(model_id=64)
    st.session_state["test_key"].simulate(duration=2, interval=2)
    run_manager = CallbackManagerForToolRun(run_id=1, handlers=[], inheritable_handlers=False)
    ask_question_tool = AskQuestionTool()
    result = ask_question_tool.run(**input_data.model_dump(), run_manager=run_manager)
    assert result is not None
    run_manager = CallbackManagerForToolRun(run_id=1,
                                            handlers=[],
                                            inheritable_handlers=False,
                                            metadata={"prompt": "Answer the question carefully."})
    result = ask_question_tool.run(**input_data.model_dump(), run_manager=run_manager)
    assert result is not None

def test_get_metadata(ask_question_tool):
    '''
    Test the get_metadata method of the AskQuestionTool class.
    '''
    metadata = ask_question_tool.get_metadata()
    assert metadata["name"] == "ask_question"
    assert metadata["description"] == "A tool to ask question about the simulation results."
    assert metadata["return_direct"] == ask_question_tool.return_direct
