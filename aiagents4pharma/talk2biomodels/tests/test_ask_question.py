'''
Test cases for ask_question.py
'''

import pytest
import streamlit as st
from langchain_core.callbacks import CallbackManagerForToolRun
from ..tools.ask_question import AskQuestionTool, AskQuestionInput, ModelData
from ..models.basico_model import BasicoModel

@pytest.fixture(name="ask_question_tool")
def ask_question_tool_fixture():
    '''
    Fixture for creating an instance of AskQuestionTool.
    '''
    return AskQuestionTool()

@pytest.fixture(name="input_data", scope="module")
def input_data_fixture():
    '''
    Fixture for creating an instance of AskQuestionInput.
    '''
    return AskQuestionInput(question="What is the concentration of Pyruvate at time 5?",
                            sys_bio_model=ModelData(modelid=64),
                            st_session_key="test_key"
                            )

def test_run_with_sbml_file(input_data, ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class with a valid session key and model data.
    '''
    input_data.sys_bio_model = ModelData(sbml_file_path="./BIOMD0000000064_url.xml")
    result = ask_question_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert isinstance(result, str)

def test_run_manager(input_data, ask_question_tool):
    '''
    Test the run manager of the AskQuestionTool class.
    '''
    run_manager = CallbackManagerForToolRun(run_id=1, handlers=[], inheritable_handlers=False)
    result = ask_question_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key,
                                        run_manager=run_manager)
    assert isinstance(result, str)
    run_manager = CallbackManagerForToolRun(run_id=1,
                                            handlers=[],
                                            inheritable_handlers=False,
                                            metadata={"prompt": "Answer the question carefully."})
    result = ask_question_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key,
                                        run_manager=run_manager)
    assert isinstance(result, str)

def test_run_with_no_model_data_at_all(input_data, ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class with a valid session key and model data.
    '''
    result = ask_question_tool.call_run(question=input_data.question,
                                        st_session_key=input_data.st_session_key)
    assert isinstance(result, str)

def test_run_with_session_key(input_data, ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class with a missing session key.
    '''
    input_data.sys_bio_model = ModelData(modelid=64)
    result = ask_question_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert isinstance(result, str)

def test_run_with_none_key(input_data, ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class with a None session key.
    '''
    input_data.st_session_key = None
    result = ask_question_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert isinstance(result, str)
    input_data.sys_bio_model = ModelData()
    result = ask_question_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    # No model data or object in the streeamlit key
    assert result == "Please provide a valid model object or \
                    Streamlit session key that contains the model object."
    input_data.st_session_key = "test_key"
    # delete the session key form the session state
    st.session_state.pop(input_data.st_session_key, None)
    result = ask_question_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert result == f"Session key {input_data.st_session_key} " \
        "not found in Streamlit session state."

def test_run_with_a_simulated_model(input_data, ask_question_tool):
    '''
    Test the _run method of the AskQuestionTool class with a valid session key and model data.
    '''
    model = BasicoModel(model_id=64)
    model.simulate(duration=2, interval=2)
    input_data.sys_bio_model = ModelData(model_object=model)
    result = ask_question_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert isinstance(result, str)

def test_get_metadata(ask_question_tool):
    '''
    Test the get_metadata method of the AskQuestionTool class.
    '''
    metadata = ask_question_tool.get_metadata()
    assert metadata["name"] == "ask_question"
    assert metadata["description"] == "A tool to ask question about the simulation results."
