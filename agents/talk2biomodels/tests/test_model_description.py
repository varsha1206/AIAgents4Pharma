'''
Test cases for model_description.py
'''

import pytest
import streamlit as st
from langchain_core.callbacks import CallbackManagerForToolRun
from ..tools.model_description import ModelDescriptionInput, ModelDescriptionTool, ModelData
from ..models.basico_model import BasicoModel

@pytest.fixture(name="model_description_tool")
def model_description_tool_fixture():
    '''
    Fixture for creating an instance of ModelDescriptionTool.
    '''
    return ModelDescriptionTool()

@pytest.fixture(name="input_data")
def input_data_fixture():
    '''
    Fixture for creating an instance of AskQuestionInput.
    '''
    return ModelDescriptionInput(question="Describe the model",
                            sys_bio_model=ModelData(modelid=64),
                            st_session_key="test_key"
                            )

def test_run_with_missing_session_key(input_data, model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with a missing session key.
    '''
    # Delete the session key from the session state.
    st.session_state.pop(input_data.st_session_key, None)
    result = model_description_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert isinstance(result, str)

def test_model_data_initialization():
    """
    Test the initialization of the ModelData class.
    """
    model_data = ModelData(modelid=1,
                        sbml_file_path="path/to/file",
                        model_object=BasicoModel(model_id=1))
    assert model_data.modelid == 1
    assert model_data.sbml_file_path == "path/to/file"
    assert isinstance(model_data.model_object, BasicoModel)

def test_check_model_object():
    """
    Test the check_model_object method of the ModelData class.
    """
    # Test with valid BasicoModel object
    model_data = ModelData(model_object=BasicoModel(model_id=1))
    validated_data = model_data.check_model_object(model_data.__dict__)
    assert validated_data['model_object'] is not None

    # Test with invalid model_object
    model_data = ModelData(model_object="invalid_object")
    validated_data = model_data.check_model_object(model_data.__dict__)
    assert validated_data['model_object'] is None

    # Test with None model_object
    model_data = ModelData(model_object=None)
    validated_data = model_data.check_model_object(model_data.__dict__)
    assert validated_data['model_object'] is None

def test_run_with_valid_key_no_model_data(input_data, model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with a valid session key.
    '''
    st.session_state["test_key"] = None
    input_data.sys_bio_model = ModelData()
    result = model_description_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert result == "Please provide a BioModels ID or an SBML file path for the model."

def test_call_run_with_different_input_model_data(input_data, model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with different input model data.
    '''
    result = model_description_tool.call_run(question=input_data.question,
                                    sys_bio_model=input_data.sys_bio_model,
                                    st_session_key=input_data.st_session_key)
    assert isinstance(result, str)
    input_data.sys_bio_model = ModelData(sbml_file_path="./BIOMD0000000064_url.xml")
    result = model_description_tool.call_run(question=input_data.question,
                                    sys_bio_model=input_data.sys_bio_model,
                                    st_session_key=input_data.st_session_key)
    assert isinstance(result, str)
    model = BasicoModel(model_id=64)
    model.simulate(duration=2, interval=2)
    input_data.sys_bio_model = ModelData(model_object=model)
    result = model_description_tool.call_run(question=input_data.question,
                                    sys_bio_model=input_data.sys_bio_model,
                                    st_session_key=input_data.st_session_key)
    assert isinstance(result, str)
    # without simulation results
    model = BasicoModel(model_id=64)
    input_data.sys_bio_model = ModelData(model_object=model)
    result = model_description_tool.call_run(question=input_data.question,
                                    sys_bio_model=input_data.sys_bio_model,
                                    st_session_key=input_data.st_session_key)
    assert isinstance(result, str)

def test_run_with_none_key(input_data, model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with a None
    '''
    input_data.st_session_key = None
    result = model_description_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert isinstance(result, str)
    input_data.sys_bio_model = ModelData()
    result = model_description_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert result == "Please provide a valid model object or " \
                    "Streamlit session key that contains the model object."
    input_data.st_session_key = "test_key"
    # delete the session key form the session state
    st.session_state.pop(input_data.st_session_key, None)
    result = model_description_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert result == f"Session key {input_data.st_session_key} " \
        "not found in Streamlit session state."

def test_run_manager(input_data, model_description_tool):
    '''
    Test the _run method of the ModelDescriptionTool class with a run_manager.
    '''
    run_manager = CallbackManagerForToolRun(run_id=1, handlers=[], inheritable_handlers=False)
    result = model_description_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key,
                                        run_manager=run_manager)
    assert isinstance(result, str)
    run_manager = CallbackManagerForToolRun(run_id=1,
                                            handlers=[],
                                            inheritable_handlers=False,
                                            metadata={"prompt": '''Given: {description},
                                                                answer the question:
                                                                {question}.'''})
    result = model_description_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key,
                                        run_manager=run_manager)
    assert isinstance(result, str)

def test_get_metadata(model_description_tool):
    '''
    Test the get_metadata method of the ModelDescriptionTool class.
    '''
    metadata = model_description_tool.get_metadata()
    assert metadata["name"] == "model_description"
    assert metadata["description"] == "A tool to ask about the description of the model."
