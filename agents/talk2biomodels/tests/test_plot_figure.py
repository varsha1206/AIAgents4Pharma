'''
Test cases for plot_figure.py
'''

import pytest
import streamlit as st
from ..tools.plot_figure import PlotImageTool, PlotImageInput, ModelData
from ..models.basico_model import BasicoModel

@pytest.fixture(name="plot_image_tool")
def plot_image_tool_fixture():
    '''
    Fixture for creating an instance of PlotImageTool.
    '''
    return PlotImageTool()

@pytest.fixture(name="input_data")
def input_data_fixture():
    '''
    Fixture for creating an instance of AskQuestionInput.
    '''
    return PlotImageInput(question="What is the concentration of Pyruvate at time 5?",
                            sys_bio_model=ModelData(modelid=64),
                            st_session_key="test_key"
                            )

def test_call_run(input_data, plot_image_tool):
    '''
    Test the _run method of the PlotImageTool class with an invalid model ID.
    '''
    input_data.sys_bio_model = ModelData()
    st.session_state["test_key"] = None
    result = plot_image_tool.call_run(question=input_data.question,
                                    sys_bio_model=input_data.sys_bio_model,
                                    st_session_key=input_data.st_session_key)
    assert result == "Please run the simulation first before plotting the figure."
    st.session_state["test_key"] = BasicoModel(model_id=64)
    st.session_state["test_key"].simulate(duration=2, interval=2)
    result = plot_image_tool.call_run(question=input_data.question,
                                    sys_bio_model=input_data.sys_bio_model,
                                    st_session_key=input_data.st_session_key)
    assert result == "Figure plotted successfully"

def test_call_run_with_different_input_model_data(input_data, plot_image_tool):
    '''
    Test the _run method of the PlotImageTool class with different input model data.
    '''
    result = plot_image_tool.call_run(question=input_data.question,
                                    sys_bio_model=input_data.sys_bio_model,
                                    st_session_key=input_data.st_session_key)
    assert result == "Figure plotted successfully"
    input_data.sys_bio_model = ModelData(sbml_file_path="./BIOMD0000000064_url.xml")
    result = plot_image_tool.call_run(question=input_data.question,
                                    sys_bio_model=input_data.sys_bio_model,
                                    st_session_key=input_data.st_session_key)
    assert result == "Figure plotted successfully"
    model = BasicoModel(model_id=64)
    model.simulate(duration=2, interval=2)
    input_data.sys_bio_model = ModelData(model_object=model)
    result = plot_image_tool.call_run(question=input_data.question,
                                    sys_bio_model=input_data.sys_bio_model,
                                    st_session_key=input_data.st_session_key)
    assert result == "Figure plotted successfully"
    # without simulation results
    model = BasicoModel(model_id=64)
    input_data.sys_bio_model = ModelData(model_object=model)
    result = plot_image_tool.call_run(question=input_data.question,
                                    sys_bio_model=input_data.sys_bio_model,
                                    st_session_key=input_data.st_session_key)
    assert result == "Figure plotted successfully"

def test_run_with_none_key(input_data, plot_image_tool):
    '''
    Test the _run method of the AskQuestionTool class with a None session key.
    '''
    input_data.st_session_key = None
    result = plot_image_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert isinstance(result, str)
    input_data.sys_bio_model = ModelData()
    result = plot_image_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert result == "Please provide a valid model object or \
                    Streamlit session key that contains the model object."
    input_data.st_session_key = "test_key"
    # delete the session key form the session state
    st.session_state.pop(input_data.st_session_key, None)
    result = plot_image_tool.call_run(question=input_data.question,
                                        sys_bio_model=input_data.sys_bio_model,
                                        st_session_key=input_data.st_session_key)
    assert result == f"Session key {input_data.st_session_key} " \
        "not found in Streamlit session state."

def test_get_metadata(plot_image_tool):
    '''
    Test the get_metadata method of the PlotImageTool class.
    '''
    metadata = plot_image_tool.get_metadata()
    assert metadata["name"] == "plot_figure"
    assert metadata["description"] == "A tool to plot or visualize the simulation results."
