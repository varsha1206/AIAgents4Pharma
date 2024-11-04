'''
Test cases for simulate_model.py
'''

import pytest
import streamlit as st
from ..tools.simulate_model import SimulateModelTool, SimulateModelInput

@pytest.fixture(name="simulate_model_tool")
def simulate_model_tool_fixture():
    '''
    Fixture for creating an instance of SimulateModelTool.
    '''
    return SimulateModelTool()

def test_run_with_valid_modelid(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class with a valid model ID.
    '''
    input_data = SimulateModelInput(
        modelid=64,
        duration=100.0,
        interval=10,
        species_name="species1",
        species_concentration=1.0,
        st_session_key="test_key"
    )
    st.session_state["test_key"] = None
    result = simulate_model_tool.run(**input_data.model_dump())
    assert result == f"Simulation results for the model {input_data.modelid}."

def test_run_with_invalid_modelid(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class with an invalid model ID.
    '''
    input_data = SimulateModelInput(
        duration=100.0,
        interval=10,
        species_name="species1",
        species_concentration=1.0,
        st_session_key="test_key"
    )
    st.session_state["test_key"] = None
    result = simulate_model_tool.run(**input_data.model_dump())
    assert result == "Please provide a valid model ID first for simulation."
    input_data2 = SimulateModelInput(
        modelid=64,
        duration=100.0,
        interval=10,
        species_name="species1",
        species_concentration=1.0,
        st_session_key="test_key"
    )
    simulate_model_tool.run(**input_data2.model_dump())
    result = simulate_model_tool.run(**input_data.model_dump())
    assert result == "Simulation results for the model 64."

def test_run_with_missing_session_key(simulate_model_tool):
    '''
    Test the _run method of the SimulateModelTool class with a missing session key.
    '''
    input_data = SimulateModelInput(
        duration=100.0,
        interval=10,
        species_name="species1",
        species_concentration=1.0,
        st_session_key="missing_key"
    )
    print (st.session_state)
    result = simulate_model_tool.run(**input_data.model_dump())
    assert result == "Session key missing_key not found in Streamlit session state."

def test_get_metadata(simulate_model_tool):
    '''
    Test the get_metadata method of the SimulateModelTool class.
    '''
    metadata = simulate_model_tool.get_metadata()
    assert metadata["name"] == "simulate_model"
    assert metadata["description"] == "A tool for simulating a model."
    assert metadata["return_direct"] == simulate_model_tool.return_direct
