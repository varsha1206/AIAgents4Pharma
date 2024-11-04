'''
Test cases for plot_figure.py
'''

import pytest
import streamlit as st
from ..tools.plot_figure import PlotImageTool, PlotImageInput
from ..models.copasimodel import CopasiModel

@pytest.fixture(name="plot_image_tool")
def plot_image_tool_fixture():
    '''
    Fixture for creating an instance of PlotImageTool.
    '''
    return PlotImageTool()

def test_call_run(plot_image_tool):
    '''
    Test the _run method of the PlotImageTool class with an invalid model ID.
    '''
    input_data = PlotImageInput(
        question="Plot a figure showing concentration of Pyruvate over time.",
        st_session_key="test_key"
    )
    st.session_state["test_key"] = None
    result = plot_image_tool.call_run(**input_data.model_dump())
    assert result == "Please provide a valid model ID for simulation."
    st.session_state["test_key"] = CopasiModel(model_id=64)
    st.session_state["test_key"].simulate(duration=2, interval=2)
    result = plot_image_tool.call_run(**input_data.model_dump())
    assert result == "Figure plotted successfully"

def test_get_metadata(plot_image_tool):
    '''
    Test the get_metadata method of the PlotImageTool class.
    '''
    metadata = plot_image_tool.get_metadata()
    assert metadata["name"] == "plot_figure"
    assert metadata["description"] == "A tool to plot or visualize the simulation results."
