#!/usr/bin/env python3

"""
Tool for simulating a model.
"""

from typing import Optional, Type, Union
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import streamlit as st
import plotly.express as px
from ..models.copasimodel import CopasiModel

class SimulateModelInput(BaseModel):
    """
    Input schema for the SimulateModel tool.
    """
    modelid: int = Field(description="model id", default=None)
    duration: Union[int, float] = Field(description="duration", default=100.0)
    interval: int = Field(description="interval", default=10)
    species_name: str = Field(description="species name", default=None)
    species_concentration: Union[int, float] = \
                            Field(description="species concentration", default=None)
    st_session_key: str = Field(description="Streamlit session key")

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class SimulateModelTool(BaseTool):
    """
    Tool for simulating a model.
    """
    name: str = "simulate_model"
    description: str = "A tool for simulating a model."
    args_schema: Type[BaseModel] = SimulateModelInput

    def _run(self,
             modelid: Optional[int] = None,
             duration: Optional[Union[int, float]] = 100.0,
             interval: Optional[int] = 10,
             species_name: Optional[str] = None,
             species_concentration: Optional[Union[int, float]] = None,
             st_session_key: str = None) -> str:
        """
        Run the tool.

        Args:
            modelid (Optional[int]): The model ID.
            duration (Optional[Union[int, float]]): The duration of the simulation.
            interval (Optional[int]): The interval of the simulation.
            species_name (Optional[str]): The species name.
            species_concentration (Optional[Union[int, float]]): The species concentration.
            st_session_key (str): The Streamlit session key.

        Returns:
            str: The result of the simulation.
        """
        if modelid is None:
            if st_session_key not in st.session_state:
                return f"Session key {st_session_key} not found in Streamlit session state."
            model_object = st.session_state[st_session_key]
            if model_object is None:
                return "Please provide a valid model ID first for simulation."
            modelid = model_object.model_id
        else:
            model_object = CopasiModel(model_id=modelid)
            st.session_state[st_session_key] = model_object
        df = model_object.simulate(parameters={species_name: species_concentration},
                                duration=duration,
                                interval=interval)
        # Convert the DataFrame to long format for plotting
        # and ignore the index column
        df = df.melt(id_vars='Time',
                    var_name='Species',
                    value_name='Concentration')
        # Plot the simulation results using Plotly
        fig = px.line(df,
                        x='Time',
                        y='Concentration',
                        color='Species',
                        title=f"Concentration of Species over Time in the model {modelid}",
                        height=600,
                        width=800
        )
        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width = False)
        return f"Simulation results for the model {modelid}."

    def run(self,
            modelid: Optional[int] = None,
            duration: Optional[Union[int, float]] = 100.0,
            interval: Optional[int] = 10,
            species_name: Optional[str] = None,
            species_concentration: Optional[Union[int, float]] = None,
            st_session_key: str = None) -> str:
        """
        Run the tool.

        Args:
            modelid (Optional[int]): The model ID.
            duration (Optional[Union[int, float]]): The duration of the simulation.
            interval (Optional[int]): The interval of the simulation.
            species_name (Optional[str]): The species name.
            species_concentration (Optional[Union[int, float]]): The species concentration.
            st_session_key (str): The Streamlit session key.

        Returns:
            str: The result of the simulation.
        """
        return self._run(modelid=modelid,
                         duration=duration,
                         interval=interval,
                         species_name=species_name,
                         species_concentration=species_concentration,
                         st_session_key=st_session_key)

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
