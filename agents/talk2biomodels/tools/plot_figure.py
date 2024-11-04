#!/usr/bin/env python3

"""
Tool for plotting a figure.
"""

import matplotlib.pyplot as plt
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser

class PlotImageInput(BaseModel):
    """
    Input schema for the PlotImage tool.
    """
    question: str = Field(description="Description of the plot")
    st_session_key: str = Field(description="Streamlit session key")

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class PlotImageTool(BaseTool):
    """
    Tool for plotting a figure.
    """
    name: str = "plot_figure"
    description: str = "A tool to plot or visualize the simulation results."
    args_schema: Type[BaseModel] = PlotImageInput

    def _run(self,
             question: str,
             st_session_key: str) -> str:
        """Use the tool."""
        model_object = st.session_state[st_session_key]
        modelid = model_object.model_id
        if modelid is None:
            return "Please provide a valid model ID for simulation."
        df = model_object.simulation_results
        tool = PythonAstREPLTool(locals={"df": df})
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        llm_with_tools = llm.bind_tools([tool], tool_choice=tool.name)
        system = f"""
                    You have access to a pandas dataframe `df`. \
                    Here is the output of `df.head().to_markdown()`:
                    {df.head().to_markdown()}
                    Given a user question, write the Python code to \
                    plot a figure of the answer using matplolib. \
                    Return ONLY the valid Python code and nothing else. \
                    The firgure size should be equal or smaller than (2, 2). \
                    Show the grid and legend. The font size of the legend should be 6. \
                    Also write a suitable title for the figure. The font size of the title should be 8. \
                    The font size of the x-axis and y-axis labels should be 8. \
                    The font size of the x-axis and y-axis ticks should be 6. \
                    Use color-blind friendly colors. The figure must be of high quality. \
                    Don't assume you have access to any libraries other \
                    than built-in Python ones, pandas, streamlit and matplotlib.
                    """
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])
        parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)
        code_chain = prompt | llm_with_tools | parser
        response = code_chain.invoke({"question": question})
        # print (response)
        exec(response['query'], globals(), {"df": df, "plt": plt})
        # load for plotly
        fig = plt.gcf()
        st.pyplot(fig, use_container_width=False)
        st.dataframe(df)
        # return None

    def get_metadata(self):
        """
        Get metadata for the tool.
        """
        return {
            "name": self.name,
            "description": self.description,
            "return_direct": self.return_direct,
        }

    def get_tool_type(self):
        """
        Get the type of the tool.
        """
        return "tool"
