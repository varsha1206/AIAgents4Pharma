#!/usr/bin/env python3

"""
Tool for loading PrimeKG from Harvard Dataverse.
The approach to load PrimeKG follows the one used in the TDC package:
https://github.com/mims-harvard/TDC/
"""

import os
from typing import Type, Optional
from dataclasses import dataclass, field
import requests
from pydantic import BaseModel, Field
import pandas as pd
from tqdm import tqdm
from langchain_core.tools import BaseTool

@dataclass
class PrimeKGData:
    """
    Dataclass for PrimeKG data.
    """
    name: Optional[str] = "primekg"
    server_path: Optional[str] = "https://dataverse.harvard.edu/api/access/datafile/"
    file_ids: Optional[dict] = field(default_factory=lambda: {"nodes": 6180617, "edges": 6180616})
    local_dir: Optional[str] = "../../../data/primekg/"

class PrimeKGLoaderInput(BaseModel):
    """
    Input schema for the PrimeKGLoaderInput tool.
    """
    data: PrimeKGData = Field(description="The PrimeKG data.", default=None)

class PrimeKGLoaderTool(BaseTool):
    """
    Tool for loading PrimeKG from Harvard Dataverse.
    """
    name: str = "primekg_loader"
    description: str = "A tool to load PrimeKG from Harvard Dataverse."
    args_schema: Type[BaseModel] = PrimeKGLoaderInput
    st_session_key: str = None

    def _run(self,
             name: str,
             server_path: str,
             file_ids: dict,
             local_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method for running the PrimeKGLoaderTool tool.

        Args:
            name (str): The name of the PrimeKG.
            server_path (str): The server path of the PrimeKG.
            file_ids (dict): The file IDs of the PrimeKG (nodes and edges).
            file_name (str): The file name of the PrimeKG.
            local_dir (str): The local directory of the PrimeKG.

        Returns:
            tuple: The PrimeKG nodes and edges dataframes.
        """
        # Make the directory if it doesn't exist
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)

        # Download the file if it does not exist in the local directory
        # Otherwise, load the data from the local directory
        # Nodes retrieval
        local_node_path = os.path.join(local_dir, f"{name}_nodes.tsv.gz")
        if os.path.exists(local_node_path):
            print(f"{local_node_path} already exists. Loading the data from the local directory.")

            # Load the dataframe
            primekg_nodes = pd.read_csv(local_node_path,
                                        sep="\t",
                                        compression="gzip",
                                        low_memory=False)
        else:
            print(f"Downloading node file from {server_path}{file_ids["nodes"]}")
            # Download nodes
            response = requests.get(f"{server_path}{file_ids["nodes"]}", stream=True, timeout=300)
            progress_bar = tqdm(total=int(response.headers.get("content-length", 0)),
                                unit="iB",
                                unit_scale=True)
            with open(os.path.join(local_dir, "nodes.tab"), "wb") as file:
                for data in response.iter_content(1024):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            # Load the dataframe of nodes
            primekg_nodes = pd.read_csv(os.path.join(local_dir, "nodes.tab"),
                                        sep="\t",
                                        low_memory=False)

            # Prepare nodes
            primekg_nodes.rename(columns={'node_name': 'node'}, inplace=True)
            primekg_nodes['node_uid'] = primekg_nodes['node_source'].astype(str) + ':' +\
                  primekg_nodes['node_id'].astype(str)
            primekg_nodes.drop(['node_source', 'node_id'], axis=1, inplace=True)
            # Reorganize columns
            primekg_nodes = primekg_nodes[['node_index', 'node', 'node_uid', 'node_type']]

            # Store the dataframe by compressing it
            primekg_nodes.to_csv(local_node_path, index=False, sep="\t", compression="gzip")

        # Edges retrieval
        local_edge_path = os.path.join(local_dir, f"{name}_edges.tsv.gz")
        if os.path.exists(local_edge_path):
            print(f"{local_edge_path} already exists. Loading the data from the local directory.")

            # Load the dataframe
            primekg_edges = pd.read_csv(local_edge_path,
                                        sep="\t",
                                        compression="gzip",
                                        low_memory=False)
        else:
            print(f"Downloading edge file from {server_path}{file_ids["edges"]}")
            # Download edges
            response = requests.get(f"{server_path}{file_ids["edges"]}", stream=True, timeout=300)
            progress_bar = tqdm(total=int(response.headers.get("content-length", 0)),
                                unit="iB",
                                unit_scale=True)
            with open(os.path.join(local_dir, "edges.csv"), "wb") as file:
                for data in response.iter_content(1024):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            # Load the dataframe of edges
            primekg_edges = pd.read_csv(os.path.join(local_dir, "edges.csv"),
                                        sep=",",
                                        low_memory=False)

            # Prepare edges
            # Add Source
            primekg_edges = primekg_edges.merge(primekg_nodes,
                                                left_on='x_index',
                                                right_on='node_index')
            primekg_edges.drop(['x_index', 'node_type'], axis=1, inplace=True)
            primekg_edges.rename(columns={'node_index': 'node_source_index',
                                          'node': 'node_source',
                                          'node_uid': 'node_source_uid'}, inplace=True)
            # Add Target
            primekg_edges = primekg_edges.merge(primekg_nodes,
                                                left_on='y_index',
                                                right_on='node_index')
            primekg_edges.drop(['y_index', 'node_type'], axis=1, inplace=True)
            primekg_edges.rename(columns={'node_index': 'node_target_index',
                                          'node': 'node_target',
                                          'node_uid': 'node_target_uid'}, inplace=True)
            # Reorganize columns
            primekg_edges.rename(columns={'display_relation': 'edge_type'}, inplace=True)
            primekg_edges = primekg_edges[['node_source_index',
                                           'node_source',
                                           'node_source_uid',
                                           'node_target_index',
                                           'node_target',
                                           'node_target_uid',
                                           'edge_type']]

            # Store the dataframe by compressing it
            primekg_edges.to_csv(local_edge_path, index=False, sep="\t", compression="gzip")

        return primekg_nodes, primekg_edges

    def call_run(self,
                 name: str,
                 server_path: str,
                 file_ids: dict,
                 local_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the tool.

        Args:
            name (str): The name of the PrimeKG.
            server_path (str): The server path of the PrimeKG.
            file_ids (dict): The file IDs of the PrimeKG (nodes and edges).
            local_dir (str): The local directory of the PrimeKG.

        Returns:
            tuple: The PrimeKG nodes and edges dataframes.
        """
        return self._run(name=name,
                         server_path=server_path,
                         file_ids=file_ids,
                         local_dir=local_dir)

    def get_metadata(self) -> dict:
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
