#!/usr/bin/env python3

"""
Tool for loading PrimeKG from Harvard Dataverse.
The approach to load PrimeKG follows the one used in the TDC package:
https://github.com/mims-harvard/TDC/
"""

import os
from typing import Type, Optional
from dataclasses import dataclass
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
    file_id: Optional[int] = 6180626
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
             file_id: int,
             local_dir: str):
        """
        Method for running the KnowledgeGraphLoader tool.

        Args:
            name (str): The name of the PrimeKG.
            server_path (str): The server path of the PrimeKG.
            file_id (int): The file ID of the PrimeKG.
            file_name (str): The file name of the PrimeKG.
            local_dir (str): The local directory of the PrimeKG.

        Returns:
            str: The status of the PrimeKG loading.
        """
        # Make the directory if it doesn't exist
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)

        # Download the file if it does not exist in the local directory
        # Otherwise, load the data from the local directory
        local_full_path = os.path.join(local_dir, f"{name}.csv")
        if os.path.exists(local_full_path):
            print(f"{local_full_path} already exists. Loading the data from the local directory.")

            # Load the dataframes
            primekg_nodes = pd.read_csv(os.path.join(local_dir, f"{name}_nodes.tsv.gz"),
                                        sep='\t', compression='gzip')
            primekg_edges = pd.read_csv(os.path.join(local_dir, f"{name}_edges.tsv.gz"),
                                        sep='\t', compression='gzip')
        else:
            print(f"Downloading file from {server_path}{file_id}")
            response = requests.get(f"{server_path}{file_id}", stream=True, timeout=300)
            progress_bar = tqdm(total=int(response.headers.get("content-length", 0)),
                                unit="iB",
                                unit_scale=True)
            with open(local_full_path, "wb") as file:
                for data in response.iter_content(1024):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            # Load the data and get a preview of the dataframe
            primekg_df = pd.read_csv(local_full_path, sep=",", low_memory=False)

            # Prepare nodes
            primekg_nodes = pd.concat([
                primekg_df[['x_name', 'x_id', 'x_type', 'x_source']].rename(
                    columns={'x_name':'node',
                            'x_id':'node_uid',
                            'x_type':'node_type',
                            'x_source':'node_source'}),
                primekg_df[['y_name', 'y_id', 'y_type', 'y_source']].rename(
                    columns={'y_name':'node',
                            'y_id':'node_uid',
                            'y_type':'node_type',
                            'y_source':'node_source'})], axis=0)
            primekg_nodes['node_id'] = primekg_nodes['node']
            primekg_nodes['node_uid'] = primekg_nodes['node_source'].astype(str) + ':' +\
                primekg_nodes['node_uid'].astype(str)
            primekg_nodes.drop(['node_source'], axis=1)
            primekg_nodes.drop_duplicates(inplace=True)
            primekg_nodes = primekg_nodes[['node',
                                        'node_id',
                                        'node_uid',
                                        'node_type']]

            # Prepare edges
            primekg_edges = primekg_df[['x_name', 'y_name', 'display_relation']].rename(
                columns={'x_name':'node_source',
                        'y_name':'node_target',
                        'display_relation':'edge_type'})
            primekg_edges['node_source_uid'] = primekg_df['x_source'].astype(str) + ':' +\
                primekg_df['x_id'].astype(str)
            primekg_edges['node_target_uid'] = primekg_df['y_source'].astype(str) + ':' +\
                primekg_df['y_id'].astype(str)
            primekg_edges = primekg_edges[['node_source',
                                        'node_source_uid',
                                        'node_target',
                                        'node_target_uid',
                                        'edge_type']]
            primekg_edges['edge_type'] = primekg_edges['edge_type'].apply(
                lambda x: x.replace(' ', '_')
            )

            # Store the dataframes
            primekg_nodes.to_csv(os.path.join(local_dir, f"{name}_nodes.tsv.gz"),
                                index=False, sep='\t', compression='gzip')
            primekg_edges.to_csv(os.path.join(local_dir, f"{name}_edges.tsv.gz"),
                                index=False, sep='\t', compression='gzip')

        return primekg_nodes, primekg_edges

    def call_run(self,
                 name: str,
                 server_path: str,
                 file_id: int,
                 local_dir: str):
        """
        Run the tool.

        Args:
            name (str): The name of the PrimeKG.
            server_path (str): The server path of the PrimeKG.
            file_id (int): The file ID of the PrimeKG.
            local_dir (str): The local directory of the PrimeKG.

        Returns:
            str: The status of the PrimeKG loading.
        """
        return self._run(name=name,
                         server_path=server_path,
                         file_id=file_id,
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
