#!/usr/bin/env python3

"""
Tool for loading StarkQA-PrimeKG dataset from HuggingFace Hub.
The approach to load StarkQA-PrimeKG follows the one used in the Stark repository:
https://github.com/snap-stanford/stark/
"""

import os
from typing import Type, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain_core.tools import BaseTool
from huggingface_hub import hf_hub_download, list_repo_files

@dataclass
class StarkQAPrimeKGData:
    """
    Dataclass for StarkQA-PrimeKG data.
    """
    repo_id: Optional[str] = "snap-stanford/stark"
    local_dir: Optional[str] = "../../../data/starkqa_primekg/"

class StarkQAPrimeKGLoaderInput(BaseModel):
    """
    Input schema for the StarkQAPrimeKGLoaderInput tool.
    """
    data: StarkQAPrimeKGData = Field(description="The StarkQAPrimeKGData data.", default=None)

class StarkQAPrimeKGLoaderTool(BaseTool):
    """
    Tool for loading StarkQA-PrimeKG from HuggingFace Hub.
    """
    name: str = "starkqa_primekg_loader"
    description: str = "A tool to load StarkQA-PrimeKG from HuggingFace Hub."
    args_schema: Type[BaseModel] = StarkQAPrimeKGLoaderInput
    st_session_key: str = None

    def _run(self,
             repo_id: str,
             local_dir: str) -> tuple[pd.DataFrame, dict]:
        """
        Method for running the KnowledgeGraphLoader tool.

        Args:
            repo_id (str): The repo ID of the StarkQA-PrimeKG.
            local_dir (str): The local directory of the StarkQA-PrimeKG.

        Returns:
            tuple: The StarkQA-PrimeKG dataset and split indices.
        """
        # Make the directory if it doesn't exist
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)

        # Download the file if it does not exist in the local directory
        # Otherwise, load the data from the local directory
        local_full_path = os.path.join(local_dir, "qa/prime/stark_qa/stark_qa.csv")
        if os.path.exists(local_full_path):
            print(f"{local_full_path} already exists. Loading the data from the local directory.")

            # Load the dataframes
            starkqa_df = pd.read_csv(os.path.join(local_dir, "qa/prime/stark_qa/stark_qa.csv"),
                                     low_memory=False)
        else:
            print(f"Downloading file from {repo_id}")
            # List all related files in the repository
            files = list_repo_files(repo_id, repo_type="dataset")
            files = [f for f in files if f.startswith("qa/prime/")]
            # Download and save each file in the folder
            for file in tqdm(files):
                _ = hf_hub_download(repo_id,
                                            file,
                                            repo_type="dataset",
                                            local_dir=local_dir)

        # Load StarkQA dataframe
        starkqa_df = pd.read_csv(os.path.join(local_dir, "qa/prime/stark_qa/stark_qa.csv"),
                                    low_memory=False)
        qa_indices = sorted(starkqa_df['id'].tolist())

        # Read split indices
        split_idx = {}
        for split in ['train', 'val', 'test', 'test-0.1']:
            indices_file = os.path.join(local_dir, "qa/prime/split", f'{split}.index')
            with open(indices_file, 'r', encoding='utf-8') as f:
                indices = f.read().strip().split('\n')
            query_ids = [int(idx) for idx in indices]
            split_idx[split] = np.array([qa_indices.index(query_id) for query_id in query_ids])

        return starkqa_df, split_idx

    def call_run(self,
                 repo_id: str,
                 local_dir: str) -> tuple[pd.DataFrame, dict]:
        """
        Run the tool.

        Args:
            repo_id (str): The repo ID of the StarkQA-PrimeKG.
            local_dir (str): The local directory of the StarkQA-PrimeKG.

        Returns:
            tuple: The StarkQA-PrimeKG dataset and split indices.
        """
        return self._run(repo_id=repo_id,
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
