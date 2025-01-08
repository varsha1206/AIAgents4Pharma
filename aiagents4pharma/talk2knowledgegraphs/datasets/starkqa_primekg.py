"""
Class for loading StarkQAPrimeKG dataset.
"""

import os
import shutil
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files
from .dataset import Dataset

class StarkQAPrimeKG(Dataset):
    """
    Class for loading StarkQAPrimeKG dataset.
    It downloads the data from the HuggingFace repo and stores it in the local directory.
    The data is then loaded into pandas DataFrame of QA pairs, dictionary of split indices,
    and node information.
    """

    def __init__(self, local_dir: str = "../../../data/starkqa_primekg/"):
        """
        Constructor for StarkQAPrimeKG class.

        Args:
            local_dir (str): The local directory to store the dataset files.
        """
        self.name: str = "starkqa_primekg"
        self.hf_repo_id: str = "snap-stanford/stark"
        self.local_dir: str = local_dir
        # Attributes to store the data
        self.starkqa: pd.DataFrame = None
        self.starkqa_split_idx: dict = None
        self.starkqa_node_info: dict = None

        # Set up the dataset
        self.setup()

    def setup(self):
        """
        A method to set up the dataset.
        """
        # Make the directory if it doesn't exist
        os.makedirs(os.path.dirname(self.local_dir), exist_ok=True)

    def _load_stark_repo(self) -> pd.DataFrame:
        """
        Private method to load related files of StarkQAPrimeKG dataset.

        Returns:
            pd.DataFrame: The nodes dataframe of StarkQAPrimeKG dataset.
        """
        # Download the file if it does not exist in the local directory
        # Otherwise, load the data from the local directory
        local_file = os.path.join(self.local_dir, "qa/prime/stark_qa/stark_qa.csv")
        if os.path.exists(local_file):
            print(f"{local_file} already exists. Loading the data from the local directory.")
        else:
            print(f"Downloading files from {self.hf_repo_id}")

            # List all related files in the HuggingFace Hub repository
            files = list_repo_files(self.hf_repo_id, repo_type="dataset")
            files = [f for f in files if ((f.startswith("qa/prime/") or
                                           f.startswith("skb/prime/")) and f.find("raw") == -1)]

            # Download and save each file in the specified folder
            for file in tqdm(files):
                _ = hf_hub_download(self.hf_repo_id,
                                    file,
                                    repo_type="dataset",
                                    local_dir=self.local_dir)

            # Unzip the processed files
            shutil.unpack_archive(
                os.path.join(self.local_dir, "skb/prime/processed.zip"),
                os.path.join(self.local_dir, "skb/prime/")
            )

        # Load StarkQA dataframe
        starkqa = pd.read_csv(
            os.path.join(self.local_dir, "qa/prime/stark_qa/stark_qa.csv"),
            low_memory=False)

        # Read split indices
        qa_indices = sorted(starkqa['id'].tolist())
        starkqa_split_idx = {}
        for split in ['train', 'val', 'test', 'test-0.1']:
            indices_file = os.path.join(self.local_dir, "qa/prime/split", f'{split}.index')
            with open(indices_file, 'r', encoding='utf-8') as f:
                indices = f.read().strip().split('\n')
            query_ids = [int(idx) for idx in indices]
            starkqa_split_idx[split] = np.array(
                [qa_indices.index(query_id) for query_id in query_ids]
            )

        # Load the node info of PrimeKG preprocessed for StarkQA
        with open(os.path.join(self.local_dir, 'skb/prime/processed/node_info.pkl'), 'rb') as f:
            starkqa_node_info = pickle.load(f)

        return starkqa, starkqa_split_idx, starkqa_node_info

    def load_data(self):
        """
        Load the StarkQAPrimeKG dataset into pandas DataFrame of QA pairs,
        dictionary of split indices, and node information.
        """
        print("Loading StarkQAPrimeKG dataset...")
        self.starkqa, self.starkqa_split_idx, self.starkqa_node_info = self._load_stark_repo()


    def get_starkqa(self) -> pd.DataFrame:
        """
        Get the dataframe of StarkQAPrimeKG dataset, containing the QA pairs.

        Returns:
            pd.DataFrame: The nodes dataframe of PrimeKG dataset.
        """
        return self.starkqa

    def get_starkqa_split_indicies(self) -> dict:
        """
        Get the split indices of StarkQAPrimeKG dataset.

        Returns:
            dict: The split indices of StarkQAPrimeKG dataset.
        """
        return self.starkqa_split_idx

    def get_starkqa_node_info(self) -> dict:
        """
        Get the node information of StarkQAPrimeKG dataset.

        Returns:
            dict: The node information of StarkQAPrimeKG dataset.
        """
        return self.starkqa_node_info
