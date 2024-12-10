"""
Test cases for starkqa_primekg_loader.py
"""

import os
import shutil
import pytest
from ..tools.starkqa_primekg_loader import (
    StarkQAPrimeKGData,
    StarkQAPrimeKGLoaderInput,
    StarkQAPrimeKGLoaderTool
)

# Remove the data folder for testing if it exists
LOCAL_DIR = "../data/starkqa_primekg_test/"
if os.path.exists(LOCAL_DIR):
    shutil.rmtree(LOCAL_DIR)

@pytest.fixture(name="starkqa_primekg_data")
def starkqa_primekg_data_fixture():
    """
    Fixture for creating an instance of StarkQAPrimeKGData.
    """
    return StarkQAPrimeKGData(repo_id="snap-stanford/stark",
                              local_dir=LOCAL_DIR)

@pytest.fixture(name="starkqa_primekg_loader_input")
def starkqa_primekg_loader_input_fixture(starkqa_primekg_data):
    """
    Fixture for creating an instance of StarkQAPrimeKGLoaderInput.
    """
    return StarkQAPrimeKGLoaderInput(data=starkqa_primekg_data)

@pytest.fixture(name="starkqa_primekg_loader_tool")
def starkqa_primekg_loader_tool_fixture():
    """
    Fixture for creating an instance of StarkQAPrimeKGLoaderTool.
    """
    return StarkQAPrimeKGLoaderTool()

def test_download_starkqa_primekg(starkqa_primekg_loader_input, starkqa_primekg_loader_tool):
    """
    Test the _run method of the StarkQAPrimeKGLoaderTool class by downloading files
    from HuggingFace Hub.
    """
    starkqa_df, split_idx = starkqa_primekg_loader_tool.call_run(
        repo_id=starkqa_primekg_loader_input.data.repo_id,
        local_dir=starkqa_primekg_loader_input.data.local_dir
    )

    # Check if the local directory exists
    assert os.path.exists(starkqa_primekg_loader_input.data.local_dir)
    # Check if downloaded files exist in the local directory
    files = ['qa/prime/split/test-0.1.index',
             'qa/prime/split/test.index',
             'qa/prime/split/train.index',
             'qa/prime/split/val.index',
             'qa/prime/stark_qa/stark_qa.csv',
             'qa/prime/stark_qa/stark_qa_human_generated_eval.csv']
    for file in files:
        path = f"{starkqa_primekg_loader_input.data.local_dir}/{file}"
        assert os.path.exists(path)
    # Check dataframe
    assert starkqa_df is not None
    assert len(starkqa_df) > 0
    assert starkqa_df.shape[0] == 11204
    # Check split indices
    assert list(split_idx.keys()) == ['train', 'val', 'test', 'test-0.1']
    assert len(split_idx['train']) == 6162
    assert len(split_idx['val']) == 2241
    assert len(split_idx['test']) == 2801
    assert len(split_idx['test-0.1']) == 280

def test_load_existing_starkqa_primekg(starkqa_primekg_loader_input, starkqa_primekg_loader_tool):
    """

    Test the _run method of the StarkQAPrimeKGLoaderTool class by loading existing files
    in the local directory.
    """
    starkqa_df, split_idx = starkqa_primekg_loader_tool.call_run(
        repo_id=starkqa_primekg_loader_input.data.repo_id,
        local_dir=starkqa_primekg_loader_input.data.local_dir
    )

    # Check if the local directory exists
    assert os.path.exists(starkqa_primekg_loader_input.data.local_dir)
    # Check if downloaded and processed files exist
    files = ['qa/prime/split/test-0.1.index',
             'qa/prime/split/test.index',
             'qa/prime/split/train.index',
             'qa/prime/split/val.index',
             'qa/prime/stark_qa/stark_qa.csv',
             'qa/prime/stark_qa/stark_qa_human_generated_eval.csv']
    for file in files:
        path = f"{starkqa_primekg_loader_input.data.local_dir}/{file}"
        assert os.path.exists(path)
    # Check dataframe
    assert starkqa_df is not None
    assert len(starkqa_df) > 0
    assert starkqa_df.shape[0] == 11204
    # Check split indices
    assert list(split_idx.keys()) == ['train', 'val', 'test', 'test-0.1']
    assert len(split_idx['train']) == 6162
    assert len(split_idx['val']) == 2241
    assert len(split_idx['test']) == 2801
    assert len(split_idx['test-0.1']) == 280

def test_get_metadata(starkqa_primekg_loader_tool):
    """
    Test the get_metadata method of the StarkQAPrimeKGLoaderTool class.
    """
    metadata = starkqa_primekg_loader_tool.get_metadata()
    # Check metadata
    assert metadata["name"] == "starkqa_primekg_loader"
    assert metadata["description"] == "A tool to load StarkQA-PrimeKG from HuggingFace Hub."
