"""
Test cases for primekg_loader.py
"""

import os
import shutil
import pytest
from ..tools.primekg_loader import PrimeKGData, PrimeKGLoaderInput, PrimeKGLoaderTool

# Remove the data folder for testing if it exists
LOCAL_DIR = "../data/primekg_test/"
if os.path.exists(LOCAL_DIR):
    shutil.rmtree(LOCAL_DIR)

@pytest.fixture(name="primekg_data")
def primekg_data_fixture():
    """
    Fixture for creating an instance of PrimeKGData.
    """
    return PrimeKGData(name="primekg",
                       server_path="https://dataverse.harvard.edu/api/access/datafile/",
                       file_id=6180626,
                       local_dir=LOCAL_DIR)

@pytest.fixture(name="primekg_loader_input")
def primekg_loader_input_fixture(primekg_data):
    """
    Fixture for creating an instance of PrimeKGLoaderInput.
    """
    return PrimeKGLoaderInput(data=primekg_data)

@pytest.fixture(name="primekg_loader_tool")
def primekg_loader_tool_fixture():
    """
    Fixture for creating an instance of PrimeKGLoaderTool.
    """
    return PrimeKGLoaderTool()

def test_download_primekg(primekg_loader_input, primekg_loader_tool):
    """
    Test the _run method of the PrimeKGLoaderTool class by downloading PrimeKG from server.
    """
    primekg_nodes, primekg_edges = primekg_loader_tool.call_run(
        name=primekg_loader_input.data.name,
        server_path=primekg_loader_input.data.server_path,
        file_id=primekg_loader_input.data.file_id,
        local_dir=primekg_loader_input.data.local_dir
    )

    # Check if the local directory exists
    assert os.path.exists(primekg_loader_input.data.local_dir)
    # Check if downloaded and processed files exist
    path = f"{primekg_loader_input.data.local_dir}/{primekg_loader_input.data.name}.csv"
    assert os.path.exists(path)
    path = f"{primekg_loader_input.data.local_dir}/{primekg_loader_input.data.name}_nodes.tsv.gz"
    assert os.path.exists(path)
    path = f"{primekg_loader_input.data.local_dir}/{primekg_loader_input.data.name}_edges.tsv.gz"
    assert os.path.exists(path)
    # Check processed PrimeKG dataframes
    # Nodes
    assert primekg_nodes is not None
    assert len(primekg_nodes) > 0
    assert primekg_nodes.shape[0] == 129375
    # Edges
    assert primekg_edges is not None
    assert len(primekg_edges) > 0
    assert primekg_edges.shape[0] == 8100498

def test_load_existing_primekg(primekg_loader_input, primekg_loader_tool):
    """
    Test the _run method of the PrimeKGLoaderTool class by loading existing PrimeKG in local.
    """
    primekg_nodes, primekg_edges = primekg_loader_tool.call_run(
        name=primekg_loader_input.data.name,
        server_path=primekg_loader_input.data.server_path,
        file_id=primekg_loader_input.data.file_id,
        local_dir=primekg_loader_input.data.local_dir
    )

    # Check if the local directory exists
    assert os.path.exists(primekg_loader_input.data.local_dir)
    # Check if downloaded and processed files exist
    path = f"{primekg_loader_input.data.local_dir}/{primekg_loader_input.data.name}.csv"
    assert os.path.exists(path)
    path = f"{primekg_loader_input.data.local_dir}/{primekg_loader_input.data.name}_nodes.tsv.gz"
    assert os.path.exists(path)
    path = f"{primekg_loader_input.data.local_dir}/{primekg_loader_input.data.name}_edges.tsv.gz"
    assert os.path.exists(path)
    # Check processed PrimeKG dataframes
    # Nodes
    assert primekg_nodes is not None
    assert len(primekg_nodes) > 0
    assert primekg_nodes.shape[0] == 129375
    # Edges
    assert primekg_edges is not None
    assert len(primekg_edges) > 0
    assert primekg_edges.shape[0] == 8100498

def test_get_metadata(primekg_loader_tool):
    """
    Test the get_metadata method of the PrimeKGLoaderTool class.
    """
    metadata = primekg_loader_tool.get_metadata()
    # Check metadata
    assert metadata["name"] == "primekg_loader"
    assert metadata["description"] == "A tool to load PrimeKG from Harvard Dataverse."
