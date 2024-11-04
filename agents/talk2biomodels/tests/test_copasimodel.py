'''
A test CopasiModel class for pytest unit testing.
'''

import pandas as pd
import basico
from ..models.copasimodel import CopasiModel

def test_with_model_id():
    """
    Test initialization of CopasiModel with model_id.
    """
    model = CopasiModel(model_id=64)
    assert model.model_id == 64
    # check if the simulation results are a pandas DataFrame object
    assert isinstance(model.simulate(parameters={'Species1': 0.5}, duration=2, interval=2),
                    pd.DataFrame)
    assert isinstance(model.simulate(parameters={None: None}, duration=2, interval=2),
                    pd.DataFrame)
    assert model.description == basico.biomodels.get_model_info(64)["description"]

def test_with_sbml_file():
    """
    Test initialization of CopasiModel with sbml_file_path.
    """
    model_object = CopasiModel(sbml_file_path="./BIOMD0000000064_url.xml")
    assert model_object.sbml_file_path == "./BIOMD0000000064_url.xml"
    assert isinstance(model_object.simulate(duration=2, interval=2), pd.DataFrame)
    assert isinstance(model_object.simulate(parameters={'NADH': 0.5}, duration=2, interval=2),
                      pd.DataFrame)

def test_with_no_model_id_or_sbml_file():
    """
    Test initialization of CopasiModel with no model_id or sbml_file_path.
    """
    try:
        CopasiModel()
    except ValueError as e:
        assert str(e) == "Either model_id or sbml_file_path must be provided."

def test_get_model_metadata():
    """
    Test the get_model_metadata method of the CopasiModel class.
    """
    model = CopasiModel(model_id=64)
    metadata = model.get_model_metadata()
    assert metadata["Model Type"] == "SBML Model (COPASI)"
    assert metadata["Parameter Count"] == len(basico.get_parameters())
