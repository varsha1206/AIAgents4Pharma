'''
This file contains the unit tests for the BioModel class.
'''

from typing import List, Dict, Union
from ..models.biomodel import BioModel

class TestBioModel(BioModel):
    '''
    A test BioModel class for unit testing.
    '''
    def __init__(self, model_id: str, name: str, description: str):
        super().__init__(model_id, name, description)
        self.model_id = model_id
        self.name = name
        self.description = description

    def get_model_metadata(self) -> Dict[str, Union[str, int]]:
        return {"author": "John Doe", "year": 2021}

    def simulate(self,
                 parameters: Dict[str, Union[float, int]],
                 duration: Union[int, float]) -> List[float]:
        param1 = parameters.get('param1', 0.0)
        param2 = parameters.get('param2', 0.0)
        return [param1 + param2 * t for t in range(int(duration))]

def test_get_model_metadata():
    '''
    Test the get_model_metadata method of the BioModel class.
    '''
    model = TestBioModel(model_id="123", name="Test Model", description="A test model")
    metadata = model.get_model_metadata()
    assert metadata["author"] == "John Doe"
    assert metadata["year"] == 2021

def test_simulate():
    '''
    Test the simulate method of the BioModel class.
    '''
    model = TestBioModel(model_id="123", name="Test Model", description="A test model")
    results = model.simulate(parameters={'param1': 1.0, 'param2': 2.0}, duration=4.0)
    assert results == [1.0, 3.0, 5.0, 7.0]
