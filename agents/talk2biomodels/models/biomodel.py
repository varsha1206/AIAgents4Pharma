'''
An abstract base class for BioModels in the BioModels repository.
'''

from abc import ABC, abstractmethod
from typing import List, Dict, Union

class BioModel(ABC):
    """
    Abstract base class for BioModels in the BioModels repository.
    This class serves as a general structure for models, allowing
    different mathematical approaches to be implemented in subclasses.
    """

    def __init__(self, model_id: str, name: str, description: str = ""):
        """
        Initialize the BioModel instance.
        
        Args:
            model_id: BioModel ID of the model.
            name: Name of the model.
            description: Description of the model.

        Returns:
            None
        """
        self.model_id = model_id
        self.name = name
        self.description = description

    @abstractmethod
    def get_model_metadata(self) -> Dict[str, Union[str, int]]:
        """
        Abstract method to retrieve metadata of the model.
        This method should return a dictionary containing model metadata.

        Returns:
            dict: Dictionary with model metadata
        """

    @abstractmethod
    def simulate(self,
                 parameters: Dict[str, Union[float, int]],
                 duration: Union[int, float]) -> List[float]:
        """
        Abstract method to run a simulation of the model.
        This method should be implemented to simulate model 
        behavior based on the provided parameters.

        Args:
            parameters: Dictionary of parameter values.
            duration: Duration of the simulation.

        Returns:
            list: List of simulation results.
        """
