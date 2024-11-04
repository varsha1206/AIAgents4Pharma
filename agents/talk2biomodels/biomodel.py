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
        
        :param model_id: Unique identifier for the model in the BioModels repository.
        :param name: Name of the model.
        :param description: Optional description of the model.
        """
        self.model_id = model_id
        self.name = name
        self.description = description

    @abstractmethod
    def get_model_metadata(self) -> Dict[str, Union[str, int]]:
        """
        Abstract method to retrieve metadata of the model.
        This method should return a dictionary containing model metadata.

        :return: Dictionary of model metadata such as authors, publication year, etc.
        """
        pass

    @abstractmethod
    def simulate(self,
                 parameters: Dict[str, Union[float, int]],
                 time_points: List[float]) -> List[float]:
        """
        Abstract method to run a simulation of the model.
        This method should be implemented to simulate model 
        behavior based on the provided parameters.

        :param parameters: Dictionary of parameters for the model simulation.
        :param time_points: List of time points to simulate.
        :return: List of simulation results at each time point.
        """
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """
        Abstract method to validate the model structure or content.
        This method should ensure the model complies with certain standards or formats.
        
        :return: Boolean indicating if the model is valid.
        """
        pass

    # def __repr__(self) -> str:
    #     """
    #     String representation of the BioModel instance.
        
    #     :return: String representation with model ID and name.
    #     """
    #     return f"BioModel(model_id='{self.model_id}', name='{self.name}')"

    def get_basic_info(self) -> Dict[str, str]:
        """
        Returns basic information about the model.
        
        :return: Dictionary with model ID, name, and description.
        """
        return {
            "Model ID": self.model_id,
            "Name": self.name,
            "Description": self.description
        }
