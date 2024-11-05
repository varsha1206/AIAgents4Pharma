#!/usr/bin/env python3

"""
CopasiModel class for loading and simulating SBML models 
using the basico package.
"""

from typing import Dict, Union, Optional
import basico
import pandas as pd
from .biomodel import BioModel

class CopasiModel(BioModel):
    """
    Model that loads and simulates SBML models using the basico package.
    Can load models from an SBML file or download them using a BioModels model_id.
    """

    def __init__(self,
                 model_id: str = None,
                 name: str = "",
                 description: str = "",
                 sbml_file_path: str = None):
        """
        Initialize the CopasiModel instance.
        
        Args:
            model_id: BioModels model ID to download and load.
            name: Name of the model.
            description: Description of the model.
            sbml_file_path: Path to an SBML file to load
        
        Raises:
            ValueError: If no model_id or sbml_file_path is provided.
        """
        # If model_id is given but no SBML file path, download the SBML file
        if model_id and not sbml_file_path:
            self.model_id = model_id
            self.copasi_model = basico.load_biomodel(model_id)
            self.description = basico.biomodels.get_model_info(self.model_id)["description"]
        # Initialize the model with downloaded or provided SBML file
        elif sbml_file_path:
            super().__init__(model_id, name, description)
            self.sbml_file_path = sbml_file_path
            self.model_id = ""
            self.copasi_model = basico.load_model(sbml_file_path)
        else:
            raise ValueError("Either model_id or sbml_file_path must be provided.")
        self.simulation_results = None

    def get_model_metadata(self) -> Dict[str, Union[str, int]]:
        """
        Retrieve metadata specific to the COPASI model.
        
        Returns:
            Dictionary of model metadata.
        """
        return {
            "Model Type": "SBML Model (COPASI)",
            "Parameter Count": len(basico.get_parameters())
        }

    def simulate(self,
                 parameters: Optional[Dict[str, Union[float, int]]] = None,
                 duration: Union[int, float] = 10,
                 interval: int = 10
                 ) -> pd.DataFrame:
        """
        Simulate the COPASI model over a specified range of time points.
        
        Args:
            parameters: Dictionary of model parameters to update before simulation.
            duration: Duration of the simulation in time units.
            interval: Interval between time points in the simulation.
        
        Returns:
            Pandas DataFrame with time-course simulation results.
        """

        # Update parameters in the model
        if parameters:
            for param_name, param_value in parameters.items():
                # check if the param_name is not None
                if param_name is None:
                    continue
                basico.model_info.set_species(name=param_name,
                                            exact=True,
                                            initial_concentration=param_value,
                                            model=self.copasi_model)

        # Run the simulation and return results
        df_result = basico.run_time_course(model=self.copasi_model,
                                        intervals=interval,
                                        duration=duration)
        df_result.columns = df_result.columns.str.replace('{', '[', regex=False).\
                    str.replace('}', ']', regex=False)
        df_result.reset_index(inplace=True)
        self.simulation_results = df_result
        return df_result.copy()
