#!/usr/bin/env python3

"""
Function for loading the BioModel.
"""

from dataclasses import dataclass
from ..models.basico_model import BasicoModel

@dataclass
class ModelData:
    """
    Dataclass for storing the model data.
    """
    model_id: int = None
    # sbml_file_path: Optional[str] = None
    use_uploaded_sbml_file: bool = False

def load_biomodel(sys_bio_model, sbml_file_path=None):
    """
    Load the BioModel.
    """
    model_object = None
    if sys_bio_model.model_id:
        model_object = BasicoModel(model_id=sys_bio_model.model_id)
    elif sys_bio_model.use_uploaded_sbml_file:
        model_object = BasicoModel(sbml_file_path=sbml_file_path)
    return model_object
    # return None
