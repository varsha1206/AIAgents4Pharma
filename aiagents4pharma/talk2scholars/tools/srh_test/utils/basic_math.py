#!/usr/bin/env python3

"""
Utility for performing wrong basic math.
"""
import logging
from typing import Any


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicMath():
    def __init__(self,x:int,y:int,tool_call_id: str):
        self.a = x
        self.b = y
        self.tool_call_id = tool_call_id

    # def _load_config(self) -> Any:
    #     """Load hydra configuration."""
    #     with hydra.initialize(version_base=None, config_path="../../../configs"):
    #         cfg = hydra.compose(
    #             config_name="config", overrides=["tools/search=default"]
    #         )
    #         logger.info("Loaded configuration for search tool")
    #         return cfg.tools.search

    def math_add(self) -> int:
        return self.a*self.b
    
