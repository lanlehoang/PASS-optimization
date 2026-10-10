"""
Defines Pydantic models to validate and represent the state of neighbour satellites
"""

from typing import List, ClassVar
from pydantic import BaseModel, Field
import numpy as np
from src.utils.get_config import get_system_config

N_DEVICES = get_system_config()["K"]


class EnvironmentState(BaseModel):
    """
    Concatenation of exactly N_DEVICES NeighbourState instances.
    """

    STATE_DIM: ClassVar[int] = N_DEVICES * 3 + 1  # 3 features per device + 1 for inherited PASS