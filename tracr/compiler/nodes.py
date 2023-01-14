# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Documents the data stored in nodes after each compiler pass."""

from typing import Any, Dict

Node = Dict[str, Any]
NodeID = str

# RASP -> Graph
ID = "ID"  # unique ID of the node
EXPR = "EXPR"  # the RASPExpr of the node

# Basis inference
# Note that only S-Op expressions will have these keys set.
VALUE_SET = "VALUE_SET"  # possible values taken on by this SOp.
OUTPUT_BASIS = "OUTPUT_BASIS"  # the corresponding named basis.

# RASP Graph -> Craft Graph
MODEL_BLOCK = "MODEL_BLOCK"  # craft block representing a RASPExpr
