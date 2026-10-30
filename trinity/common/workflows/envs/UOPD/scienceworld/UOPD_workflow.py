# -*- coding: utf-8 -*-
"""UOPD workflow on ScienceWorld: UOPD's algorithm exactly as in ../alfworld/UOPD_workflow.py (teacher
confidence, adaptive quantile threshold, decaying intervention budget, teacher-executed turns trained
with SFT), run on ScienceWorld games through OnPolicyDistillScienceWorldWorkflow. Not in the UOPD
release (ALFWorld and WebShop only); ported here.

No UOPD code is copied: the class combines the ALFWorld UOPD class (all of the algorithm, via the
OPD workflow's hooks) with the ScienceWorld OPD workflow (the environment layer). MRO:
UOPDScienceWorldWorkflow -> UOPDAlfworldWorkflow -> OnPolicyDistillScienceWorldWorkflow ->
OnPolicyDistillVerlAgentAlfworldWorkflow, so UOPDAlfworldWorkflow.run_async's super() call reaches the
ScienceWorld episode loop.
"""

from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.UOPD.alfworld.UOPD_workflow import UOPDAlfworldWorkflow
from trinity.common.workflows.envs.UOPD.scienceworld.OPD_workflow import (
    OnPolicyDistillScienceWorldWorkflow,
)


@WORKFLOWS.register_module("UOPD_scienceworld_workflow")
class UOPDScienceWorldWorkflow(UOPDAlfworldWorkflow, OnPolicyDistillScienceWorldWorkflow):
    """UOPD (ALFWorld algorithm, unchanged) on ScienceWorld."""
