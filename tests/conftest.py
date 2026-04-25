"""
Shared pytest fixtures for DataClean-Env tests.
"""
from __future__ import annotations

import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataclean.env import DataCleanEnv  # noqa: E402
from dataclean.tasks import TASK_REGISTRY  # noqa: E402


@pytest.fixture
def env() -> DataCleanEnv:
    return DataCleanEnv()


@pytest.fixture(params=list(TASK_REGISTRY.keys()))
def task_id(request) -> str:
    return request.param
