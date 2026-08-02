import sys
import os
from unittest.mock import MagicMock

# Add the repo root to sys.path so test files can import remote_workflow_node directly.
sys.path.insert(0, os.path.dirname(__file__))

# Mock ComfyUI runtime modules and optional dependencies before any test imports the node.
sys.modules.setdefault("folder_paths", MagicMock())
sys.modules.setdefault("websocket", MagicMock())
