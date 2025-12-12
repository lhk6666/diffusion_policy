from typing import Dict
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

class BaseImageRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseImagePolicy) -> Dict:
        # Default implementation returns empty dict (no simulation environment)
        # Override this method for tasks with actual environments
        return {}
