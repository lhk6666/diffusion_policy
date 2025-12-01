from typing import Dict
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class VLARunner(BaseImageRunner):
    def __init__(self, 
                 output_dir, 
                 n_train=0,
                 n_train_vis=0,
                 n_test=0,
                 n_test_vis=0,
                 test_start_seed=0,
                 max_steps=100,
                 n_obs_steps=2,
                 n_action_steps=8,
                 fps=10,
                 past_action=False,
                 n_envs=None
                 ):
        super().__init__(output_dir)
        # Swallow arguments
        
    def run(self, policy: BaseImagePolicy) -> Dict:
        # No simulator available for VLA navigation in this context
        # Just return empty dict
        return {}
