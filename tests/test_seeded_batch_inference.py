import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from evaluate_dp_dataset import predict_trajectory, predict_trajectory_batch


class _RandomActionPolicy:
    """Small policy double exercising the same generator-list contract."""

    def __init__(self) -> None:
        self.kwargs = {}
        self.observed_images = []

    def predict_action(self, obs_dict):
        image = obs_dict["image"]
        self.observed_images.append(image.detach().cpu().clone())
        batch_size = int(image.shape[0])
        generator = self.kwargs.get("generator")
        initial = randn_tensor(
            (batch_size, 4, 2),
            generator=generator,
            device=image.device,
            dtype=image.dtype,
        )
        step_noise = randn_tensor(
            (batch_size, 4, 2),
            generator=generator,
            device=image.device,
            dtype=image.dtype,
        )
        return {"action": initial + 0.125 * step_noise}


def test_seeded_batch_matches_independent_serial_rng_streams_on_cpu() -> None:
    images = [
        np.full((3, 5, 3), 64, dtype=np.uint8),
        np.full((3, 5, 3), 192, dtype=np.uint8),
    ]
    starts = [
        np.asarray([0.1, 0.2], dtype=np.float32),
        np.asarray([0.3, 0.4], dtype=np.float32),
    ]
    instructions = ["first", "second"]
    seeds = [1123, 9981]

    serial_policy = _RandomActionPolicy()
    serial = [
        predict_trajectory(
            serial_policy,
            image,
            start,
            instruction=instruction,
            device="cpu",
            noise_seed=seed,
            action_definition="positions",
        )
        for image, start, instruction, seed in zip(
            images, starts, instructions, seeds
        )
    ]

    batch_policy = _RandomActionPolicy()
    batched = predict_trajectory_batch(
        batch_policy,
        images,
        starts,
        instructions,
        device="cpu",
        noise_seeds=seeds,
        action_definition="positions",
    )

    for serial_trajectory, batched_trajectory in zip(serial, batched):
        torch.testing.assert_close(
            torch.from_numpy(batched_trajectory),
            torch.from_numpy(serial_trajectory),
            rtol=0.0,
            atol=0.0,
        )

    expected_images = torch.stack(
        [
            torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            for image in images
        ]
    ).unsqueeze(1)
    torch.testing.assert_close(
        batch_policy.observed_images[0],
        expected_images,
        rtol=0.0,
        atol=0.0,
    )
    assert "generator" not in batch_policy.kwargs


def test_seeded_batch_restores_preexisting_policy_generator() -> None:
    policy = _RandomActionPolicy()
    previous_generator = torch.Generator(device="cpu").manual_seed(7)
    policy.kwargs["generator"] = previous_generator

    predict_trajectory_batch(
        policy,
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        [np.asarray([0.0, 0.0], dtype=np.float32)],
        ["test"],
        device="cpu",
        noise_seeds=[123],
        action_definition="positions",
    )

    assert policy.kwargs["generator"] is previous_generator
