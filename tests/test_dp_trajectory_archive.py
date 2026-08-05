import json
from pathlib import Path

import numpy as np
import pytest

from evaluate_dp_dataset import (
    TRAJECTORY_COORDINATE_CONVENTION,
    TRAJECTORY_ARCHIVE_FILENAME,
    TRAJECTORY_ARCHIVE_INCLUDES_START,
    TRAJECTORY_ARCHIVE_SCHEMA_VERSION,
    build_evaluation_config,
    build_trajectory_archive_payload,
    save_trajectory_archive,
)


def _episode_meta():
    return [
        {
            "sample_id": "sample_0",
            "scene_id": "scene_0",
            "start": [0.10, 0.20],
        },
        {
            "sample_id": "sample_1",
            "scene_id": "scene_1",
            "start": [0.30, 0.40],
        },
    ]


def test_archive_prepends_start_and_nan_pads(tmp_path: Path) -> None:
    results = [
        {
            "episode_idx": 1,
            "sample_id": "sample_1",
            "scene_id": "scene_1",
        },
        {
            "episode_idx": 0,
            "sample_id": "sample_0",
            "scene_id": "scene_0",
            "error": "inference failed",
        },
    ]
    predictions = [
        np.asarray([[0.31, 0.41], [0.32, 0.42]], dtype=np.float64),
        np.asarray([[0.11, 0.21]], dtype=np.float32),
    ]

    archive_path = tmp_path / "trajectory_archive.npz"
    save_trajectory_archive(
        archive_path,
        results=results,
        pred_trajectories=predictions,
        episode_meta=_episode_meta(),
    )

    with np.load(archive_path, allow_pickle=False) as archive:
        assert archive["dataset_index"].tolist() == [1, 0]
        assert archive["sample_id"].tolist() == ["sample_1", "sample_0"]
        assert archive["scene_id"].tolist() == ["scene_1", "scene_0"]
        assert archive["success"].tolist() == [True, False]
        assert archive["error"].tolist() == ["", "inference failed"]
        assert archive["trajectory_length"].tolist() == [3, 2]
        assert archive["trajectory_xy_norm"].dtype == np.float32
        np.testing.assert_allclose(
            archive["trajectory_xy_norm"][0, :3],
            [[0.30, 0.40], [0.31, 0.41], [0.32, 0.42]],
        )
        np.testing.assert_allclose(
            archive["trajectory_xy_norm"][1, :2],
            [[0.10, 0.20], [0.11, 0.21]],
        )
        assert np.isnan(archive["trajectory_xy_norm"][1, 2]).all()
        np.testing.assert_allclose(
            archive["trajectory_xy_norm"][:, 0], archive["start_xy_norm"]
        )
        assert bool(archive["trajectory_includes_start"])
        assert str(archive["coordinate_convention"]) == (
            TRAJECTORY_COORDINATE_CONVENTION
        )


def test_archive_rejects_duplicate_dataset_indices() -> None:
    results = [
        {"episode_idx": 0, "sample_id": "sample_0", "scene_id": "scene_0"},
        {"episode_idx": 0, "sample_id": "sample_0", "scene_id": "scene_0"},
    ]
    predictions = [np.zeros((1, 2), dtype=np.float32)] * 2

    with pytest.raises(ValueError, match="dataset_index values must be unique"):
        build_trajectory_archive_payload(results, predictions, _episode_meta())


def test_archive_rejects_metadata_mismatch() -> None:
    results = [
        {"episode_idx": 0, "sample_id": "wrong", "scene_id": "scene_0"},
    ]

    with pytest.raises(ValueError, match="sample_id mismatch"):
        build_trajectory_archive_payload(
            results,
            [np.zeros((1, 2), dtype=np.float32)],
            _episode_meta(),
        )


def test_evaluation_config_records_requested_and_resolved_metadata() -> None:
    config = build_evaluation_config(
        checkpoint="model.ckpt",
        dataset="dataset/val",
        num_episodes=9015,
        use_rollout=False,
        seed=3,
        device="cuda:1",
        batch_size=128,
        num_inference_steps_requested=5,
        num_inference_steps_actual=5,
        k=None,
        visualize_every=0,
        action_definition_requested="auto",
        action_definition_resolved="delta_normed_anchor",
        action_delta_anchor_resolved=0.004617704774695214,
        action_eps_resolved=1e-6,
    )

    assert config["device"] == "cuda:1"
    assert config["batch_size"] == 128
    assert config["num_inference_steps_requested"] == 5
    assert config["num_inference_steps_actual"] == 5
    assert config["k"] is None
    assert config["visualize_every"] == 0
    assert config["action_definition_requested"] == "auto"
    assert config["action_definition_resolved"] == "delta_normed_anchor"
    assert config["action_delta_anchor_resolved"] == pytest.approx(
        0.004617704774695214
    )
    assert config["action_eps_resolved"] == pytest.approx(1e-6)
    assert config["action_scale_resolved"] == pytest.approx(
        0.004618704774695214
    )
    assert config["trajectory_archive"] == {
        "filename": TRAJECTORY_ARCHIVE_FILENAME,
        "schema_version": TRAJECTORY_ARCHIVE_SCHEMA_VERSION,
        "includes_start": TRAJECTORY_ARCHIVE_INCLUDES_START,
        "coordinate_convention": TRAJECTORY_COORDINATE_CONVENTION,
    }
    json.dumps(config)
