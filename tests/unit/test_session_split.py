"""Session-grouped dataset splits: no imaging session may appear in more than one split."""

from __future__ import annotations

import pytest

from training.pipelines.session_split import session_key, split_by_session


def _task(i: int, image: str, **data) -> dict:
    return {"id": i, "data": {"image": image, **data}}


def _sessions(n_sessions: int, per_session: int) -> list[dict]:
    tasks, i = [], 0
    for s in range(n_sessions):
        for f in range(per_session):
            tasks.append(_task(i, f"/data/local-files/?d=microscope/plant{s}_slide1/frame_{f:04d}.jpg"))
            i += 1
    return tasks


class TestSessionKey:
    def test_explicit_session_field_wins(self):
        assert session_key(_task(1, "/data/upload/1/a.jpg", session="2026-09-30-A")) == ("session:2026-09-30-A", False)

    def test_meta_session(self):
        assert session_key({"id": 1, "data": {"image": "x.jpg"}, "meta": {"session_id": 7}}) == ("session:7", False)

    def test_parent_directory_from_local_files_url(self):
        key, alone = session_key(_task(1, "/data/local-files/?d=scope%2Fplant3%2Fimg_001.jpg"))
        assert key == "dir:scope/plant3" and not alone

    def test_generic_upload_folder_uses_file_series(self):
        key, alone = session_key(_task(1, "/data/upload/12/1a2b3c4d-PlantA_slide2_0042.jpg"))
        assert key == "name:planta_slide2" and not alone

    def test_frame_variants_share_a_key(self):
        keys = {session_key(_task(i, f"/data/upload/1/{name}"))[0] for i, name in
                enumerate(["run7_frame12.png", "run7_frame13.png", "run7 (2).png", "run7-0099.png"])}
        assert keys == {"name:run7"}

    def test_no_information_is_reported(self):
        assert session_key(_task(5, "/data/upload/1/photo.jpg")) == ("task:5", True)


class TestSplitBySession:
    def test_no_session_in_two_splits(self):
        result = split_by_session(_sessions(10, 7), 0.7, 0.15, seed=42)
        seen = {}
        for split, tasks in result.splits.items():
            for t in tasks:
                key = session_key(t)[0]
                assert seen.setdefault(key, split) == split, f"{key} leaks between {seen[key]} and {split}"
        assert sum(len(v) for v in result.splits.values()) == 70
        assert not result.warnings

    def test_every_split_gets_a_session_with_three_sessions(self):
        result = split_by_session(_sessions(3, 20), 0.9, 0.05, seed=1)
        assert all(len(result.sessions[s]) == 1 for s in ("train", "val", "test"))

    def test_ratios_are_approached(self):
        result = split_by_session(_sessions(40, 5), 0.7, 0.15, seed=42)
        assert 120 <= len(result.splits["train"]) <= 150
        assert len(result.splits["val"]) >= 20 and len(result.splits["test"]) >= 20

    def test_deterministic_for_a_seed(self):
        a = split_by_session(_sessions(12, 4), 0.7, 0.15, seed=42)
        b = split_by_session(list(reversed(_sessions(12, 4))), 0.7, 0.15, seed=42)
        assert a.sessions == b.sessions

    def test_seed_changes_assignment(self):
        a = split_by_session(_sessions(12, 4), 0.7, 0.15, seed=1)
        b = split_by_session(_sessions(12, 4), 0.7, 0.15, seed=2)
        assert a.sessions != b.sessions

    def test_too_few_sessions_warns(self):
        result = split_by_session(_sessions(2, 10), 0.7, 0.15)
        assert any("Only 2 session" in w for w in result.warnings)

    def test_images_without_session_info_warn(self):
        tasks = [_task(i, f"/data/upload/1/photo{c}.jpg") for i, c in enumerate("abcdef")]
        result = split_by_session(tasks, 0.7, 0.15)
        assert result.fallback_sessions == 6
        assert any("no session information" in w for w in result.warnings)

    @pytest.mark.parametrize("train,val", [(0, 0.1), (1.0, 0), (0.8, 0.3), (0.7, -0.1)])
    def test_invalid_ratios(self, train, val):
        with pytest.raises(ValueError):
            split_by_session(_sessions(3, 2), train, val)
