"""Agent labelling (Claude Code over MCP): rendering, mask checks, session, Label Studio payload, path guard."""

from __future__ import annotations

import asyncio
import json

import cv2
import numpy as np
import pytest

from vlm_labeling.agent.label_studio import prediction_result
from vlm_labeling.agent.masks import head_only, shape_of
from vlm_labeling.agent.render import nice_grid, render_view
from vlm_labeling.agent.session import AgentLabelSession

H, W = 400, 600


def disk(cx, cy, r):
    yy, xx = np.ogrid[:H, :W]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def head_with_stalk(cx, cy, r):
    m = disk(cx, cy, r)
    m[cy:cy + 6 * r, cx - r // 3:cx + r // 3] = True     # a stalk hanging below the head
    return m


class FakeSAM:
    """Returns the head (with a stalk for x >= 300) under each point, nothing on the background."""

    heads = [(100, 100, 20), (350, 120, 25)]

    def segment(self, image, points):
        out = []
        for x, y in points:
            hit = next(((cx, cy, r) for cx, cy, r in self.heads if (x - cx) ** 2 + (y - cy) ** 2 <= r * r), None)
            if hit is None:
                out.append(np.zeros((H, W), bool))
            else:
                out.append(head_with_stalk(*hit) if hit[0] >= 300 else disk(*hit))
        return out


@pytest.fixture()
def image(tmp_path):
    img = np.full((H, W, 3), 200, np.uint8)
    for cx, cy, r in FakeSAM.heads:
        cv2.circle(img, (cx, cy), r, (240, 240, 240), -1)
    p = tmp_path / "scope" / "img_001.jpg"
    p.parent.mkdir()
    cv2.imwrite(str(p), img)
    return p


class TestMasks:
    def test_disk_is_round(self):
        assert shape_of(disk(100, 100, 20)).round

    def test_stalk_is_trimmed_to_head(self):
        m, status = head_only(head_with_stalk(350, 120, 25), (350, 120))
        assert status == "trimmed" and shape_of(m).round
        assert m[:, :].sum() < head_with_stalk(350, 120, 25).sum()

    def test_click_outside_mask_is_rejected(self):
        assert head_only(head_with_stalk(350, 120, 25), (10, 10))[1] == "rejected"
        assert head_only(np.zeros((H, W), bool), (10, 10))[1] == "rejected"


class TestRender:
    def test_grid_steps(self):
        assert nice_grid(1280) == 200 and nice_grid(400) == 50 and nice_grid(100) == 10

    def test_zoom_region_and_scale(self):
        img = np.zeros((H, W, 3), np.uint8)
        v = render_view(img, (100, 50, 300, 150))
        assert v.region == (100, 50, 300, 150) and v.scale == 4.0          # zoom capped at 4x
        assert cv2.imdecode(np.frombuffer(v.jpeg, np.uint8), 1).shape[:2] == (400, 800)

    def test_region_outside_image(self):
        with pytest.raises(ValueError):
            render_view(np.zeros((H, W, 3), np.uint8), (590, 390, 700, 500))


class TestSession:
    def test_add_remove_save(self, image):
        s = AgentLabelSession(image, FakeSAM())
        added = s.add([(100, 100), (350, 120), (500, 300)], ["stalked", "sessile", "stalked"])
        assert [h.status for h in added] == ["ok", "trimmed", "rejected"]
        assert abs(added[0].diameter_px - 40) < 3
        assert s.remove([2]) == [2]
        js, txt = s.save("claude-code", "test")
        rec = json.loads(js.read_text())
        assert rec["status"] == "pending_review" and rec["annotator"] == "claude-code"
        assert len(rec["heads"]) == 2 and rec["heads"][0]["polygon"]
        lines = txt.read_text().splitlines()
        assert lines[0].startswith("0 ") and lines[1].startswith("1 ")      # stalked=0, sessile=1
        _cls, cx, cy, w, _h = map(float, lines[0].split())
        assert abs(cx * W - 100) < 2 and abs(cy * H - 100) < 2 and abs(w * W - 41) < 3

    def test_invalid_input(self, image):
        s = AgentLabelSession(image, FakeSAM())
        with pytest.raises(ValueError, match="outside"):
            s.add([(W + 5, 10)])
        with pytest.raises(ValueError, match="unknown kind"):
            s.add([(100, 100)], ["amber"])
        with pytest.raises(FileNotFoundError):
            AgentLabelSession(image.parent / "missing.jpg", FakeSAM())

    def test_label_studio_prediction_is_percent(self, image):
        s = AgentLabelSession(image, FakeSAM())
        s.add([(100, 100)])
        res = prediction_result(s.record("claude-code"))
        v = res[0]["value"]
        assert res[0]["type"] == "rectanglelabels" and v["rectanglelabels"] == ["stalked"]
        assert abs(v["x"] - 100 * 80 / W) < 0.5 and abs(v["width"] - 100 * 41 / W) < 0.6


class TestServer:
    def test_tools_and_path_guard(self, image, monkeypatch):
        pytest.importorskip("mcp")
        monkeypatch.setenv("CTIP_LABEL_ROOT", str(image.parent))
        import apps.mcp.label_server as srv

        names = {t.name for t in asyncio.run(srv.server.list_tools())}
        assert {"open_image", "view", "add_heads", "remove_heads", "save", "push_to_label_studio"} <= names
        assert srv._resolve("img_001.jpg") == image.resolve()
        with pytest.raises(ValueError, match="outside the label root"):
            srv._resolve("../../etc/passwd")
        with pytest.raises(FileNotFoundError):
            srv._resolve("nope.jpg")
