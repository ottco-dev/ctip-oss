"""Push an agent's labels to Label Studio as *predictions* - a person accepts or corrects them there."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests

from vlm_labeling.agent.session import LabelRecord

MODEL_VERSION = "claude-code-agent"


def prediction_result(record: LabelRecord, from_name: str = "label", to_name: str = "image") -> list[dict]:
    """Label Studio rectangle results (percent coordinates) for the CTIP trichome project template."""
    out = []
    for h in record.heads:
        x0, y0, x1, y1 = h.box
        out.append({
            "id": f"agent-{h.id}", "from_name": from_name, "to_name": to_name, "type": "rectanglelabels",
            "original_width": record.width, "original_height": record.height, "image_rotation": 0,
            "value": {"x": 100 * x0 / record.width, "y": 100 * y0 / record.height,
                      "width": 100 * (x1 - x0) / record.width, "height": 100 * (y1 - y0) / record.height,
                      "rotation": 0, "rectanglelabels": [h.kind]},
        })
    return out


@dataclass
class LabelStudio:
    url: str
    token: str
    timeout: float = 30.0

    def _h(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.token}"}

    def project(self, title: str, label_config: str) -> int:
        """Id of the project with this title; created with the CTIP label config if missing."""
        r = requests.get(f"{self.url}/api/projects", headers=self._h(), params={"page_size": 200}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        for p in data.get("results", data) if isinstance(data, dict) else data:
            if p.get("title") == title:
                return int(p["id"])
        r = requests.post(f"{self.url}/api/projects", headers=self._h(), timeout=self.timeout,
                          json={"title": title, "label_config": label_config,
                                "description": "Pre-labels by an AI agent (Claude Code + SAM2) - review every task."})
        r.raise_for_status()
        return int(r.json()["id"])

    def push(self, project_id: int, image: Path, record: LabelRecord, score: float | None = None) -> int:
        """Upload the image as a task and attach the agent's labels as a prediction. Returns the task id."""
        with image.open("rb") as fh:
            r = requests.post(f"{self.url}/api/projects/{project_id}/import", headers=self._h(), timeout=self.timeout,
                              files={"file": (image.name, fh)}, params={"return_task_ids": "true"})
        r.raise_for_status()
        task_id = int(r.json()["task_ids"][0])
        body = {"task": task_id, "model_version": MODEL_VERSION, "result": prediction_result(record)}
        if score is not None:
            body["score"] = score
        r = requests.post(f"{self.url}/api/predictions", headers=self._h(), json=body, timeout=self.timeout)
        r.raise_for_status()
        return task_id
