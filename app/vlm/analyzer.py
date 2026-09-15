import base64
import json
import os

import cv2
from groq import Groq

from app.vlm.models import IncidentAnalysis
from app.vlm.prompts import INCIDENT_ANALYSIS_PROMPT


OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "phone_usage": {
            "type": "boolean"
        },
        "ppe_violation": {
            "type": "boolean"
        },
        "machine_interaction": {
            "type": "boolean"
        },
        "unsafe_position": {
            "type": "boolean"
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    },
    "required": [
        "phone_usage",
        "ppe_violation",
        "machine_interaction",
        "unsafe_position",
        "evidence"
    ],
    "additionalProperties": False
}


class VLMAnalyzer:
    def __init__(self):
        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = os.getenv(
            "GROQ_VLM_MODEL",
            "qwen/qwen3.8-27b",
        )

    def _crop_worker(
        self,
        frame,
        box,
    ):
        height, width = frame.shape[:2]

        x1, y1, x2, y2 = box

        worker_width = x2 - x1
        worker_height = y2 - y1

        padding_x = int(
            worker_width * 0.8
        )

        padding_y = int(
            worker_height * 0.5
        )

        crop_x1 = max(
            0,
            x1 - padding_x,
        )

        crop_y1 = max(
            0,
            y1 - padding_y,
        )

        crop_x2 = min(
            width,
            x2 + padding_x,
        )

        crop_y2 = min(
            height,
            y2 + padding_y,
        )

        crop = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]

        if crop.size == 0:
            raise RuntimeError(
                "Worker crop is empty."
            )

        return crop

    def analyze_worker(
        self,
        frame,
        box,
        worker_id,
        timestamp,
    ):
        crop = self._crop_worker(
            frame,
            box,
        )

        success, encoded = cv2.imencode(
            ".jpg",
            crop,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                90,
            ],
        )

        if not success:
            raise RuntimeError(
                "Failed to encode worker crop."
            )

        image_base64 = base64.b64encode(
            encoded.tobytes()
        ).decode("utf-8")

        content = [
            {
                "type": "text",
                "text": INCIDENT_ANALYSIS_PROMPT,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        f"{image_base64}"
                    )
                },
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "safety_observations",
                    "strict": True,
                    "schema": OBSERVATION_SCHEMA,
                },
            },
            reasoning_effort="none",
            temperature=0.1,
            max_completion_tokens=400,
        )

        result = response.choices[0].message.content

        observations = json.loads(
            result
        )

        return IncidentAnalysis(
            worker_id=worker_id,
            timestamp=timestamp,
            observations=observations,
        )