import base64
import json
import os

import cv2
from groq import Groq

from app.vlm.prompts import STARTUP_SCENE_PROMPT


MACHINE_SCHEMA = {
    "type": "object",
    "properties": {
        "machines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "machine_id": {
                        "type": "string"
                    },
                    "machine_type": {
                        "type": "string"
                    },
                    "approximate_location_in_frame": {
                        "type": "string"
                    },
                    "static": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "machine_id",
                    "machine_type",
                    "approximate_location_in_frame",
                    "static"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "machines"
    ],
    "additionalProperties": False
}


class VLMAnalyzer:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

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

    def analyze_startup_frames(self, frames):
        if not frames:
            return {
                "machines": []
            }

        content = [
            {
                "type": "text",
                "text": STARTUP_SCENE_PROMPT,
            }
        ]

        for item in frames[:3]:
            success, encoded = cv2.imencode(
                ".jpg",
                item["frame"],
            )

            if not success:
                continue

            image_base64 = base64.b64encode(
                encoded.tobytes()
            ).decode("utf-8")

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/jpeg;base64,"
                            f"{image_base64}"
                        )
                    },
                }
            )

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
                    "name": "machine_scene",
                    "strict": True,
                    "schema": MACHINE_SCHEMA,
                },
            },
            reasoning_effort="none",
            temperature=0.7,
            max_completion_tokens=1000,
        )

        result = response.choices[0].message.content

        return json.loads(result)