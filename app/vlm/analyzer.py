import base64
import json
import os

import cv2
from groq import Groq

from app.vlm.models import SceneContext
from app.vlm.prompts import STARTUP_SCENE_PROMPT


MACHINE_SCHEMA = {
    "type": "object",
    "properties": {
        "machines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "machine_type": {
                        "type": "string"
                    },
                    "location": {
                        "type": "string"
                    },
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "integer"
                            },
                            "minItems": 2,
                            "maxItems": 2
                        },
                        "minItems": 3,
                        "maxItems": 5
                    }
                },
                "required": [
                    "machine_type",
                    "location",
                    "points"
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
            return SceneContext(
                machines=[]
            )

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
            temperature=0.1,
            max_completion_tokens=900,
        )

        result = response.choices[0].message.content

        return SceneContext.model_validate(
            json.loads(result)
        )