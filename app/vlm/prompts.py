STARTUP_SCENE_PROMPT = """
You are analyzing the first few frames from a fixed industrial
workshop camera.

Identify only clearly visible fixed industrial machines or major
fixed equipment.

For each machine provide:
- machine_id
- machine_type
- approximate_location_in_frame
- static

Do not invent machines that are not clearly visible.

The camera is fixed, so machines that are clearly visible should
normally remain in the same scene location during this video.

Return the result using the provided JSON schema.
"""