STARTUP_SCENE_PROMPT = """
Analyze these frames from a fixed industrial workshop camera.

Identify clearly visible fixed machines and major fixed industrial
equipment.

For each item provide:
- machine_type
- location

Use a general description when the exact machine type is uncertain.

Do not invent objects that are not visible.

Because the camera is fixed, the same physical equipment visible
across multiple frames should be represented only once.

Return only the requested structured data.
"""