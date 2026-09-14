STARTUP_SCENE_PROMPT = """
Analyze these frames from a fixed industrial workshop camera.

Identify clearly visible fixed machines and major fixed industrial
equipment.

For each distinct machine provide:

- machine_type
- location
- points

The points are VERY IMPORTANT.

Provide 3 to 5 points that are clearly INSIDE THE ACTUAL MACHINE BODY.

These points will be used as positive foreground prompts for an image
segmentation model.

A point must lie on the physical machine itself.

Do NOT place points on:

- floor
- walls
- safety cages
- guardrails
- workers
- pipes
- cables
- nearby equipment
- background objects

Spread the points across different visible parts of the same machine.

For example, if a large machine has a left body, center body, and right
body, place points inside those three visible machine areas.

Coordinates must use the ORIGINAL IMAGE coordinate system.

Each point must be:

[x, y]

where:

x = horizontal pixel coordinate
y = vertical pixel coordinate

The same physical machine visible across multiple frames should be
listed only once.

Use the most specific machine description that can reasonably be
determined from the images.

Examples:

- hydraulic press
- industrial press
- metalworking machine
- cutting machine
- drilling machine
- welding machine
- manufacturing machine

Do not include workers as machines.

Do not provide bounding boxes.

Return only the requested structured data.
"""