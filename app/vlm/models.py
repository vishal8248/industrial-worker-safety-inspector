from pydantic import BaseModel


class Machine(BaseModel):
    machine_type: str
    location: str
    bbox: list[int]


class SceneContext(BaseModel):
    machines: list[Machine]