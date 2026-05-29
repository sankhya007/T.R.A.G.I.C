from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np


class AgentState(Enum):
    CALM = auto()
    AWARE = auto()
    PANICKING = auto()
    EVACUATING = auto()
    DEAD = auto()
    SAFE = auto()


class AgentProfile(Enum):
    ADULT = auto()
    CHILD = auto()
    ELDERLY = auto()
    MOBILITY_IMPAIRED = auto()


# (speed_calm, speed_panicked, panic_sensitivity, shoulder_radius)
PROFILE_STATS: dict[AgentProfile, dict] = {
    AgentProfile.ADULT:            {"speed_calm": 1.2, "speed_panicked": 3.5, "panic_sensitivity": 0.6, "shoulder_radius": 0.27},
    AgentProfile.CHILD:            {"speed_calm": 1.0, "speed_panicked": 2.5, "panic_sensitivity": 0.8, "shoulder_radius": 0.20},
    AgentProfile.ELDERLY:          {"speed_calm": 0.8, "speed_panicked": 1.5, "panic_sensitivity": 0.7, "shoulder_radius": 0.27},
    AgentProfile.MOBILITY_IMPAIRED:{"speed_calm": 0.5, "speed_panicked": 1.0, "panic_sensitivity": 0.9, "shoulder_radius": 0.30},
}


@dataclass
class Agent:
    id: int
    pos: np.ndarray                         # (x, y) float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    facing: float = 0.0                     # angle in radians
    panic: float = 0.0                      # 0.0 calm -> 1.0 full panic
    memory: dict = field(default_factory=dict)   # {exit_id: (x,y), hazard_id: (x,y)}
    goal: tuple | None = None               # current target exit coordinate
    state: AgentState = AgentState.CALM
    profile: AgentProfile = AgentProfile.ADULT
    speed_max: float = 1.2                  # set from profile at spawn time
    shoulder_radius: float = 0.27

    @property
    def fov_deg(self) -> float:
        """FOV narrows from 120° to 60° as panic rises."""
        return 120.0 - 60.0 * self.panic

    @property
    def fov_rad(self) -> float:
        return np.deg2rad(self.fov_deg)

    def current_speed_max(self) -> float:
        """Interpolate between calm and panic speed."""
        s = PROFILE_STATS[self.profile]
        return s["speed_calm"] + self.panic * (s["speed_panicked"] - s["speed_calm"])

    def update_panic(self, hazard_proximity: float, dt: float) -> None:
        """
        hazard_proximity: 0.0 = no hazard nearby, 1.0 = hazard right on top.
        """
        sensitivity = PROFILE_STATS[self.profile]["panic_sensitivity"]
        if hazard_proximity > 0:
            self.panic = min(1.0, self.panic + sensitivity * hazard_proximity * dt)
        else:
            self.panic = max(0.0, self.panic - 0.05 * dt)  # slow calm-down

        if self.state not in (AgentState.DEAD, AgentState.SAFE):
            if self.panic > 0.7:
                self.state = AgentState.PANICKING
            elif self.panic > 0.2:
                self.state = AgentState.AWARE
            elif self.goal is not None:
                self.state = AgentState.EVACUATING
            else:
                self.state = AgentState.CALM

    def can_see(self, target_pos: np.ndarray) -> bool:
        """Is target_pos inside this agent's FOV cone?"""
        diff = target_pos - self.pos
        if np.linalg.norm(diff) < 1e-6:
            return True
        angle_to_target = np.arctan2(diff[1], diff[0])
        angle_diff = abs(np.arctan2(
            np.sin(angle_to_target - self.facing),
            np.cos(angle_to_target - self.facing)
        ))
        return angle_diff <= self.fov_rad / 2