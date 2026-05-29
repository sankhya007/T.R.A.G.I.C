from __future__ import annotations
import json
import random
import numpy as np
from shapely.geometry import Point, Polygon

from agents import Agent, AgentProfile, AgentState, PROFILE_STATS


def poisson_disk_sample(
    polygon: Polygon,
    min_dist: float,
    max_attempts: int = 30,
    rng: random.Random | None = None,
) -> list[np.ndarray]:
    """Bridson's Poisson disk sampling inside a Shapely polygon."""
    if rng is None:
        rng = random.Random()

    minx, miny, maxx, maxy = polygon.bounds
    cell = min_dist / np.sqrt(2)
    grid: dict[tuple[int, int], np.ndarray] = {}

    def to_grid(p):
        return int((p[0] - minx) / cell), int((p[1] - miny) / cell)

    def too_close(p):
        gx, gy = to_grid(p)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nb = grid.get((gx + dx, gy + dy))
                if nb is not None and np.linalg.norm(p - nb) < min_dist:
                    return True
        return False

    def rand_in_polygon():
        for _ in range(100):
            p = np.array([rng.uniform(minx, maxx), rng.uniform(miny, maxy)])
            if polygon.contains(Point(p)):
                return p
        return None

    first = rand_in_polygon()
    if first is None:
        return []
    result, active = [first], [first]
    grid[to_grid(first)] = first

    while active:
        idx = rng.randint(0, len(active) - 1)
        base = active[idx]
        placed = False
        for _ in range(max_attempts):
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(min_dist, 2 * min_dist)
            c = base + np.array([np.cos(angle), np.sin(angle)]) * r
            if polygon.contains(Point(c)) and not too_close(c):
                result.append(c)
                grid[to_grid(c)] = c
                active.append(c)
                placed = True
                break
        if not placed:
            active.pop(idx)

    return result


def _pick_profile(zone: dict) -> AgentProfile:
    weights = zone.get("profile_weights", {
        "adult": 0.70, "child": 0.15, "elderly": 0.10, "mobility_impaired": 0.05,
    })
    choices = list(AgentProfile)
    keys = ["adult", "child", "elderly", "mobility_impaired"]
    w = [weights.get(k, 0) for k in keys]
    total = sum(w)
    return random.choices(choices, weights=[x / total for x in w], k=1)[0]


def spawn_agents(
    zones_path: str = "zones.json",
    min_dist: float = 0.6,   # ~2 * shoulder_radius + small gap
    seed: int | None = None,
) -> list[Agent]:
    """
    Read walkable zones from zones.json, Poisson-disk sample positions,
    assign profiles, and return initialised Agent list.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    with open(zones_path) as f:
        zone_data = json.load(f)

    agents: list[Agent] = []
    agent_id = 0

    for zone in zone_data.get("zones", []):
        if zone.get("type") in ("exit", "hazard", "wall"):
            continue
        coords = zone.get("polygon", [])
        if len(coords) < 3:
            continue
        poly = Polygon(coords)
        if not poly.is_valid or poly.area < 0.1:
            continue

        for pos in poisson_disk_sample(poly, min_dist=min_dist, rng=rng):
            profile = _pick_profile(zone)
            stats = PROFILE_STATS[profile]
            agents.append(Agent(
                id=agent_id,
                pos=pos.copy(),
                facing=float(np_rng.uniform(0, 2 * np.pi)),
                profile=profile,
                speed_max=stats["speed_calm"],
                shoulder_radius=stats["shoulder_radius"],
            ))
            agent_id += 1

    return agents


if __name__ == "__main__":
    agents = spawn_agents(seed=42)
    from collections import Counter
    print(f"Spawned {len(agents)} agents")
    print("Profile breakdown:", dict(Counter(a.profile.name for a in agents)))