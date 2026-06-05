"""
CA_evacuation.py  —  Cellular Automata Crowd Evacuation
Navigation : BFS flow field (pre-computed, wall-aware)
Movement   : CA rules (Moore neighbourhood, stochastic, speed-matched to SFM)
"""

import cv2
import json
import numpy as np
from collections import deque

# ══════════════════════════════════════════════════════════════════════
# CONFIG  — tweak everything here
# ══════════════════════════════════════════════════════════════════════
CONFIG = {
    "mask_path":    "stitched_mask.png",
    "config_path":  "stitched_mask_zone_config.json",
    "out_image":    "output_ca_paths.png",
    "out_report":   "output_ca_report.txt",

    "dt":           0.05,    # seconds per tick
    "max_time":     120.0,   # hard cap in seconds

    # 55 px/s × 0.05s = 3 px/tick  (matches SFM benchmark speed)
    "desired_speed":    55.0,
    "agent_wall_min":   6,       # min px from wall at spawn
    "randomness":       0.06,    # stochastic direction noise (0–1)
    "exit_radius":      22,      # px — within this distance = evacuated

    "score_time_good":  20.0,
    "score_time_bad":   80.0,
    "bn_low_fraction":  0.05,
    "bn_high_fraction": 0.50,
    "exit_underuse_pct": 40.0,
}

# 8-connected Moore neighbourhood — cardinal first, then diagonal
DIRS  = [(0,-1),(0,1),(-1,0),(1,0), (-1,-1),(-1,1),(1,-1),(1,1)]
DCOST = [  1.0,  1.0,  1.0,  1.0,    1.41,   1.41,  1.41,  1.41]

np.random.seed(42)


def load_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load mask: {path}")
    H, W     = img.shape
    walkable = img < 128          # white=wall, black=walkable
    walk_u8  = walkable.astype(np.uint8) * 255
    print(f"Mask loaded: {W}x{H}  walkable={walkable.sum()} px")
    return img, walkable, walk_u8, H, W


def build_bfs(walkable, exits_cfg, H, W, exit_radius):
    cost      = np.full((H, W), -1, dtype=np.int32)
    exit_zone = np.zeros((H, W), dtype=bool)
    queue     = deque()

    for ex in exits_cfg:
        ex_x, ex_y = int(ex["x"]), int(ex["y"])
        for dy in range(-exit_radius, exit_radius + 1):
            for dx in range(-exit_radius, exit_radius + 1):
                nx, ny = ex_x + dx, ex_y + dy
                if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx]:
                    exit_zone[ny, nx] = True
                    if cost[ny, nx] == -1:
                        cost[ny, nx] = 0
                        queue.append((nx, ny))

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in DIRS:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx] and cost[ny, nx] == -1:
                cost[ny, nx] = cost[cy, cx] + 1
                queue.append((nx, ny))

    print(f"Flow field built. Reachable cells: {(cost >= 0).sum()}")
    return cost, exit_zone


def spawn_agents(cfg_data, walkable, walk_u8, cost, exit_zone, agent_wall_min):
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    H, W         = walkable.shape
    dist_to_wall = cv2.distanceTransform(walk_u8, cv2.DIST_L2, 5)
    dist_norm    = cv2.normalize(dist_to_wall, None, 0, 1.0, cv2.NORM_MINMAX)
    coords       = peak_local_max(dist_norm, min_distance=40, labels=walk_u8)
    seed_mask    = np.zeros(dist_norm.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers, _   = ndi.label(seed_mask)
    labels       = watershed(-dist_to_wall, markers, mask=walk_u8)

    zone_pixels = {}
    for zdata in cfg_data["zones"]:
        zid    = zdata["zone_id"]
        ys, xs = np.where(labels == zid)
        valid  = (
            (dist_to_wall[ys, xs] > agent_wall_min) &
            (cost[ys, xs] >= 0) &
            (~exit_zone[ys, xs])
        )
        if valid.any():
            zone_pixels[zid] = list(zip(xs[valid].tolist(), ys[valid].tolist()))

    occupied = np.zeros((H, W), dtype=bool)
    agents   = []

    for zdata in cfg_data["zones"]:
        n      = zdata["agents"]
        zid    = zdata["zone_id"]
        pixels = zone_pixels.get(zid, [])
        if n == 0 or not pixels:
            continue
        pix = list(pixels)
        np.random.shuffle(pix)
        placed = 0
        for px, py in pix:
            if placed >= n:
                break
            if not occupied[py, px]:
                occupied[py, px] = True
                agents.append({
                    "x": px, "y": py,
                    "evacuated": False, "time": None, "exit_used": None,
                    "trail": [(px, py)],
                })
                placed += 1

    print(f"Spawned {len(agents)} agents")
    return agents, occupied


def run_simulation(agents, occupied, cost, exit_zone, walkable, exits_cfg, cfg):
    dt             = cfg["dt"]
    max_time       = cfg["max_time"]
    step_size      = max(1, int(cfg["desired_speed"] * dt))
    randomness     = cfg["randomness"]
    exit_radius    = cfg["exit_radius"]
    H, W           = walkable.shape
    exit_pts       = np.array([[e["x"], e["y"]] for e in exits_cfg], dtype=np.float32)
    density_map    = np.zeros((H, W), dtype=np.float32)
    congestion_map = np.zeros((H, W), dtype=np.float32)
    MAX_TICKS      = int(max_time / dt)
    sim_time       = 0.0

    print(f"Starting CA simulation  step={step_size}px/tick  max_t={max_time}s")

    for tick in range(MAX_TICKS):
        sim_time = tick * dt
        active   = [a for a in agents if not a["evacuated"]]
        if not active:
            break

        np.random.shuffle(active)   # stochastic ordering — core CA rule

        for agent in active:
            x, y = agent["x"], agent["y"]

            # Exit check before movement
            pos     = np.array([float(x), float(y)], dtype=np.float32)
            dists   = np.linalg.norm(exit_pts - pos, axis=1)
            nearest = int(np.argmin(dists))
            if dists[nearest] < exit_radius:
                agent["evacuated"] = True
                agent["time"]      = sim_time
                agent["exit_used"] = nearest
                occupied[y, x]     = False
                continue

            cur_cost = cost[y, x]
            if cur_cost < 0:
                continue

            # CA rule: find best free Moore neighbour with lower BFS cost
            cands = []
            for i, (dx, dy) in enumerate(DIRS):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if not walkable[ny, nx]:
                    continue
                nc = cost[ny, nx]
                if nc < 0 or nc >= cur_cost:
                    continue
                if occupied[ny, nx] and not exit_zone[ny, nx]:
                    continue   # hitbox — occupied cell blocks movement
                cands.append((nc, DCOST[i], dx, dy))

            if not cands:
                congestion_map[y, x] += dt
                density_map[y, x]    += 1
                agent["trail"].append((x, y))
                continue

            cands.sort()
            if randomness > 0 and np.random.random() < randomness and len(cands) > 1:
                _, _, dx, dy = cands[np.random.randint(len(cands))]
            else:
                _, _, dx, dy = cands[0]

            # Walk step_size pixels; re-query at each sub-step for natural curves
            cur_x, cur_y = x, y
            for _ in range(step_size):
                cc      = cost[cur_y, cur_x]
                best_nx = best_ny = -1
                best_nc = cc

                ordered = [(dx, dy, DCOST[DIRS.index((dx, dy))])]
                for i2, (ddx, ddy) in enumerate(DIRS):
                    if (ddx, ddy) != (dx, dy):
                        ordered.append((ddx, ddy, DCOST[i2]))

                for ddx, ddy, _ in ordered:
                    nnx, nny = cur_x + ddx, cur_y + ddy
                    if not (0 <= nnx < W and 0 <= nny < H):
                        continue
                    if not walkable[nny, nnx]:
                        continue
                    nc2 = cost[nny, nnx]
                    if nc2 < 0 or nc2 >= best_nc:
                        continue
                    if occupied[nny, nnx] and not exit_zone[nny, nnx]:
                        continue
                    best_nc = nc2
                    best_nx, best_ny = nnx, nny
                    break

                if best_nx < 0:
                    break

                if not exit_zone[cur_y, cur_x]:
                    occupied[cur_y, cur_x] = False
                cur_x, cur_y = best_nx, best_ny
                if not exit_zone[cur_y, cur_x]:
                    occupied[cur_y, cur_x] = True

                # Mid-step exit check
                pos2   = np.array([float(cur_x), float(cur_y)], dtype=np.float32)
                dists2 = np.linalg.norm(exit_pts - pos2, axis=1)
                n2     = int(np.argmin(dists2))
                if dists2[n2] < exit_radius:
                    agent["evacuated"] = True
                    agent["time"]      = sim_time
                    agent["exit_used"] = n2
                    if not exit_zone[cur_y, cur_x]:
                        occupied[cur_y, cur_x] = False
                    break

            agent["x"], agent["y"] = cur_x, cur_y
            agent["trail"].append((cur_x, cur_y))
            density_map[cur_y, cur_x]    += 1
            congestion_map[cur_y, cur_x] += dt

        if tick % int(5.0 / dt) == 0:
            evac = sum(a["evacuated"] for a in agents)
            print(f"  t={sim_time:6.1f}s  active={len(active):4d}  evacuated={evac:4d}")

    evac_final = sum(a["evacuated"] for a in agents)
    print(f"Done  t={sim_time:.1f}s  evacuated={evac_final}/{len(agents)}")
    return agents, density_map, congestion_map, evac_final, sim_time


def analyse(agents, walkable, congestion_map, exits_cfg, evac_final, sim_time_final, cfg):
    exit_counts = [0] * len(exits_cfg)
    for a in agents:
        if a["evacuated"] and a["exit_used"] is not None:
            exit_counts[a["exit_used"]] += 1
    total_evacuated = sum(exit_counts)

    wc        = congestion_map[walkable]
    threshold = np.percentile(wc[wc > 0], 80) if wc.max() > 0 else 1.0
    bn_mask     = (congestion_map > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bottlenecks = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        M  = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else bx + bw // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else by + bh // 2
        bottlenecks.append({
            "cx": cx, "cy": cy,
            "agent_seconds": float(congestion_map[by:by+bh, bx:bx+bw].sum()),
            "width_px": min(bw, bh),
        })
    bottlenecks.sort(key=lambda b: b["agent_seconds"], reverse=True)
    top_bn = bottlenecks[:5]

    total  = len(agents)
    times  = [a["time"] for a in agents if a["evacuated"] and a["time"] is not None]
    rate   = evac_final / total

    score_rate = rate * 50
    if times:
        mean_t     = np.mean(times)
        score_time = (20.0 if mean_t <= cfg["score_time_good"]
                      else max(0.0, 20 * (1 - (mean_t - cfg["score_time_good"]) /
                                           (cfg["score_time_bad"] - cfg["score_time_good"]))))
    else:
        mean_t, score_time = cfg["max_time"], 0.0

    if total_evacuated > 0:
        fracs         = [c / total_evacuated for c in exit_counts]
        ideal         = 1.0 / len(exits_cfg)
        score_balance = max(0.0, 15 * (1 - max(abs(f - ideal) for f in fracs) / ideal))
    else:
        score_balance = 0.0

    total_at = congestion_map.sum()
    bn_at    = sum(b["agent_seconds"] for b in top_bn)
    bn_frac  = bn_at / (total_at + 1e-8)
    score_bn = max(0.0, 15 * (1 - (bn_frac - cfg["bn_low_fraction"]) /
                               (cfg["bn_high_fraction"] - cfg["bn_low_fraction"])))

    final_score = min(100, max(0, int(score_rate + score_time + score_balance + score_bn)))

    worst = min(
        ("evacuation_rate", score_rate / 50),
        ("exit_balance",    score_balance / 15),
        ("bottlenecks",     score_bn / 15),
        ("evac_time",       score_time / 20),
        key=lambda x: x[1],
    )
    bn_pos = f"({top_bn[0]['cx']}, {top_bn[0]['cy']})" if top_bn else "unknown"
    if total_evacuated > 0:
        min_exit_idx = int(np.argmin(exit_counts))
        min_exit_pos = f"({int(exits_cfg[min_exit_idx]['x'])}, {int(exits_cfg[min_exit_idx]['y'])})"
    else:
        min_exit_idx, min_exit_pos = 0, "N/A"

    RECS = {
        "evacuation_rate": "Too many agents failed to evacuate — check for isolated rooms with no path to any exit.",
        "exit_balance":    f"Exit {min_exit_idx} at {min_exit_pos} handled almost no traffic. Consider repositioning it closer to high-density zones or adding signage.",
        "bottlenecks":     f"The corridor at {bn_pos} is your biggest chokepoint. Widen it or add a parallel route.",
        "evac_time":       "Evacuation is too slow. Add an exit closer to the center of the building.",
    }

    return {
        "total": total, "evac_final": evac_final, "times": times, "mean_t": mean_t,
        "rate": rate, "exit_counts": exit_counts, "total_evacuated": total_evacuated,
        "top_bn": top_bn, "final_score": final_score, "score_rate": score_rate,
        "score_time": score_time, "score_balance": score_balance, "score_bn": score_bn,
        "recommendation": RECS[worst[0]], "sim_time_final": sim_time_final,
    }


def render_output(img, agents, density_map, exits_cfg, analysis, out_path, walkable):
    H, W  = img.shape
    base  = np.zeros((H, W, 3), dtype=np.uint8)
    base[~walkable] = [60, 60, 60]

    if density_map.max() > 0:
        dn    = (density_map / density_map.max() * 255).astype(np.uint8)
        heat  = cv2.applyColorMap(dn, cv2.COLORMAP_HOT)
        mask3 = walkable.astype(np.uint8)[:, :, None]
        base  = cv2.addWeighted(base, 0.45, heat * mask3, 0.55, 0)

    for agent in agents:
        color = (0, 200, 0) if agent["evacuated"] else (0, 110, 220)
        trail = agent["trail"]
        pts   = trail[::2] + [trail[-1]]
        for i in range(1, len(pts)):
            cv2.line(base, pts[i-1], pts[i], color, 1)

    total_evac = analysis["total_evacuated"]
    for idx, ex in enumerate(exits_cfg):
        ep  = (int(ex["x"]), int(ex["y"]))
        pct = (analysis["exit_counts"][idx] / total_evac * 100) if total_evac > 0 else 0
        cv2.circle(base, ep, 14, (0, 0, 0), -1)
        cv2.circle(base, ep, 14, (200, 215, 0), 2)
        cv2.putText(base, f"E{idx} {pct:.0f}%", (ep[0] - 18, ep[1] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 215, 0), 1)

    for rank, bn in enumerate(analysis["top_bn"]):
        cv2.circle(base, (bn["cx"], bn["cy"]), 14, (0, 0, 255), 2)
        cv2.putText(base, f"B{rank+1}", (bn["cx"] - 8, bn["cy"] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    sc          = analysis["final_score"]
    score_color = (0,200,0) if sc >= 80 else (0,180,255) if sc >= 60 else (0,0,255)
    cv2.rectangle(base, (4, 4), (195, 30), (0, 0, 0), -1)
    cv2.putText(base, f"SCORE: {sc}/100  [CA]",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 1)

    r, ef, tot, st = analysis["rate"], analysis["evac_final"], analysis["total"], analysis["sim_time_final"]
    cv2.putText(base, f"Agents:{tot}  Evac:{ef}({100*r:.0f}%)  t={st:.0f}s",
                (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    cv2.imwrite(out_path, base)
    print(f"Saved: {out_path}")


def print_report(analysis, exits_cfg, cfg):
    r     = analysis
    lines = [
        "=" * 55,
        "       CA EVACUATION ANALYSIS REPORT",
        "=" * 55, "",
        f"  OVERALL SCORE : {r['final_score']} / 100", "",
        f"  Total agents    : {r['total']}",
        f"  Evacuated       : {r['evac_final']}  ({100*r['rate']:.1f}%)",
        f"  Trapped/timeout : {r['total'] - r['evac_final']}  ({100*(1-r['rate']):.1f}%)",
    ]
    if r["times"]:
        lines += [
            f"  Fastest evac    : {min(r['times']):.1f}s",
            f"  Mean evac time  : {r['mean_t']:.1f}s",
            f"  Slowest evac    : {max(r['times']):.1f}s",
        ]

    lines += ["", "-"*55, "  EXIT UTILIZATION", "-"*55]
    ideal_pct = 100.0 / max(len(exits_cfg), 1)
    for idx, ex in enumerate(exits_cfg):
        pct    = (r["exit_counts"][idx] / r["total_evacuated"] * 100) if r["total_evacuated"] > 0 else 0
        bar    = "█" * int(pct / 5)
        status = "⚠ UNDERUSED" if pct < ideal_pct * (cfg["exit_underuse_pct"] / 100.0) else ""
        lines.append(
            f"  Exit {idx} ({int(ex['x'])},{int(ex['y'])}): "
            f"{r['exit_counts'][idx]:3d} agents  {pct:5.1f}%  {bar} {status}"
        )

    lines += ["", "-"*55, "  TOP BOTTLENECKS  (ranked by agent-seconds lost)", "-"*55]
    for rank, bn in enumerate(r["top_bn"]):
        lines.append(
            f"  B{rank+1}  position ({bn['cx']:4d},{bn['cy']:4d})  "
            f"corridor width ~{bn['width_px']}px  "
            f"{bn['agent_seconds']:.0f} agent-seconds"
        )

    lines += [
        "", "-"*55, "  RECOMMENDATION", "-"*55,
        f"  {r['recommendation']}", "",
        "  Score breakdown:",
        f"    Evacuation rate  : {r['score_rate']:.0f}/50",
        f"    Evacuation speed : {r['score_time']:.0f}/20",
        f"    Exit balance     : {r['score_balance']:.0f}/15",
        f"    Bottleneck sev.  : {r['score_bn']:.0f}/15",
        "=" * 55,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    with open(cfg["out_report"], "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved: {cfg['out_report']}")


def main():
    C = CONFIG
    img, walkable, walk_u8, H, W = load_mask(C["mask_path"])

    with open(C["config_path"]) as f:
        cfg_data = json.load(f)
    exits_cfg = cfg_data["exits"]

    cost, exit_zone  = build_bfs(walkable, exits_cfg, H, W, C["exit_radius"])
    agents, occupied = spawn_agents(cfg_data, walkable, walk_u8, cost, exit_zone, C["agent_wall_min"])

    agents, density_map, congestion_map, evac_final, sim_time_final = run_simulation(
        agents, occupied, cost, exit_zone, walkable, exits_cfg, C
    )
    analysis = analyse(agents, walkable, congestion_map, exits_cfg, evac_final, sim_time_final, C)
    render_output(img, agents, density_map, exits_cfg, analysis, C["out_image"], walkable)
    print_report(analysis, exits_cfg, C)


if __name__ == "__main__":
    main()