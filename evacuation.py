# """
# evacuation.py
# -------------
# 1. Load mask + zone config
# 2. Build BFS flow field from exits (once)
# 3. Spawn agents in walkable zones
# 4. Run simulation: agents follow flow field + wall repulsion
# 5. Output: path image + text report
# """

# import cv2
# import json
# import numpy as np
# from collections import deque

# # ── CONFIG ─────────────────────────────────────────────────────────────
# MASK_PATH   = "stitched_mask.png"
# CONFIG_PATH = "stitched_mask_zone_config.json"
# OUT_PATHS   = "output_paths.png"
# OUT_REPORT  = "output_report.txt"

# AGENT_RADIUS  = 2       # px, for collision/display
# SPEED         = 30      # px per second
# DT            = 0.1     # seconds per tick
# MAX_TIME      = 100     # seconds before we stop

# WALL_REPEL_STRENGTH = 80   # how hard agents push away from walls
# WALL_REPEL_DIST     = 12   # px, how close before wall repulsion kicks in
# # ───────────────────────────────────────────────────────────────────────


# # ══════════════════════════════════════════════════════════════════════
# # STEP 1 — Load mask
# # ══════════════════════════════════════════════════════════════════════

# img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
# H, W = img.shape

# # White = walls, Black = walkable  (your mask convention)
# walkable = img < 128   # True where agents can walk

# # Distance transform — every walkable cell gets distance to nearest wall
# walk_u8 = walkable.astype(np.uint8) * 255
# dist_to_wall = cv2.distanceTransform(walk_u8, cv2.DIST_L2, 5)


# # ══════════════════════════════════════════════════════════════════════
# # STEP 2 — BFS flow field from all exits
# # ══════════════════════════════════════════════════════════════════════

# with open(CONFIG_PATH) as f:
#     cfg = json.load(f)

# exits = cfg["exits"]   # list of {x, y}

# # cost map: how many steps to reach nearest exit from each cell
# # -1 = unreachable / wall
# cost = np.full((H, W), -1, dtype=np.int32)

# # direction map: unit vector pointing toward exit
# # stored as two float arrays dx, dy
# flow_x = np.zeros((H, W), dtype=np.float32)
# flow_y = np.zeros((H, W), dtype=np.float32)

# # seed BFS from every exit cell
# queue = deque()
# for ex in exits:
#     ex_x, ex_y = int(ex["x"]), int(ex["y"])
#     # mark a small radius around each exit as cost=0
#     for dy in range(-AGENT_RADIUS, AGENT_RADIUS+1):
#         for dx in range(-AGENT_RADIUS, AGENT_RADIUS+1):
#             nx, ny = ex_x + dx, ex_y + dy
#             if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx]:
#                 if cost[ny, nx] == -1:
#                     cost[ny, nx] = 0
#                     queue.append((nx, ny))

# # BFS outward — 8-connected
# neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

# while queue:
#     cx, cy = queue.popleft()
#     for dx, dy in neighbors:
#         nx, ny = cx + dx, cy + dy
#         if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx] and cost[ny, nx] == -1:
#             cost[ny, nx] = cost[cy, cx] + 1
#             queue.append((nx, ny))

# # Build direction vectors: point from each cell toward its lowest-cost neighbor
# for y in range(H):
#     for x in range(W):
#         if not walkable[y, x] or cost[y, x] == -1:
#             continue
#         best_cost = cost[y, x]
#         bx, by = 0.0, 0.0
#         for dx, dy in neighbors:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < W and 0 <= ny < H and cost[ny, nx] != -1:
#                 if cost[ny, nx] < best_cost:
#                     best_cost = cost[ny, nx]
#                     bx, by = float(dx), float(dy)
#         # normalise
#         mag = (bx**2 + by**2)**0.5
#         if mag > 0:
#             flow_x[y, x] = bx / mag
#             flow_y[y, x] = by / mag

# print(f"Flow field built. Reachable cells: {(cost >= 0).sum()}")


# # ══════════════════════════════════════════════════════════════════════
# # STEP 3 — Spawn agents from zone config
# # ══════════════════════════════════════════════════════════════════════

# # First label the zones (same watershed logic your zone_detector uses)
# from scipy import ndimage as ndi
# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max

# dist_norm = cv2.normalize(dist_to_wall, None, 0, 1.0, cv2.NORM_MINMAX)
# coords = peak_local_max(dist_norm, min_distance=40, labels=walk_u8)
# seed_mask = np.zeros(dist_norm.shape, dtype=bool)
# seed_mask[tuple(coords.T)] = True
# markers, _ = ndi.label(seed_mask)
# labels = watershed(-dist_to_wall, markers, mask=walk_u8)

# # Map zone_id → list of walkable pixel coords
# zone_pixels = {}
# for zdata in cfg["zones"]:
#     zid = zdata["zone_id"]
#     ys, xs = np.where(labels == zid)
#     # only keep pixels far enough from walls
#     valid = dist_to_wall[ys, xs] > WALL_REPEL_DIST
#     if valid.any():
#         zone_pixels[zid] = list(zip(xs[valid], ys[valid]))

# # Spawn agents
# agents = []   # each agent: [x, y, evacuated, trail]

# for zdata in cfg["zones"]:
#     n = zdata["agents"]
#     if n == 0:
#         continue
#     zid = zdata["zone_id"]
#     pixels = zone_pixels.get(zid, [])
#     if not pixels:
#         continue
#     chosen = [pixels[i] for i in np.random.choice(len(pixels), min(n, len(pixels)), replace=False)]
#     for px, py in chosen:
#         agents.append({
#             "x": float(px),
#             "y": float(py),
#             "evacuated": False,
#             "time": None,
#             "trail": [(float(px), float(py))]
#         })

# print(f"Spawned {len(agents)} agents")


# # ══════════════════════════════════════════════════════════════════════
# # STEP 4 — Run simulation
# # ══════════════════════════════════════════════════════════════════════

# # Exit positions as numpy array for fast distance check
# exit_pts = np.array([[e["x"], e["y"]] for e in exits], dtype=np.float32)

# sim_time = 0.0
# steps = int(MAX_TIME / DT)

# # density heatmap accumulator
# density_map = np.zeros((H, W), dtype=np.float32)

# for step in range(steps):
#     sim_time = step * DT
#     active = [a for a in agents if not a["evacuated"]]
#     if not active:
#         break

#     for agent in active:
#         x, y = agent["x"], agent["y"]
#         ix, iy = int(np.clip(x, 0, W-1)), int(np.clip(y, 0, H-1))

#         # -- flow field direction --
#         fx = flow_x[iy, ix]
#         fy = flow_y[iy, ix]

#         # -- wall repulsion --
#         d = dist_to_wall[iy, ix]
#         if d < WALL_REPEL_DIST and d > 0:
#             # gradient of distance transform = direction away from wall
#             gx = dist_to_wall[iy, min(ix+1, W-1)] - dist_to_wall[iy, max(ix-1, 0)]
#             gy = dist_to_wall[min(iy+1, H-1), ix] - dist_to_wall[max(iy-1, 0), ix]
#             strength = (WALL_REPEL_DIST - d) / WALL_REPEL_DIST
#             fx += gx * strength * (WALL_REPEL_STRENGTH / SPEED)
#             fy += gy * strength * (WALL_REPEL_STRENGTH / SPEED)

#         # normalise combined direction
#         mag = (fx**2 + fy**2)**0.5
#         if mag > 0:
#             fx /= mag
#             fy /= mag

#         # move
#         nx = x + fx * SPEED * DT
#         ny = y + fy * SPEED * DT

#         # clamp to bounds, only move if walkable
#         nx = np.clip(nx, 0, W-1)
#         ny = np.clip(ny, 0, H-1)
#         if walkable[int(ny), int(nx)]:
#             agent["x"], agent["y"] = nx, ny
#         # else: stay (wall blocked)

#         agent["trail"].append((agent["x"], agent["y"]))

#         # update density
#         density_map[int(agent["y"]), int(agent["x"])] += 1

#         # check if reached any exit
#         pos = np.array([agent["x"], agent["y"]])
#         dists = np.linalg.norm(exit_pts - pos, axis=1)
#         if dists.min() < 18:
#             agent["evacuated"] = True
#             agent["time"] = sim_time

#     if step % 50 == 0:
#         print(f"  t={sim_time:.1f}s  active={len(active)}  evacuated={sum(a['evacuated'] for a in agents)}")

# print(f"Simulation done at t={sim_time:.1f}s")


# # ══════════════════════════════════════════════════════════════════════
# # STEP 5 — Output: path image + report
# # ══════════════════════════════════════════════════════════════════════

# # --- Path image ---
# # Base: grayscale mask converted to BGR
# base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# # Draw density heatmap as a semi-transparent overlay
# if density_map.max() > 0:
#     dn = (density_map / density_map.max() * 255).astype(np.uint8)
#     heat = cv2.applyColorMap(dn, cv2.COLORMAP_HOT)
#     mask_area = walkable.astype(np.uint8)[:, :, None]
#     base = cv2.addWeighted(base, 0.4, heat * mask_area, 0.6, 0)

# # Draw agent trails
# for agent in agents:
#     trail = agent["trail"]
#     color = (0, 200, 0) if agent["evacuated"] else (0, 0, 200)
#     for i in range(1, len(trail)):
#         p1 = (int(trail[i-1][0]), int(trail[i-1][1]))
#         p2 = (int(trail[i][0]),   int(trail[i][1]))
#         cv2.line(base, p1, p2, color, 1)

# # Draw exits
# for ex in exits:
#     cv2.circle(base, (int(ex["x"]), int(ex["y"])), 10, (0, 255, 255), 2)
#     cv2.putText(base, "EXIT", (int(ex["x"])-15, int(ex["y"])-14),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

# # Draw bottleneck zones (where density was very high)
# thresh = density_map.max() * 0.6
# bottleneck_mask = (density_map > thresh).astype(np.uint8) * 255
# contours, _ = cv2.findContours(bottleneck_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(base, contours, -1, (0, 0, 255), 2)

# cv2.imwrite(OUT_PATHS, base)
# print(f"Saved: {OUT_PATHS}")

# # --- Text report ---
# total     = len(agents)
# evacuated = sum(a["evacuated"] for a in agents)
# trapped   = total - evacuated
# times     = [a["time"] for a in agents if a["evacuated"] and a["time"] is not None]

# report_lines = [
#     "=" * 50,
#     "EVACUATION ANALYSIS REPORT",
#     "=" * 50,
#     f"Total agents      : {total}",
#     f"Evacuated         : {evacuated}  ({100*evacuated/total:.1f}%)",
#     f"Trapped / timeout : {trapped}   ({100*trapped/total:.1f}%)",
#     "",
# ]

# if times:
#     report_lines += [
#         f"Fastest evacuation: {min(times):.1f}s",
#         f"Mean evac time    : {np.mean(times):.1f}s",
#         f"Slowest evacuated : {max(times):.1f}s",
#         "",
#     ]

# report_lines += [
#     f"Bottleneck zones  : {len(contours)} high-density areas found",
#     "  (red outlines on output image)",
#     "",
#     "VERDICT:",
# ]

# rate = evacuated / total
# if rate >= 0.90:
#     report_lines.append("  GOOD — 90%+ evacuated. Building layout is efficient.")
# elif rate >= 0.70:
#     report_lines.append("  MODERATE — 70-90% evacuated. Some bottlenecks exist.")
# else:
#     report_lines.append("  POOR — Under 70% evacuated. Serious layout problems.")

# report_lines.append("=" * 50)

# report = "\n".join(report_lines)
# print("\n" + report)

# with open(OUT_REPORT, "w") as f:
#     f.write(report)
# print(f"\nSaved: {OUT_REPORT}")







# """
# evacuation.py
# -------------
# 1. Load mask + zone config
# 2. Build BFS flow field from exits (once)
# 3. Spawn agents in walkable zones
# 4. Run simulation: agents follow flow field, nudge when stuck
# 5. Output: path image + text report
# """

# import cv2
# import json
# import numpy as np
# from collections import deque

# # ── CONFIG ──────────────────────────────────────────────────────────────
# MASK_PATH   = "stitched_mask.png"
# CONFIG_PATH = "stitched_mask_zone_config.json"
# OUT_PATHS   = "output_paths.png"
# OUT_REPORT  = "output_report.txt"

# AGENT_RADIUS   = 500     # px display radius
# SPEED          = 30    # px per second
# DT             = 0.1  # seconds per tick
# MAX_TIME       = 50   # stop after this many seconds

# # Stuck detection: if agent travels < STUCK_DIST px in STUCK_TICKS, nudge it
# STUCK_DIST  = 2
# STUCK_TICKS = 15
# EXIT_RADIUS = 20       # px — how close to count as evacuated
# # ────────────────────────────────────────────────────────────────────────


# # ═══════════════════════════════════════════════════════════════════════
# # STEP 1 — Load mask
# # ═══════════════════════════════════════════════════════════════════════

# img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
# H, W = img.shape

# walkable     = img < 128                              # True = agent can walk here
# walk_u8      = walkable.astype(np.uint8) * 255
# dist_to_wall = cv2.distanceTransform(walk_u8, cv2.DIST_L2, 5)


# # ═══════════════════════════════════════════════════════════════════════
# # STEP 2 — BFS flow field from all exits
# # ═══════════════════════════════════════════════════════════════════════

# with open(CONFIG_PATH) as f:
#     cfg = json.load(f)

# exits = cfg["exits"]

# # BFS cost map: steps to nearest exit. -1 = wall/unreachable
# cost   = np.full((H, W), -1, dtype=np.int32)
# flow_x = np.zeros((H, W), dtype=np.float32)
# flow_y = np.zeros((H, W), dtype=np.float32)

# queue = deque()
# for ex in exits:
#     ex_x, ex_y = int(ex["x"]), int(ex["y"])
#     for dy in range(-EXIT_RADIUS, EXIT_RADIUS + 1):
#         for dx in range(-EXIT_RADIUS, EXIT_RADIUS + 1):
#             nx, ny = ex_x + dx, ex_y + dy
#             if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx] and cost[ny, nx] == -1:
#                 cost[ny, nx] = 0
#                 queue.append((nx, ny))

# DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

# while queue:
#     cx, cy = queue.popleft()
#     for dx, dy in DIRS:
#         nx, ny = cx + dx, cy + dy
#         if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx] and cost[ny, nx] == -1:
#             cost[ny, nx] = cost[cy, cx] + 1
#             queue.append((nx, ny))

# # Build direction vectors — each cell points toward its lowest-cost neighbour
# for y in range(H):
#     for x in range(W):
#         if not walkable[y, x] or cost[y, x] == -1:
#             continue
#         best = cost[y, x]
#         bx = by = 0.0
#         for dx, dy in DIRS:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < W and 0 <= ny < H and 0 <= cost[ny, nx] < best:
#                 best = cost[ny, nx]
#                 bx, by = float(dx), float(dy)
#         mag = (bx**2 + by**2) ** 0.5
#         if mag > 0:
#             flow_x[y, x] = bx / mag
#             flow_y[y, x] = by / mag

# print(f"Flow field built. Reachable cells: {(cost >= 0).sum()}")


# # ═══════════════════════════════════════════════════════════════════════
# # STEP 3 — Spawn agents from zone config
# # ═══════════════════════════════════════════════════════════════════════

# from scipy import ndimage as ndi
# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max

# dist_norm = cv2.normalize(dist_to_wall, None, 0, 1.0, cv2.NORM_MINMAX)
# coords    = peak_local_max(dist_norm, min_distance=40, labels=walk_u8)
# seed_mask = np.zeros(dist_norm.shape, dtype=bool)
# seed_mask[tuple(coords.T)] = True
# markers, _ = ndi.label(seed_mask)
# labels     = watershed(-dist_to_wall, markers, mask=walk_u8)

# # zone_id → list of walkable pixel coords that are reachable
# zone_pixels = {}
# for zdata in cfg["zones"]:
#     zid    = zdata["zone_id"]
#     ys, xs = np.where(labels == zid)
#     # only keep pixels that are walkable AND reachable by BFS
#     valid  = (dist_to_wall[ys, xs] > 3) & (cost[ys, xs] >= 0)
#     if valid.any():
#         zone_pixels[zid] = list(zip(xs[valid].tolist(), ys[valid].tolist()))

# agents = []
# for zdata in cfg["zones"]:
#     n      = zdata["agents"]
#     zid    = zdata["zone_id"]
#     pixels = zone_pixels.get(zid, [])
#     if n == 0 or not pixels:
#         continue
#     chosen = [pixels[i] for i in np.random.choice(len(pixels), min(n, len(pixels)), replace=False)]
#     for px, py in chosen:
#         agents.append({
#             "x":         float(px),
#             "y":         float(py),
#             "evacuated": False,
#             "time":      None,
#             "trail":     [(float(px), float(py))],
#             "stuck_buf": [],          # recent positions for stuck detection
#         })

# print(f"Spawned {len(agents)} agents")


# # ═══════════════════════════════════════════════════════════════════════
# # STEP 4 — Simulation loop
# # ═══════════════════════════════════════════════════════════════════════

# exit_pts    = np.array([[e["x"], e["y"]] for e in exits], dtype=np.float32)
# density_map = np.zeros((H, W), dtype=np.float32)
# total_steps = int(MAX_TIME / DT)

# for step in range(total_steps):
#     sim_time = step * DT
#     active   = [a for a in agents if not a["evacuated"]]
#     if not active:
#         break

#     for agent in active:
#         x, y   = agent["x"], agent["y"]
#         ix, iy = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))

#         # Flow field direction — this is all we need for navigation
#         fx = flow_x[iy, ix]
#         fy = flow_y[iy, ix]

#         # If flow is zero (cell has no direction), do a small random nudge now
#         if fx == 0 and fy == 0:
#             fx = np.random.uniform(-1, 1)
#             fy = np.random.uniform(-1, 1)

#         # Stuck detection: if barely moved recently, add noise to escape
#         buf = agent["stuck_buf"]
#         buf.append((x, y))
#         if len(buf) > STUCK_TICKS:
#             buf.pop(0)
#             dx_total = abs(buf[-1][0] - buf[0][0])
#             dy_total = abs(buf[-1][1] - buf[0][1])
#             if dx_total + dy_total < STUCK_DIST:
#                 # Agent is stuck — nudge in a random walkable direction
#                 angle  = np.random.uniform(0, 2 * np.pi)
#                 fx     = np.cos(angle)
#                 fy     = np.sin(angle)

#         # Normalise
#         mag = (fx**2 + fy**2) ** 0.5
#         if mag > 0:
#             fx /= mag
#             fy /= mag

#         # Try to move
#         nx = np.clip(x + fx * SPEED * DT, 0, W - 1)
#         ny = np.clip(y + fy * SPEED * DT, 0, H - 1)

#         if walkable[int(ny), int(nx)]:
#             agent["x"], agent["y"] = nx, ny
#         else:
#             # Wall ahead — try sliding along x or y axis separately
#             if walkable[int(y),  int(nx)]:
#                 agent["x"] = nx
#             elif walkable[int(ny), int(x)]:
#                 agent["y"] = ny
#             # else: truly blocked, stay put this tick

#         agent["trail"].append((agent["x"], agent["y"]))
#         density_map[int(agent["y"]), int(agent["x"])] += 1

#         # Check exit
#         pos   = np.array([agent["x"], agent["y"]])
#         dists = np.linalg.norm(exit_pts - pos, axis=1)
#         if dists.min() < EXIT_RADIUS:
#             agent["evacuated"] = True
#             agent["time"]      = sim_time

#     if step % 100 == 0:
#         evac = sum(a["evacuated"] for a in agents)
#         print(f"  t={sim_time:.1f}s  active={len(active)}  evacuated={evac}")

# evac_final = sum(a["evacuated"] for a in agents)
# print(f"Simulation done at t={sim_time:.1f}s  |  evacuated={evac_final}/{len(agents)}")


# # ═══════════════════════════════════════════════════════════════════════
# # STEP 5 — Output
# # ═══════════════════════════════════════════════════════════════════════

# # Base image
# base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# # Density heatmap overlay
# if density_map.max() > 0:
#     dn   = (density_map / density_map.max() * 255).astype(np.uint8)
#     heat = cv2.applyColorMap(dn, cv2.COLORMAP_HOT)
#     mask_area = walkable.astype(np.uint8)[:, :, None]
#     base = cv2.addWeighted(base, 0.45, heat * mask_area, 0.55, 0)

# # Agent trails — green = evacuated, blue = stuck
# for agent in agents:
#     trail = agent["trail"]
#     color = (0, 200, 0) if agent["evacuated"] else (200, 100, 0)
#     for i in range(1, len(trail)):
#         p1 = (int(trail[i-1][0]), int(trail[i-1][1]))
#         p2 = (int(trail[i][0]),   int(trail[i][1]))
#         cv2.line(base, p1, p2, color, 1)

# # Exits
# for ex in exits:
#     cv2.circle(base, (int(ex["x"]), int(ex["y"])), 12, (0, 255, 255), 2)
#     cv2.putText(base, "EXIT", (int(ex["x"]) - 15, int(ex["y"]) - 16),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

# # Bottleneck contours
# thresh          = density_map.max() * 0.65
# bn_mask         = (density_map > thresh).astype(np.uint8) * 255
# contours, _     = cv2.findContours(bn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(base, contours, -1, (0, 0, 255), 2)

# cv2.imwrite(OUT_PATHS, base)
# print(f"Saved: {OUT_PATHS}")

# # Report
# total     = len(agents)
# evacuated = evac_final
# trapped   = total - evacuated
# times     = [a["time"] for a in agents if a["evacuated"] and a["time"] is not None]

# lines = [
#     "=" * 50,
#     "EVACUATION ANALYSIS REPORT",
#     "=" * 50,
#     f"Total agents      : {total}",
#     f"Evacuated         : {evacuated}  ({100*evacuated/total:.1f}%)",
#     f"Trapped / timeout : {trapped}   ({100*trapped/total:.1f}%)",
#     "",
# ]
# if times:
#     lines += [
#         f"Fastest evacuation: {min(times):.1f}s",
#         f"Mean evac time    : {np.mean(times):.1f}s",
#         f"Slowest evacuated : {max(times):.1f}s",
#         "",
#     ]
# lines += [
#     f"Bottleneck zones  : {len(contours)} high-density areas",
#     "  (red outlines on output image)",
#     "",
#     "VERDICT:",
# ]
# rate = evacuated / total
# if rate >= 0.90:
#     lines.append("  GOOD — 90%+ evacuated. Layout is efficient.")
# elif rate >= 0.70:
#     lines.append("  MODERATE — 70-90% evacuated. Some bottlenecks exist.")
# else:
#     lines.append("  POOR — Under 70% evacuated. Serious layout problems.")
# lines.append("=" * 50)

# report = "\n".join(lines)
# print("\n" + report)
# with open(OUT_REPORT, "w") as f:
#     f.write(report)
# print(f"Saved: {OUT_REPORT}")








# """
# evacuation.py
# -------------
# Navigation : BFS flow field  (tells agents WHERE to go)
# Movement   : Social Force Model (tells agents HOW to move)

# SFM forces on each agent every tick:
#   1. Driving force  — wants to walk at desired speed toward BFS direction
#   2. Agent repulsion — pushed away from nearby agents (crowd pressure)
#   3. Wall repulsion  — pushed away from walls using distance transform gradient
# """

# import cv2
# import json
# import numpy as np
# from collections import deque

# # ── CONFIG ──────────────────────────────────────────────────────────────
# MASK_PATH   = "stitched_mask.png"
# CONFIG_PATH = "stitched_mask_zone_config.json"
# OUT_PATHS   = "output_paths.png"
# OUT_REPORT  = "output_report.txt"

# DT       = 0.05   # seconds per tick
# MAX_TIME = 120    # stop after this many seconds

# # ── SFM parameters ──────────────────────────────────────────────────────
# DESIRED_SPEED   = 55.0   # px/s  — how fast agent wants to walk
# RELAXATION_TIME = 0.3    # s     — how fast agent reaches desired speed (tau)
# AGENT_RADIUS    = 6      # px    — physical body radius for repulsion

# # Agent-agent repulsion  (Helbing 1995: A * exp(-d/B))
# AA_STRENGTH  = 300.0    # A
# AA_RANGE     = 14.0       # B  (px)

# # Wall repulsion
# WA_STRENGTH  = 350.0
# WA_RANGE     = 7.0      # px — effective range

# EXIT_RADIUS  = 22        # px — close enough to count as evacuated

# # Stuck recovery
# STUCK_DIST   = 2         # px total travel over window before nudge
# STUCK_TICKS  = 20
# # ────────────────────────────────────────────────────────────────────────


# # ═══════════════════════════════════════════════════════════════════════
# # STEP 1 — Load mask + distance transform
# # ═══════════════════════════════════════════════════════════════════════

# img          = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
# H, W         = img.shape
# walkable     = img < 128
# walk_u8      = walkable.astype(np.uint8) * 255
# dist_to_wall = cv2.distanceTransform(walk_u8, cv2.DIST_L2, 5)

# # Precompute wall-repulsion gradient (direction away from nearest wall)
# # Use numpy gradient of the distance transform
# wall_grad_y, wall_grad_x = np.gradient(dist_to_wall.astype(np.float32))
# # Normalise so we get a unit vector field
# wall_grad_mag = np.sqrt(wall_grad_x**2 + wall_grad_y**2) + 1e-8
# wall_grad_x  /= wall_grad_mag
# wall_grad_y  /= wall_grad_mag


# # ═══════════════════════════════════════════════════════════════════════
# # STEP 2 — BFS flow field (navigation only — WHERE to go)
# # ═══════════════════════════════════════════════════════════════════════

# with open(CONFIG_PATH) as f:
#     cfg = json.load(f)

# exits = cfg["exits"]

# cost   = np.full((H, W), -1, dtype=np.int32)
# flow_x = np.zeros((H, W), dtype=np.float32)
# flow_y = np.zeros((H, W), dtype=np.float32)

# queue = deque()
# for ex in exits:
#     ex_x, ex_y = int(ex["x"]), int(ex["y"])
#     for dy in range(-EXIT_RADIUS, EXIT_RADIUS + 1):
#         for dx in range(-EXIT_RADIUS, EXIT_RADIUS + 1):
#             nx, ny = ex_x + dx, ex_y + dy
#             if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx] and cost[ny, nx] == -1:
#                 cost[ny, nx] = 0
#                 queue.append((nx, ny))

# DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

# while queue:
#     cx, cy = queue.popleft()
#     for dx, dy in DIRS:
#         nx, ny = cx + dx, cy + dy
#         if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx] and cost[ny, nx] == -1:
#             cost[ny, nx] = cost[cy, cx] + 1
#             queue.append((nx, ny))

# for y in range(H):
#     for x in range(W):
#         if not walkable[y, x] or cost[y, x] == -1:
#             continue
#         best = cost[y, x]
#         bx = by = 0.0
#         for dx, dy in DIRS:
#             nx, ny = x + dx, y + dy
#             if 0 <= nx < W and 0 <= ny < H and 0 <= cost[ny, nx] < best:
#                 best = cost[ny, nx]
#                 bx, by = float(dx), float(dy)
#         mag = (bx**2 + by**2) ** 0.5
#         if mag > 0:
#             flow_x[y, x] = bx / mag
#             flow_y[y, x] = by / mag

# print(f"Flow field built. Reachable cells: {(cost >= 0).sum()}")


# # ═══════════════════════════════════════════════════════════════════════
# # STEP 3 — Spawn agents
# # ═══════════════════════════════════════════════════════════════════════

# from scipy import ndimage as ndi
# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max

# dist_norm  = cv2.normalize(dist_to_wall, None, 0, 1.0, cv2.NORM_MINMAX)
# coords     = peak_local_max(dist_norm, min_distance=40, labels=walk_u8)
# seed_mask  = np.zeros(dist_norm.shape, dtype=bool)
# seed_mask[tuple(coords.T)] = True
# markers, _ = ndi.label(seed_mask)
# labels     = watershed(-dist_to_wall, markers, mask=walk_u8)

# zone_pixels = {}
# for zdata in cfg["zones"]:
#     zid    = zdata["zone_id"]
#     ys, xs = np.where(labels == zid)
#     valid  = (dist_to_wall[ys, xs] > AGENT_RADIUS) & (cost[ys, xs] >= 0)
#     if valid.any():
#         zone_pixels[zid] = list(zip(xs[valid].tolist(), ys[valid].tolist()))

# agents = []
# for zdata in cfg["zones"]:
#     n      = zdata["agents"]
#     zid    = zdata["zone_id"]
#     pixels = zone_pixels.get(zid, [])
#     if n == 0 or not pixels:
#         continue
#     chosen = [pixels[i] for i in np.random.choice(len(pixels), min(n, len(pixels)), replace=False)]
#     for px, py in chosen:
#         agents.append({
#             "x":   float(px),   "y":   float(py),
#             "vx":  0.0,         "vy":  0.0,         # velocity
#             "evacuated": False, "time": None,
#             "trail":     [(float(px), float(py))],
#             "stuck_buf": [],
#         })

# print(f"Spawned {len(agents)} agents")


# # ═══════════════════════════════════════════════════════════════════════
# # STEP 4 — Simulation: SFM forces each tick
# # ═══════════════════════════════════════════════════════════════════════

# exit_pts    = np.array([[e["x"], e["y"]] for e in exits], dtype=np.float32)
# density_map = np.zeros((H, W), dtype=np.float32)
# total_steps = int(MAX_TIME / DT)

# for step in range(total_steps):
#     sim_time = step * DT
#     active   = [a for a in agents if not a["evacuated"]]
#     if not active:
#         break

#     # Collect positions of all active agents for repulsion calculation
#     pos_arr = np.array([[a["x"], a["y"]] for a in active], dtype=np.float32)

#     for i, agent in enumerate(active):
#         x, y   = agent["x"], agent["y"]
#         vx, vy = agent["vx"], agent["vy"]
#         ix = int(np.clip(x, 0, W - 1))
#         iy = int(np.clip(y, 0, H - 1))

#         # ── Force 1: Driving force ──────────────────────────────────────
#         # Agent wants to move at DESIRED_SPEED in the BFS direction
#         ex_dir_x = flow_x[iy, ix]
#         ex_dir_y = flow_y[iy, ix]

#         # If flow is zero, small random push to escape dead spots
#         if ex_dir_x == 0 and ex_dir_y == 0:
#             ex_dir_x = np.random.uniform(-1, 1)
#             ex_dir_y = np.random.uniform(-1, 1)
#             mag = (ex_dir_x**2 + ex_dir_y**2) ** 0.5
#             ex_dir_x /= mag
#             ex_dir_y /= mag

#         desired_vx = DESIRED_SPEED * ex_dir_x
#         desired_vy = DESIRED_SPEED * ex_dir_y

#         # Driving force = (desired_velocity - current_velocity) / tau
#         f_drive_x = (desired_vx - vx) / RELAXATION_TIME
#         f_drive_y = (desired_vy - vy) / RELAXATION_TIME

#         # ── Force 2: Agent-agent repulsion ──────────────────────────────
#         f_agent_x = 0.0
#         f_agent_y = 0.0

#         for j, other in enumerate(active):
#             if j == i:
#                 continue
#             dx = x - other["x"]
#             dy = y - other["y"]
#             dist = (dx**2 + dy**2) ** 0.5

#             if dist < 1e-3:
#                 continue

#             # Only repel if within interaction range
#             if dist < AGENT_RADIUS * 6:
#                 # Helbing: A * exp((r_ij - d) / B) * n_ij
#                 r_ij = AGENT_RADIUS * 2   # sum of radii
#                 magnitude = AA_STRENGTH * np.exp((r_ij - dist) / AA_RANGE)
#                 f_agent_x += magnitude * (dx / dist)
#                 f_agent_y += magnitude * (dy / dist)

#         # ── Force 3: Wall repulsion ──────────────────────────────────────
#         d = dist_to_wall[iy, ix]
#         f_wall_x = 0.0
#         f_wall_y = 0.0

#         if d < WA_RANGE * 2:
#             magnitude = WA_STRENGTH * np.exp(-d / WA_RANGE)
#             f_wall_x  = magnitude * wall_grad_x[iy, ix]
#             f_wall_y  = magnitude * wall_grad_y[iy, ix]

#         # ── Total force → acceleration → velocity → position ────────────
#         total_fx = f_drive_x + f_agent_x + f_wall_x
#         total_fy = f_drive_y + f_agent_y + f_wall_y

#         # Update velocity (mass = 1)
#         vx += total_fx * DT
#         vy += total_fy * DT

#         # Speed cap — panic can exceed desired but not wildly
#         speed = (vx**2 + vy**2) ** 0.5
#         max_speed = DESIRED_SPEED * 1.5
#         if speed > max_speed:
#             vx *= max_speed / speed
#             vy *= max_speed / speed

#         # Stuck detection — if barely moved, random nudge
#         buf = agent["stuck_buf"]
#         buf.append((x, y))
#         if len(buf) > STUCK_TICKS:
#             buf.pop(0)
#             travel = abs(buf[-1][0]-buf[0][0]) + abs(buf[-1][1]-buf[0][1])
#             if travel < STUCK_DIST:
#                 angle = np.random.uniform(0, 2 * np.pi)
#                 vx = np.cos(angle) * DESIRED_SPEED * 0.5
#                 vy = np.sin(angle) * DESIRED_SPEED * 0.5

#         # Try to move
#         nx_pos = np.clip(x + vx * DT, 0, W - 1)
#         ny_pos = np.clip(y + vy * DT, 0, H - 1)

#         if walkable[int(ny_pos), int(nx_pos)]:
#             agent["x"], agent["y"] = nx_pos, ny_pos
#         elif walkable[int(y), int(nx_pos)]:
#             agent["x"] = nx_pos
#             vy *= 0.5
#         elif walkable[int(ny_pos), int(x)]:
#             agent["y"] = ny_pos
#             vx *= 0.5
#         else:
#             # Fully blocked — zero velocity and wait for forces to redirect
#             vx, vy = 0.0, 0.0

#         agent["vx"], agent["vy"] = vx, vy
#         agent["trail"].append((agent["x"], agent["y"]))
#         density_map[int(agent["y"]), int(agent["x"])] += 1

#         # Exit check
#         pos   = np.array([agent["x"], agent["y"]])
#         dists = np.linalg.norm(exit_pts - pos, axis=1)
#         if dists.min() < EXIT_RADIUS:
#             agent["evacuated"] = True
#             agent["time"]      = sim_time

#     if step % 100 == 0:
#         evac = sum(a["evacuated"] for a in agents)
#         print(f"  t={sim_time:.1f}s  active={len(active)}  evacuated={evac}")

# evac_final = sum(a["evacuated"] for a in agents)
# print(f"Done  t={sim_time:.1f}s  evacuated={evac_final}/{len(agents)}")


# # ═══════════════════════════════════════════════════════════════════════
# # STEP 5 — Output
# # ═══════════════════════════════════════════════════════════════════════

# base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# if density_map.max() > 0:
#     dn   = (density_map / density_map.max() * 255).astype(np.uint8)
#     heat = cv2.applyColorMap(dn, cv2.COLORMAP_HOT)
#     mask_area = walkable.astype(np.uint8)[:, :, None]
#     base = cv2.addWeighted(base, 0.45, heat * mask_area, 0.55, 0)

# for agent in agents:
#     trail = agent["trail"]
#     color = (0, 200, 0) if agent["evacuated"] else (0, 80, 200)
#     for i in range(1, len(trail)):
#         p1 = (int(trail[i-1][0]), int(trail[i-1][1]))
#         p2 = (int(trail[i][0]),   int(trail[i][1]))
#         cv2.line(base, p1, p2, color, 1)

# for ex in exits:
#     cv2.circle(base, (int(ex["x"]), int(ex["y"])), 12, (0, 255, 255), 2)
#     cv2.putText(base, "EXIT", (int(ex["x"]) - 15, int(ex["y"]) - 16),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

# thresh      = density_map.max() * 0.65
# bn_mask     = (density_map > thresh).astype(np.uint8) * 255
# contours, _ = cv2.findContours(bn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(base, contours, -1, (0, 0, 255), 2)

# cv2.imwrite(OUT_PATHS, base)
# print(f"Saved: {OUT_PATHS}")

# total     = len(agents)
# evacuated = evac_final
# times     = [a["time"] for a in agents if a["evacuated"] and a["time"] is not None]

# lines = [
#     "=" * 50,
#     "EVACUATION ANALYSIS REPORT",
#     "=" * 50,
#     f"Total agents      : {total}",
#     f"Evacuated         : {evacuated}  ({100*evacuated/total:.1f}%)",
#     f"Trapped / timeout : {total-evacuated}   ({100*(total-evacuated)/total:.1f}%)",
#     "",
# ]
# if times:
#     lines += [
#         f"Fastest evacuation: {min(times):.1f}s",
#         f"Mean evac time    : {np.mean(times):.1f}s",
#         f"Slowest evacuated : {max(times):.1f}s",
#         "",
#     ]
# lines += [
#     f"Bottleneck zones  : {len(contours)} high-density areas",
#     "  (red outlines on image)",
#     "",
#     "VERDICT:",
#     "  GOOD — 90%+ evacuated. Layout is efficient." if evacuated/total >= 0.90
#     else "  MODERATE — 70-90% evacuated. Some bottlenecks exist." if evacuated/total >= 0.70
#     else "  POOR — Under 70% evacuated. Serious layout problems.",
#     "=" * 50,
# ]

# report = "\n".join(lines)
# print("\n" + report)
# with open(OUT_REPORT, "w") as f:
#     f.write(report)
# print(f"Saved: {OUT_REPORT}")








"""
evacuation.py
Navigation : BFS flow field
Movement   : Social Force Model

Analysis:
  1. Bottleneck scoring — top chokepoints ranked by agent-seconds lost
  2. Exit utilization — traffic share per exit
  3. Score 0-100 with single recommendation
"""

import cv2
import json
import numpy as np
from collections import deque

# ── CONFIG ──────────────────────────────────────────────────────────────
MASK_PATH   = "stitched_mask.png"
CONFIG_PATH = "stitched_mask_zone_config.json"
OUT_PATHS   = "output_paths.png"
OUT_REPORT  = "output_report.txt"

DT       = 0.05
MAX_TIME = 120

DESIRED_SPEED   = 55.0
RELAXATION_TIME = 0.3
AGENT_RADIUS    = 6
AA_STRENGTH     = 250.0
AA_RANGE        = 14.0
WA_STRENGTH     = 350.0
WA_RANGE        = 7.0
EXIT_RADIUS     = 22
STUCK_DIST      = 2
STUCK_TICKS     = 20
# ────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════
# STEP 1 — Load mask
# ═══════════════════════════════════════════════════════════════════════

img          = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
H, W         = img.shape
walkable     = img < 128
walk_u8      = walkable.astype(np.uint8) * 255
dist_to_wall = cv2.distanceTransform(walk_u8, cv2.DIST_L2, 5)

wall_grad_y, wall_grad_x = np.gradient(dist_to_wall.astype(np.float32))
wall_grad_mag = np.sqrt(wall_grad_x**2 + wall_grad_y**2) + 1e-8
wall_grad_x  /= wall_grad_mag
wall_grad_y  /= wall_grad_mag


# ═══════════════════════════════════════════════════════════════════════
# STEP 2 — BFS flow field
# ═══════════════════════════════════════════════════════════════════════

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

exits = cfg["exits"]

cost   = np.full((H, W), -1, dtype=np.int32)
flow_x = np.zeros((H, W), dtype=np.float32)
flow_y = np.zeros((H, W), dtype=np.float32)

queue = deque()
for ex in exits:
    ex_x, ex_y = int(ex["x"]), int(ex["y"])
    for dy in range(-EXIT_RADIUS, EXIT_RADIUS + 1):
        for dx in range(-EXIT_RADIUS, EXIT_RADIUS + 1):
            nx, ny = ex_x + dx, ex_y + dy
            if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx] and cost[ny, nx] == -1:
                cost[ny, nx] = 0
                queue.append((nx, ny))

DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

while queue:
    cx, cy = queue.popleft()
    for dx, dy in DIRS:
        nx, ny = cx + dx, cy + dy
        if 0 <= nx < W and 0 <= ny < H and walkable[ny, nx] and cost[ny, nx] == -1:
            cost[ny, nx] = cost[cy, cx] + 1
            queue.append((nx, ny))

for y in range(H):
    for x in range(W):
        if not walkable[y, x] or cost[y, x] == -1:
            continue
        best = cost[y, x]
        bx = by = 0.0
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and 0 <= cost[ny, nx] < best:
                best = cost[ny, nx]
                bx, by = float(dx), float(dy)
        mag = (bx**2 + by**2) ** 0.5
        if mag > 0:
            flow_x[y, x] = bx / mag
            flow_y[y, x] = by / mag

print(f"Flow field built. Reachable cells: {(cost >= 0).sum()}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 3 — Spawn agents
# ═══════════════════════════════════════════════════════════════════════

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

dist_norm  = cv2.normalize(dist_to_wall, None, 0, 1.0, cv2.NORM_MINMAX)
coords     = peak_local_max(dist_norm, min_distance=40, labels=walk_u8)
seed_mask  = np.zeros(dist_norm.shape, dtype=bool)
seed_mask[tuple(coords.T)] = True
markers, _ = ndi.label(seed_mask)
labels     = watershed(-dist_to_wall, markers, mask=walk_u8)

zone_pixels = {}
for zdata in cfg["zones"]:
    zid    = zdata["zone_id"]
    ys, xs = np.where(labels == zid)
    valid  = (dist_to_wall[ys, xs] > AGENT_RADIUS) & (cost[ys, xs] >= 0)
    if valid.any():
        zone_pixels[zid] = list(zip(xs[valid].tolist(), ys[valid].tolist()))

agents = []
for zdata in cfg["zones"]:
    n      = zdata["agents"]
    zid    = zdata["zone_id"]
    pixels = zone_pixels.get(zid, [])
    if n == 0 or not pixels:
        continue
    chosen = [pixels[i] for i in np.random.choice(len(pixels), min(n, len(pixels)), replace=False)]
    for px, py in chosen:
        agents.append({
            "x": float(px), "y": float(py),
            "vx": 0.0,      "vy": 0.0,
            "evacuated": False, "time": None,
            "exit_used": None,              # which exit index this agent used
            "trail": [(float(px), float(py))],
            "stuck_buf": [],
        })

print(f"Spawned {len(agents)} agents")


# ═══════════════════════════════════════════════════════════════════════
# STEP 4 — Simulation
# ═══════════════════════════════════════════════════════════════════════

exit_pts    = np.array([[e["x"], e["y"]] for e in exits], dtype=np.float32)
density_map = np.zeros((H, W), dtype=np.float32)

# Congestion map: accumulates agent-ticks per cell
# We use this to measure how long agents spent in each area
congestion_map = np.zeros((H, W), dtype=np.float32)

total_steps = int(MAX_TIME / DT)

for step in range(total_steps):
    sim_time = step * DT
    active   = [a for a in agents if not a["evacuated"]]
    if not active:
        break

    for i, agent in enumerate(active):
        x, y   = agent["x"], agent["y"]
        vx, vy = agent["vx"], agent["vy"]
        ix = int(np.clip(x, 0, W - 1))
        iy = int(np.clip(y, 0, H - 1))

        # Force 1: Driving
        ex_dir_x = flow_x[iy, ix]
        ex_dir_y = flow_y[iy, ix]
        if ex_dir_x == 0 and ex_dir_y == 0:
            a = np.random.uniform(0, 2 * np.pi)
            ex_dir_x, ex_dir_y = np.cos(a), np.sin(a)

        f_drive_x = (DESIRED_SPEED * ex_dir_x - vx) / RELAXATION_TIME
        f_drive_y = (DESIRED_SPEED * ex_dir_y - vy) / RELAXATION_TIME

        # Force 2: Agent repulsion
        f_agent_x = f_agent_y = 0.0
        for j, other in enumerate(active):
            if j == i:
                continue
            dx = x - other["x"]
            dy = y - other["y"]
            dist = (dx**2 + dy**2) ** 0.5
            if dist < 1e-3 or dist >= AGENT_RADIUS * 6:
                continue
            mag = AA_STRENGTH * np.exp((AGENT_RADIUS * 2 - dist) / AA_RANGE)
            f_agent_x += mag * dx / dist
            f_agent_y += mag * dy / dist

        # Force 3: Wall repulsion
        d = dist_to_wall[iy, ix]
        f_wall_x = f_wall_y = 0.0
        if d < WA_RANGE * 2:
            mag = WA_STRENGTH * np.exp(-d / WA_RANGE)
            f_wall_x = mag * wall_grad_x[iy, ix]
            f_wall_y = mag * wall_grad_y[iy, ix]

        vx += (f_drive_x + f_agent_x + f_wall_x) * DT
        vy += (f_drive_y + f_agent_y + f_wall_y) * DT

        spd = (vx**2 + vy**2) ** 0.5
        if spd > DESIRED_SPEED * 1.5:
            vx *= DESIRED_SPEED * 1.5 / spd
            vy *= DESIRED_SPEED * 1.5 / spd

        # Stuck check
        buf = agent["stuck_buf"]
        buf.append((x, y))
        if len(buf) > STUCK_TICKS:
            buf.pop(0)
            if abs(buf[-1][0]-buf[0][0]) + abs(buf[-1][1]-buf[0][1]) < STUCK_DIST:
                a = np.random.uniform(0, 2 * np.pi)
                vx = np.cos(a) * DESIRED_SPEED * 0.5
                vy = np.sin(a) * DESIRED_SPEED * 0.5

        nx_pos = np.clip(x + vx * DT, 0, W - 1)
        ny_pos = np.clip(y + vy * DT, 0, H - 1)

        if   walkable[int(ny_pos), int(nx_pos)]: agent["x"], agent["y"] = nx_pos, ny_pos
        elif walkable[int(y),      int(nx_pos)]: agent["x"] = nx_pos;  vy *= 0.5
        elif walkable[int(ny_pos), int(x)     ]: agent["y"] = ny_pos;  vx *= 0.5
        else: vx, vy = 0.0, 0.0

        agent["vx"], agent["vy"] = vx, vy
        agent["trail"].append((agent["x"], agent["y"]))

        cell_x = int(agent["x"])
        cell_y = int(agent["y"])
        density_map[cell_y, cell_x]    += 1
        congestion_map[cell_y, cell_x] += DT   # seconds spent here

        # Exit check — record WHICH exit
        pos   = np.array([agent["x"], agent["y"]])
        dists = np.linalg.norm(exit_pts - pos, axis=1)
        nearest_exit = int(np.argmin(dists))
        if dists[nearest_exit] < EXIT_RADIUS:
            agent["evacuated"] = True
            agent["time"]      = sim_time
            agent["exit_used"] = nearest_exit

    if step % 100 == 0:
        evac = sum(a["evacuated"] for a in agents)
        print(f"  t={sim_time:.1f}s  active={len(active)}  evacuated={evac}")

evac_final = sum(a["evacuated"] for a in agents)
print(f"Done  t={sim_time:.1f}s  evacuated={evac_final}/{len(agents)}")


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 1 — Bottleneck scoring
# Find high-congestion blobs, rank by total agent-seconds lost there
# ═══════════════════════════════════════════════════════════════════════

# Threshold: cells where congestion is in top 20%
walkable_congestion = congestion_map[walkable]
if walkable_congestion.max() > 0:
    threshold = np.percentile(walkable_congestion[walkable_congestion > 0], 80)
else:
    threshold = 1.0

bn_mask     = (congestion_map > threshold).astype(np.uint8) * 255
contours, _ = cv2.findContours(bn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bottlenecks = []
for cnt in contours:
    # bounding box of this blob
    bx, by, bw, bh = cv2.boundingRect(cnt)
    # total agent-seconds spent inside this blob
    region_congestion = congestion_map[by:by+bh, bx:bx+bw].sum()
    # narrowness: how tight is this corridor
    # use the minimum dimension of the bounding box as a proxy for width
    narrowness = min(bw, bh)
    # centroid
    M = cv2.moments(cnt)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = bx + bw//2, by + bh//2
    bottlenecks.append({
        "cx": cx, "cy": cy,
        "agent_seconds": float(region_congestion),
        "width_px": narrowness,
    })

# Sort by agent-seconds lost (worst first)
bottlenecks.sort(key=lambda b: b["agent_seconds"], reverse=True)
top_bottlenecks = bottlenecks[:5]


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 2 — Exit utilization
# ═══════════════════════════════════════════════════════════════════════

exit_counts = [0] * len(exits)
for a in agents:
    if a["evacuated"] and a["exit_used"] is not None:
        exit_counts[a["exit_used"]] += 1

total_evacuated = sum(exit_counts)


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 3 — Score 0-100
# Components:
#   - Evacuation rate         (0-50 pts)
#   - Mean evacuation time    (0-20 pts)  faster = better
#   - Exit balance            (0-15 pts)  even distribution = better
#   - Bottleneck severity     (0-15 pts)  less congestion = better
# ═══════════════════════════════════════════════════════════════════════

total  = len(agents)
times  = [a["time"] for a in agents if a["evacuated"] and a["time"] is not None]
rate   = evac_final / total

# Component 1: evacuation rate
score_rate = rate * 50

# Component 2: mean time (cap at MAX_TIME for non-evacuated)
if times:
    mean_t = np.mean(times)
    # 20 pts if mean < 20s, 0 pts if mean > 80s
    score_time = max(0, 20 * (1 - (mean_t - 20) / 60)) if mean_t > 20 else 20.0
else:
    score_time = 0.0

# Component 3: exit balance (how evenly distributed is traffic)
if total_evacuated > 0:
    fractions = [c / total_evacuated for c in exit_counts]
    ideal     = 1.0 / len(exits)
    # max deviation from ideal
    max_dev   = max(abs(f - ideal) for f in fractions)
    # 0 dev = 15 pts, dev = 1.0 = 0 pts
    score_balance = max(0, 15 * (1 - max_dev / ideal))
else:
    score_balance = 0.0

# Component 4: bottleneck severity
# measure as fraction of total agent-time spent in bottlenecks
total_agent_time  = congestion_map.sum()
bn_agent_time     = sum(b["agent_seconds"] for b in top_bottlenecks)
bn_fraction       = bn_agent_time / (total_agent_time + 1e-8)
# 0.05 fraction = 15 pts, 0.5 fraction = 0 pts
score_bn          = max(0, 15 * (1 - (bn_fraction - 0.05) / 0.45))

final_score = int(score_rate + score_time + score_balance + score_bn)
final_score = min(100, max(0, final_score))

# Single recommendation based on worst component
worst = min(
    ("evacuation_rate", score_rate / 50),
    ("exit_balance",    score_balance / 15),
    ("bottlenecks",     score_bn / 15),
    ("evac_time",       score_time / 20),
    key=lambda x: x[1]
)

# Find worst bottleneck position for the recommendation
if top_bottlenecks:
    wb = top_bottlenecks[0]
    bn_pos = f"({wb['cx']}, {wb['cy']})"
else:
    bn_pos = "unknown"

# Find most underused exit
if total_evacuated > 0:
    min_exit_idx  = int(np.argmin(exit_counts))
    min_exit_pos  = f"({int(exits[min_exit_idx]['x'])}, {int(exits[min_exit_idx]['y'])})"
else:
    min_exit_idx  = 0
    min_exit_pos  = "N/A"

RECOMMENDATIONS = {
    "evacuation_rate": f"Too many agents failed to evacuate — check for isolated rooms with no path to any exit.",
    "exit_balance":    f"Exit {min_exit_idx} at {min_exit_pos} handled almost no traffic. Consider repositioning it closer to high-density zones or adding signage.",
    "bottlenecks":     f"The corridor at {bn_pos} is your biggest chokepoint. Widen it or add a parallel route.",
    "evac_time":       f"Evacuation is too slow. Add an exit closer to the center of the building.",
}
recommendation = RECOMMENDATIONS[worst[0]]


# ═══════════════════════════════════════════════════════════════════════
# STEP 5 — Output image
# ═══════════════════════════════════════════════════════════════════════

base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Density heatmap
if density_map.max() > 0:
    dn   = (density_map / density_map.max() * 255).astype(np.uint8)
    heat = cv2.applyColorMap(dn, cv2.COLORMAP_HOT)
    base = cv2.addWeighted(base, 0.45, heat * walkable.astype(np.uint8)[:,:,None], 0.55, 0)

# Agent trails
for agent in agents:
    color = (0, 200, 0) if agent["evacuated"] else (0, 80, 200)
    trail = agent["trail"]
    for i in range(1, len(trail)):
        cv2.line(base,
                 (int(trail[i-1][0]), int(trail[i-1][1])),
                 (int(trail[i][0]),   int(trail[i][1])),
                 color, 1)

# Exits — labeled with utilization %
for idx, ex in enumerate(exits):
    pct   = (exit_counts[idx] / total_evacuated * 100) if total_evacuated > 0 else 0
    color = (0, 255, 255)
    cv2.circle(base, (int(ex["x"]), int(ex["y"])), 12, color, 2)
    cv2.putText(base, f"E{idx} {pct:.0f}%",
                (int(ex["x"]) - 18, int(ex["y"]) - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

# Top bottlenecks — numbered red circles
for rank, bn in enumerate(top_bottlenecks):
    cv2.circle(base, (bn["cx"], bn["cy"]), 14, (0, 0, 255), 2)
    cv2.putText(base, f"B{rank+1}",
                (bn["cx"] - 8, bn["cy"] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

# Score box top-left
score_color = (0,200,0) if final_score >= 80 else (0,180,255) if final_score >= 60 else (0,0,255)
cv2.rectangle(base, (4, 4), (130, 30), (0, 0, 0), -1)
cv2.putText(base, f"SCORE: {final_score}/100",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color, 1)

cv2.imwrite(OUT_PATHS, base)
print(f"Saved: {OUT_PATHS}")


# ═══════════════════════════════════════════════════════════════════════
# STEP 6 — Report
# ═══════════════════════════════════════════════════════════════════════

lines = [
    "=" * 55,
    "       EVACUATION ANALYSIS REPORT",
    "=" * 55,
    "",
    f"  OVERALL SCORE : {final_score} / 100",
    "",
    f"  Total agents    : {total}",
    f"  Evacuated       : {evac_final}  ({100*rate:.1f}%)",
    f"  Trapped/timeout : {total - evac_final}  ({100*(1-rate):.1f}%)",
]

if times:
    lines += [
        f"  Fastest evac    : {min(times):.1f}s",
        f"  Mean evac time  : {np.mean(times):.1f}s",
        f"  Slowest evac    : {max(times):.1f}s",
    ]

lines += [
    "",
    "-" * 55,
    "  EXIT UTILIZATION",
    "-" * 55,
]
for idx, ex in enumerate(exits):
    pct    = (exit_counts[idx] / total_evacuated * 100) if total_evacuated > 0 else 0
    bar    = "█" * int(pct / 5)
    status = "⚠ UNDERUSED" if pct < (100 / len(exits) * 0.4) else ""
    lines.append(f"  Exit {idx} ({int(ex['x'])},{int(ex['y'])}): {exit_counts[idx]:3d} agents  {pct:5.1f}%  {bar} {status}")

lines += [
    "",
    "-" * 55,
    "  TOP BOTTLENECKS  (ranked by agent-seconds lost)",
    "-" * 55,
]
for rank, bn in enumerate(top_bottlenecks):
    lines.append(
        f"  B{rank+1}  position ({bn['cx']:4d},{bn['cy']:4d})  "
        f"corridor width ~{bn['width_px']}px  "
        f"{bn['agent_seconds']:.0f} agent-seconds"
    )

lines += [
    "",
    "-" * 55,
    "  RECOMMENDATION",
    "-" * 55,
    f"  {recommendation}",
    "",
    "  Score breakdown:",
    f"    Evacuation rate  : {score_rate:.0f}/50",
    f"    Evacuation speed : {score_time:.0f}/20",
    f"    Exit balance     : {score_balance:.0f}/15",
    f"    Bottleneck sev.  : {score_bn:.0f}/15",
    "=" * 55,
]

report = "\n".join(lines)
print("\n" + report)
with open(OUT_REPORT, "w") as f:
    f.write(report)
print(f"\nSaved: {OUT_REPORT}")