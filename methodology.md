# Methodology: Simulation Model Implementations

TRAGIC implements four independent crowd-evacuation models. Each is grounded in
a specific published model rather than invented from scratch. This section
documents, for each model: the reference, its core formulation, what TRAGIC
actually implements, and where/why the implementation deviates from the paper.

A pattern repeats across all four: every reference paper assumes either a
simple open domain or an idealized exit geometry. None of them ship a
floorplan-aware navigation layer for arbitrary building shapes. TRAGIC adds
the same fix to all four — a wall-aware shortest-path field (BFS or fast
marching) computed once from the segmented floorplan mask — and a fire/hazard
repulsion term, neither of which exists in any of the four source papers.
That layer is TRAGIC's own contribution sitting on top of the literature, and
it is worth stating explicitly rather than leaving implicit.

---

## 1. Social Force Model — `SFM_evacuation.py`

**Reference:** Helbing, D., Farkas, I., Vicsek, T. (2000). *Simulating
dynamical features of escape panic.* Nature 407, 487–490.
(Summary used: pedestriandynamics.org/models/social_force_model)

**Core formulation.** The model sums three forces per agent $i$:

Driving force, pulling the agent toward its desired velocity:

$$\vec{F_i^{\mathrm{drv}}} = \frac{v_i^0 \vec{e_i^0} - \vec{v_i}}{\tau}$$

Agent–agent repulsion, exponential with distance, plus a contact term and a
tangential friction term that activate only on physical overlap:

$$\vec{f_{ij}} = \big[A_i\exp[(r_{ij}-d_{ij})/B_i] + k\,g(r_{ij}-d_{ij})\big]\vec{n_{ij}} + \kappa\,g(r_{ij}-d_{ij})\,\Delta v_{ji}^t\,\vec{t_{ij}}$$

Obstacle force, the same structure applied to the nearest point on each wall
segment instead of a neighboring agent.

**What TRAGIC implements.** The driving force is implemented exactly
(`f_drive = (DESIRED_SPEED*ex_dir - vel) / RELAXATION_TIME`). Agent–agent and
wall repulsion implement only the exponential *pushing* term — the contact
compression term ($k\,g(\cdot)$) and the tangential *sliding/friction* term
($\kappa\,g(\cdot)\Delta v^t$) are both omitted.

**Deviations & why:**
- **No friction/sliding term.** This is the biggest simplification. It means
  TRAGIC's agents don't reproduce the "shoving"/stop-and-go waves seen in
  real crowd-crush footage — only the isotropic exponential push. Acceptable
  for the density ranges this project targets (building evacuation, not
  stadium-crush analysis), but worth stating as a known limit.
- **Wall force via precomputed distance field, not segment geometry.**
  Helbing's obstacle force projects onto the *closest point on each wall
  line segment* analytically. TRAGIC instead precomputes a Euclidean
  distance-transform field once (`dist_to_wall`) and reads its gradient per
  agent per step. Same exponential-decay shape, O(1) lookup regardless of
  wall complexity, instead of iterating wall segments per agent per step —
  a performance trade-off, not a behavioral one.
- **No explicit mass term.** Helbing divides $(F_{rep}+F_{obst})$ by agent
  mass $m_i$; TRAGIC's code has no mass variable (effectively $m_i=1$ for
  everyone). Drops heterogeneous-body-mass effects, not used by this project.
- **Desired direction $\vec{e_i^0}$ replaced by a flow field.** The base
  model assumes a straight line toward a single goal point. TRAGIC replaces
  this with the BFS flow field (`build_flow_field`), so the *driving* force
  still follows Helbing's equation, but the direction it drives toward is
  now wall-aware instead of straight-line.
- **Added:** fire/hazard repulsion force ($F_{fire}$, gradient of the fire
  intensity field) — not present in the 2000 paper at all.

---

## 2. Reciprocal Velocity Obstacles (RVO/ORCA) — `RVO_evacuation.py`

**Reference:** Bera, A., Manocha, D. (2014). *Realtime Multilevel Crowd
Tracking using Reciprocal Velocity Obstacles.* ICPR 2014.

**Honesty note on this reference.** This paper is not an evacuation or
crowd-generation paper — it's a pedestrian *tracking* paper. Its actual
contribution is a multi-level particle filter that fits RVO parameters to
real video footage (using an Ensemble Kalman Filter + EM for parameter
estimation), so that a tracker can predict where each visible pedestrian
will go next. The only part of that paper TRAGIC borrows is the underlying
*motion model itself* — the ORCA velocity-selection formulation that the
paper uses as its prediction function — not the tracking, confidence
estimation, or video-fitting machinery, which solves a different problem
(inferring motion from observed pixels) than the one TRAGIC has (generating
motion from a known floorplan with known parameters).

**Core formulation (the part actually used).** Given a preferred velocity
$v_{pref}$, the new collision-free velocity is the closest feasible point to
$v_{pref}$ outside all pairwise velocity obstacles:

$$v_{RVO} = \arg\max_{v \in \mathrm{ORCA}} \lVert v - v_{pref}\rVert$$

where each neighboring agent induces a half-plane constraint on velocity
space (the standard ORCA half-plane from van den Berg et al., reproduced
inside Bera & Manocha as their motion model).

**What TRAGIC implements:** `orca_halfplane()` builds the same truncated
velocity-obstacle apex/half-plane construction; `resolve_velocity()`
projects the preferred velocity against all active half-planes.

**Deviations & why:**
- **Heuristic projection instead of a true incremental LP.** The actual ORCA
  formulation solves velocity selection as a proper 2D linear program over
  all half-plane constraints. TRAGIC's `resolve_velocity()` is a simpler
  iterative tangent-line projection, not a full LP solver. Cheaper, but can
  produce a slightly worse (not provably optimal) velocity under heavy
  constraint load.
- **Minimum-speed floor (not in ORCA).** Under that simplified resolver,
  agents could converge to near-zero velocity and freeze. TRAGIC adds a
  floor of ~15% of preferred speed whenever a nonzero preferred direction
  exists, purely to prevent deadlock — an engineering patch, not part of
  the cited model.
- **Static obstacles handled by a separate flow field, not as VO geometry.**
  ORCA's original formulation treats static obstacles as additional
  geometric constraints inside velocity-obstacle space. TRAGIC instead
  routes the preferred velocity through a precomputed wall-aware BFS flow
  field and lets ORCA only resolve *agent–agent* collisions — a layered
  design rather than the original's unified treatment.
- **Added:** the same fire/hazard repulsion and hazard-avoidance fallback
  used in SFM.

---

## 3. Continuum Crowds — `continuum_evacuation_path.py`

**Reference:** Treuille, A., Cooper, S., Popović, Z. (2006). *Continuum
Crowds.* ACM Transactions on Graphics (SIGGRAPH) 25(3), 1160–1168.

**Core formulation.** Crowd motion is modeled as a per-particle energy
minimization over a continuum, not per-agent forces. Speed depends on local
density, interpolated between a free-flow ("topographical") speed and a
flow-following speed once density exceeds a threshold:

$$f(x,\theta) = f_T(x,\theta) + \left(\frac{\rho(x+r n_\theta)-\rho_{min}}{\rho_{max}-\rho_{min}}\right)\big(f_{\bar v}(x,\theta) - f_T(x,\theta)\big)$$

A unit cost field combines path length, time, and discomfort:

$$C = \frac{\alpha f + \beta + \gamma g}{f}$$

and the potential field $\phi$ satisfies an eikonal equation
$\lVert\nabla\phi\rVert = C$, solved with the fast marching method; agents
then move opposite the potential gradient, scaled by local speed.

**What TRAGIC implements:** this is the closest 1:1 match of the four. The
`CFG` dict (`alpha`, `beta`, `rho_min`, `rho_max`, `density_radius`) maps
directly onto the paper's variables, and `_speed()` / `_cost()` implement
Equations 8–10 essentially as written. `build_phi()` is a Dijkstra-style
heap solve of the same eikonal cost field.

**Deviations & why:**
- **4-connected grid instead of anisotropic 4-direction-per-cell fast
  marching.** The paper's fast-marching solve is genuinely anisotropic —
  cost and speed are stored per cell *per compass direction* (E/N/W/S), and
  the potential update solves a quadratic over both axes (Eq. 11) to
  correctly handle direction-dependent cost. TRAGIC's `build_phi()` instead
  treats cost as isotropic per cell and does a standard 4-neighbor
  Dijkstra-style relaxation. Simpler and faster, at the cost of some
  grid-axis bias in paths (a known artifact of non-anisotropic grid solves).
- **No discomfort field.** $g(x)$ is set to zero everywhere; TRAGIC has no
  mechanism for "preferred but not fastest" routing (e.g., avoiding a
  visually unpleasant but technically faster route). Not needed for
  evacuation scoring, so left out.
- **No predictive discomfort.** The paper's Section 3.3 extension — agents
  project their own future position forward and deposit discomfort ahead of
  themselves to reduce head-on collisions — is not implemented.
- **Minimum-distance enforcement replaced with a continuous repulsion
  force.** The paper enforces minimum separation as a discrete post-step
  correction (symmetric push-apart pass over nearby pairs). TRAGIC instead
  folds a SFM-style exponential repulsion force directly into velocity
  integration (`f_rep_x/y`). Functionally similar outcome, different and
  more SFM-like mechanism.
- **Field rebuilt every 30 ticks, not every tick.** Pure performance
  trade-off; the paper assumes per-frame rebuilds.
- **Added:** fire repulsion force layered the same way as in SFM/RVO.

---

## 4. Cellular Automata — `CA_evacuation.py`

**Reference:** Dhaliwal, A. S., Ghosh, A., Mansukhani, N. (2020). *A
Cellular Automata Model for Predicting Crowd Movement during Evacuation.*
Azim Premji University At Right Angles, November 2020, pp. 53–62.

**Note on this reference.** This is a secondary-school enrichment-magazine
article, not a peer-reviewed research paper — worth stating plainly rather
than implying otherwise. It's still a legitimate methodological source: it
describes a specific, citable rule set (Moore-neighbourhood movement with
exit-priority conflict resolution) that TRAGIC's CA model is recognizably
built from.

**Core rules (the paper's "Scenario 2," collision-free variant).** Each
occupied cell may move to one of its 8 Moore neighbors per tick. A cell
picks the neighbor that minimizes (Euclidean) distance to the exit, cannot
move further from the exit, and conflicts are resolved by giving movement
priority to whichever competing cell is closer to the exit (ties broken by
a fixed label order). This guarantees zero collisions but requires
recomputing a global priority order every tick.

**What TRAGIC implements:** Moore-neighborhood stepping with an occupancy
grid blocking movement into already-occupied cells (`occupied[ny, nx]`) —
same family of model, same neighborhood structure.

**Deviations & why:**
- **BFS path-cost field instead of raw Euclidean distance.** The reference
  paper's exit-distance metric is straight-line distance — fine for their
  simple rectangular rooms/corridors with exits defined as a literal shared
  wall edge, but it has no concept of walls blocking a straight line. TRAGIC
  replaces this with the same wall-aware BFS cost grid used by the other
  three models, so a cell always moves toward *lower path cost*, not lower
  straight-line distance. This is the one deviation that's a strict
  improvement rather than a simplification — it's required for the model
  to work on arbitrary floorplans at all.
- **Random sequential update instead of distance-priority ordering.** The
  paper's collision-free guarantee comes from processing cells in strict
  order of exit-closeness every tick. TRAGIC instead shuffles agent order
  randomly each tick (`np.random.shuffle(active)`) and resolves conflicts
  via the occupancy grid (first agent to claim a cell wins). This is a
  *different*, more common CA update scheme (random sequential update,
  as used in most floor-field CA literature) rather than the paper's
  specific deterministic priority rule — it no longer guarantees zero
  collisions, but is simpler and avoids recomputing a global ranking
  every tick.
- **Multi-pixel sub-stepping per tick.** The paper's model moves exactly one
  cell per tick by construction (cell size = step size). TRAGIC's grid is in
  raw pixels, so `step_size` sub-steps are taken per tick to map a
  `desired_speed` (px/s) onto the grid — a discretization detail the
  original model didn't need to handle.
- **Added:** a `randomness` parameter for stochastic non-optimal steps
  (absent from the paper, which is either fully greedy or fully
  conflict-free with no in-between), and fire intensity as a secondary
  sort key when picking between equally-good candidate cells.

---

## References

1. Helbing, D., Farkas, I., Vicsek, T. (2000). Simulating dynamical features
   of escape panic. *Nature*, 407, 487–490.
2. Bera, A., Manocha, D. (2014). Realtime Multilevel Crowd Tracking using
   Reciprocal Velocity Obstacles. *ICPR 2014*.
3. van den Berg, J., Guy, S. J., Lin, M., Manocha, D. (2011). Reciprocal
   n-body collision avoidance. *Robotics Research*, 3–19. (Source of the
   ORCA formulation used inside Ref. 2.)
4. Treuille, A., Cooper, S., Popović, Z. (2006). Continuum Crowds. *ACM
   Transactions on Graphics (SIGGRAPH)*, 25(3), 1160–1168.
5. Dhaliwal, A. S., Ghosh, A., Mansukhani, N. (2020). A Cellular Automata
   Model for Predicting Crowd Movement during Evacuation. *Azim Premji
   University At Right Angles*, November 2020, 53–62.