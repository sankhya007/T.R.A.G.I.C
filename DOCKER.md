# Running TRAGIC in Docker

Docker packages your app + all its dependencies into an isolated box called a **container**.  
No Python setup needed on the other person's machine. They just need Docker installed.

---

## Before you start

You need:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- `stitched_mask.png` and `zone_config.json` in the project folder

Rename `requirements.txt` to `requirements_original.txt`, then rename `requirements_docker.txt` to `requirements.txt`.  
The Docker version strips out PyQt6 since the container has no screen.

---

## Step 1 — Build the image

```bash
docker compose build
```

This reads the `Dockerfile` and creates an image (think of it as a snapshot of a Linux machine with all your deps installed).  
You only need to do this once, or whenever you change `requirements.txt`.

---

## Step 2 — Run a simulation

```bash
docker compose up
```

This runs SFM by default. Output lands in `./output/` on your machine.

To run a different model:

```bash
docker compose run tragic python RVO_evacuation.py stitched_mask.png zone_config.json
docker compose run tragic python CA_evacuation.py stitched_mask.png zone_config.json
docker compose run tragic python continuum_evacuation_path.py stitched_mask.png zone_config.json
```

---

## Step 3 — Get a shell inside the container (useful for debugging)

```bash
docker compose run tragic bash
```

You're now inside the container. Run anything you want. Type `exit` to leave.

---

## What each file does

| File | What it is |
|---|---|
| `Dockerfile` | Recipe for building the container image |
| `docker-compose.yml` | Shortcut config so you don't type long `docker run` commands |
| `requirements_docker.txt` | Deps without PyQt6 (the container has no display) |

---

## Why no GUI in the container?

Containers don't have a screen. The launcher (`Tragic_launcher.py`) needs PyQt6 which needs a display.  
The simulation scripts (`SFM_evacuation.py`, etc.) run fine without any of that — they just read files and write output.  
Use the `.exe` release for the GUI. Use Docker for running simulations programmatically or on a server.
