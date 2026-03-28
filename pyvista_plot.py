import os
import numpy as np
import pyvista as pv

# ============================================================
# SETTINGS
# ============================================================
FILENAME = "lorenz.txt"

FPS = 30
PLAYBACK_SPEED = 2.0
WINDOW_SIZE = (1400, 900)
BACKGROUND_COLOR = "white"
SEEK_SECONDS = 1.0

# Full faint trajectory
SHADOW_COLOR = "#a9b8c9"
SHADOW_OPACITY = 0.05
SHADOW_LINE_WIDTH = 1

# Active trajectory
ACTIVE_CMAP = "inferno"
ACTIVE_LINE_WIDTH = 6

# Moving point
POINT_SIZE = 6
POINT_COLOR = "#f5f5f5"
POINT_EDGE_COLOR = "#3a4654"
POINT_EDGE_SIZE = 11
POINT_EDGE_OPACITY = 0.28

# Camera
ROTATE_CAMERA = True
CAMERA_AZIMUTH_SPEED = 3.0

# Walls / grid
WALL_COLOR = "#edf2f7"
WALL_OPACITY = 0.10
GRID_COLOR = "#dbe4ee"
GRID_OPACITY = 0.18
GRID_LINE_WIDTH = 1.0
GRID_RESOLUTION = 14

PAD_FRAC = 0.16

# GIF export
GIF_WINDOW_SIZE = (600, 600)
SAVE_GIF = True
GIF_NAME = "lorenz_clip_nice.gif"
GIF_FPS = 15
GIF_START_TIME = 32.0      # simulation time where gif starts
GIF_DURATION = 5.0        # simulation-time length of gif
GIF_EXPORT_STRIDE = 20


# ============================================================
# HELPERS
# ============================================================

def get_line_data(idx: int, stride: int = 1):
    idx = max(2, idx)

    if stride > 1:
        line_xyz = xyz[:idx:stride].copy()
        line_t = t[:idx:stride]

        # always include latest endpoint
        if len(line_t) == 0 or line_t[-1] != t[idx - 1]:
            line_xyz = np.vstack([line_xyz, xyz[idx - 1]])
            line_t = np.append(line_t, t[idx - 1])
    else:
        line_xyz = xyz[:idx].copy()
        line_t = t[:idx]

    if len(line_xyz) < 2:
        line_xyz = xyz[:2].copy()
        line_t = t[:2]

    return line_xyz, line_t


def make_polyline(points: np.ndarray) -> pv.PolyData:
    poly = pv.PolyData(points)
    n = len(points)
    if n >= 2:
        poly.lines = np.hstack(([n], np.arange(n, dtype=np.int64)))
    return poly


def interp_point(tt, t, xyz):
    x = np.interp(tt, t, xyz[:, 0])
    y = np.interp(tt, t, xyz[:, 1])
    z = np.interp(tt, t, xyz[:, 2])
    return np.array([[x, y, z]], dtype=float)


def make_quad(p0, p1, p2, p3):
    points = np.array([p0, p1, p2, p3], dtype=float)
    faces = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    return pv.PolyData(points, faces)


def make_structured_grid_on_plane(origin, uvec, vvec, nu=12, nv=12):
    segments = []

    for j in range(nv + 1):
        b = j / nv
        p0 = origin + b * vvec
        p1 = origin + uvec + b * vvec
        segments.append(np.vstack([p0, p1]))

    for i in range(nu + 1):
        a = i / nu
        p0 = origin + a * uvec
        p1 = origin + a * uvec + vvec
        segments.append(np.vstack([p0, p1]))

    all_points = []
    all_lines = []
    idx = 0
    for seg in segments:
        all_points.extend(seg)
        all_lines.extend([2, idx, idx + 1])
        idx += 2

    poly = pv.PolyData(np.array(all_points, dtype=float))
    poly.lines = np.array(all_lines, dtype=np.int64)
    return poly

# ============================================================
# GIF EXPORT FUNCTIONALITY
# ============================================================

def downsample_end_index(idx: int, stride: int, min_points: int = 2) -> int:
    if stride <= 1:
        return max(min_points, idx)
    return max(min_points, ((idx - 1) // stride) + 1)

def sim_time_to_frame(sim_t: float) -> int:
    sim_t = max(0.0, min(sim_t, TMAX))
    progress = sim_t / TMAX
    return int(round(progress * (n_frames - 1)))

def export_gif(filename, start_time, duration, gif_fps):
    start_time = max(0.0, min(start_time, TMAX))
    end_time = max(start_time, min(start_time + duration, TMAX))

    if end_time <= start_time:
        raise ValueError("GIF duration must give a non-empty interval.")

    gif_nframes = max(2, int(round(duration * gif_fps)))
    gif_times = np.linspace(start_time, end_time, gif_nframes)

    was_playing = state["playing"]
    old_frame = state["frame"]

    state["playing"] = False

    camera_step = CAMERA_AZIMUTH_SPEED / gif_fps

    # swap in lighter shadow for GIF export
    shadow_xyz = xyz[::GIF_EXPORT_STRIDE].copy()
    if len(shadow_xyz) < 2:
        shadow_xyz = xyz[:2].copy()
    gif_shadow = make_polyline(shadow_xyz)
    shadow_actor.mapper.dataset.copy_from(gif_shadow)

    pl.open_gif(filename)

    for sim_t in gif_times:
        frame_idx = sim_time_to_frame(sim_t)
        state["frame"] = frame_idx

        update_scene(
            frame_idx,
            rotate_camera=False,
            do_render=False,
            line_stride=GIF_EXPORT_STRIDE,
        )

        if ROTATE_CAMERA:
            pl.camera.Azimuth(camera_step)

        update_visible_walls()
        pl.render()
        pl.write_frame()

    pl.close()

    # restore full-resolution shadow after export
    full_shadow = make_polyline(xyz)
    shadow_actor.mapper.dataset.copy_from(full_shadow)

    state["playing"] = was_playing
    state["frame"] = old_frame

    print(f"Saved GIF: {filename}")

# ============================================================
# LOAD DATA
# ============================================================
data = np.loadtxt(FILENAME, skiprows=1)

t = data[:, 0].astype(float)
xyz = data[:, 1:4].astype(float)

t = t - t[0]
TMAX = float(t[-1])

if len(t) < 2:
    raise ValueError("Need at least 2 data points.")

xmin, ymin, zmin = xyz.min(axis=0)
xmax, ymax, zmax = xyz.max(axis=0)

dx = xmax - xmin
dy = ymax - ymin
dz = zmax - zmin

xmin -= PAD_FRAC * dx
xmax += PAD_FRAC * dx
ymin -= PAD_FRAC * dy
ymax += PAD_FRAC * dy
zmin -= PAD_FRAC * dz
zmax += PAD_FRAC * dz

center = np.array([
    0.5 * (xmin + xmax),
    0.5 * (ymin + ymax),
    0.5 * (zmin + zmax),
], dtype=float)

scene_scale = max(xmax - xmin, ymax - ymin, zmax - zmin)


# ============================================================
# WALLS
# ============================================================
floor = make_quad(
    [xmin, ymin, zmin],
    [xmax, ymin, zmin],
    [xmax, ymax, zmin],
    [xmin, ymax, zmin],
)
floor_grid = make_structured_grid_on_plane(
    origin=np.array([xmin, ymin, zmin], dtype=float),
    uvec=np.array([xmax - xmin, 0, 0], dtype=float),
    vvec=np.array([0, ymax - ymin, 0], dtype=float),
    nu=GRID_RESOLUTION,
    nv=GRID_RESOLUTION,
)

# x = xmin wall
xwall_min = make_quad(
    [xmin, ymin, zmin],
    [xmin, ymax, zmin],
    [xmin, ymax, zmax],
    [xmin, ymin, zmax],
)
xwall_min_grid = make_structured_grid_on_plane(
    origin=np.array([xmin, ymin, zmin], dtype=float),
    uvec=np.array([0, ymax - ymin, 0], dtype=float),
    vvec=np.array([0, 0, zmax - zmin], dtype=float),
    nu=GRID_RESOLUTION,
    nv=GRID_RESOLUTION,
)

# x = xmax wall
xwall_max = make_quad(
    [xmax, ymin, zmin],
    [xmax, ymax, zmin],
    [xmax, ymax, zmax],
    [xmax, ymin, zmax],
)
xwall_max_grid = make_structured_grid_on_plane(
    origin=np.array([xmax, ymin, zmin], dtype=float),
    uvec=np.array([0, ymax - ymin, 0], dtype=float),
    vvec=np.array([0, 0, zmax - zmin], dtype=float),
    nu=GRID_RESOLUTION,
    nv=GRID_RESOLUTION,
)

# y = ymin wall
ywall_min = make_quad(
    [xmin, ymin, zmin],
    [xmax, ymin, zmin],
    [xmax, ymin, zmax],
    [xmin, ymin, zmax],
)
ywall_min_grid = make_structured_grid_on_plane(
    origin=np.array([xmin, ymin, zmin], dtype=float),
    uvec=np.array([xmax - xmin, 0, 0], dtype=float),
    vvec=np.array([0, 0, zmax - zmin], dtype=float),
    nu=GRID_RESOLUTION,
    nv=GRID_RESOLUTION,
)

# y = ymax wall
ywall_max = make_quad(
    [xmin, ymax, zmin],
    [xmax, ymax, zmin],
    [xmax, ymax, zmax],
    [xmin, ymax, zmax],
)
ywall_max_grid = make_structured_grid_on_plane(
    origin=np.array([xmin, ymax, zmin], dtype=float),
    uvec=np.array([xmax - xmin, 0, 0], dtype=float),
    vvec=np.array([0, 0, zmax - zmin], dtype=float),
    nu=GRID_RESOLUTION,
    nv=GRID_RESOLUTION,
)


# ============================================================
# PLOTTER
# ============================================================
pv.set_plot_theme("document")
plot_window_size = GIF_WINDOW_SIZE if SAVE_GIF else WINDOW_SIZE
pl = pv.Plotter(window_size=plot_window_size)
pl.set_background(BACKGROUND_COLOR)

floor_actor = pl.add_mesh(
    floor,
    color=WALL_COLOR,
    opacity=WALL_OPACITY,
    show_edges=False,
)
floor_grid_actor = pl.add_mesh(
    floor_grid,
    color=GRID_COLOR,
    opacity=GRID_OPACITY,
    line_width=GRID_LINE_WIDTH,
)

xwall_min_actor = pl.add_mesh(
    xwall_min,
    color=WALL_COLOR,
    opacity=WALL_OPACITY,
    show_edges=False,
)
xwall_min_grid_actor = pl.add_mesh(
    xwall_min_grid,
    color=GRID_COLOR,
    opacity=GRID_OPACITY,
    line_width=GRID_LINE_WIDTH,
)

xwall_max_actor = pl.add_mesh(
    xwall_max,
    color=WALL_COLOR,
    opacity=WALL_OPACITY,
    show_edges=False,
)
xwall_max_grid_actor = pl.add_mesh(
    xwall_max_grid,
    color=GRID_COLOR,
    opacity=GRID_OPACITY,
    line_width=GRID_LINE_WIDTH,
)

ywall_min_actor = pl.add_mesh(
    ywall_min,
    color=WALL_COLOR,
    opacity=WALL_OPACITY,
    show_edges=False,
)
ywall_min_grid_actor = pl.add_mesh(
    ywall_min_grid,
    color=GRID_COLOR,
    opacity=GRID_OPACITY,
    line_width=GRID_LINE_WIDTH,
)

ywall_max_actor = pl.add_mesh(
    ywall_max,
    color=WALL_COLOR,
    opacity=WALL_OPACITY,
    show_edges=False,
)
ywall_max_grid_actor = pl.add_mesh(
    ywall_max_grid,
    color=GRID_COLOR,
    opacity=GRID_OPACITY,
    line_width=GRID_LINE_WIDTH,
)

shadow = make_polyline(xyz)
shadow_actor = pl.add_mesh(
    shadow,
    color=SHADOW_COLOR,
    opacity=SHADOW_OPACITY,
    line_width=SHADOW_LINE_WIDTH,
)

init_idx = 2
active = make_polyline(xyz[:init_idx].copy())
active["progress"] = t[:init_idx]

active_actor = pl.add_mesh(
    active,
    scalars="progress",
    cmap=ACTIVE_CMAP,
    clim=[t[0], t[-1]],
    line_width=ACTIVE_LINE_WIDTH,
    show_scalar_bar=False,
    render_lines_as_tubes=False,
)

p0 = interp_point(0.0, t, xyz)

point_outline = pv.PolyData(p0.copy())
point_main = pv.PolyData(p0.copy())

point_outline_actor = pl.add_mesh(
    point_outline,
    render_points_as_spheres=True,
    point_size=POINT_EDGE_SIZE,
    color=POINT_EDGE_COLOR,
    opacity=POINT_EDGE_OPACITY,
)

point_actor = pl.add_mesh(
    point_main,
    render_points_as_spheres=True,
    point_size=POINT_SIZE,
    color=POINT_COLOR,
)

pl.camera_position = [
    center + np.array([2.1, -2.3, 1.25]) * scene_scale,
    center,
    (0, 0, 1),
]
pl.camera.zoom(1.0)


# ============================================================
# STATE + UPDATE
# ============================================================
base_duration = TMAX
playback_duration = base_duration / PLAYBACK_SPEED
n_frames = max(2, int(np.round(playback_duration * FPS)))

state = {
    "frame": 0,
    "playing": True,
    "closed": False,
}

timer_id = None


def update_visible_walls():
    cam_pos = np.array(pl.camera.position, dtype=float)

    show_xmin = int(cam_pos[0] >= center[0])
    show_ymin = int(cam_pos[1] >= center[1])

    xwall_min_actor.SetVisibility(show_xmin)
    xwall_min_grid_actor.SetVisibility(show_xmin)

    xwall_max_actor.SetVisibility(1 - show_xmin)
    xwall_max_grid_actor.SetVisibility(1 - show_xmin)

    ywall_min_actor.SetVisibility(show_ymin)
    ywall_min_grid_actor.SetVisibility(show_ymin)

    ywall_max_actor.SetVisibility(1 - show_ymin)
    ywall_max_grid_actor.SetVisibility(1 - show_ymin)


def update_scene(
    frame_idx: int,
    rotate_camera: bool = False,
    do_render: bool = True,
    line_stride: int = 1,
):
    if state["closed"]:
        return

    frame_idx = max(0, min(frame_idx, n_frames - 1))

    progress = frame_idx / (n_frames - 1)
    sim_t = progress * TMAX

    idx = np.searchsorted(t, sim_t, side="right")
    idx = max(2, idx)

    line_xyz, line_t = get_line_data(idx, stride=line_stride)

    new_line = make_polyline(line_xyz)
    new_line["progress"] = line_t
    active_actor.mapper.dataset.copy_from(new_line)

    new_p = interp_point(sim_t, t, xyz)
    point_actor.mapper.dataset.points = new_p
    point_outline_actor.mapper.dataset.points = new_p

    if ROTATE_CAMERA and rotate_camera:
        pl.camera.Azimuth(CAMERA_AZIMUTH_SPEED / FPS)

    update_visible_walls()

    if do_render:
        pl.render()


# ============================================================
# CONTROLS
# ============================================================
def toggle_play():
    state["playing"] = not state["playing"]
    print("Play" if state["playing"] else "Pause")


def step_forward():
    jump = max(1, int(round(SEEK_SECONDS / TMAX * (n_frames - 1))))
    state["frame"] = min(state["frame"] + jump, n_frames - 1)
    update_scene(state["frame"], rotate_camera=False)


def step_backward():
    jump = max(1, int(round(SEEK_SECONDS / TMAX * (n_frames - 1))))
    state["frame"] = max(state["frame"] - jump, 0)
    update_scene(state["frame"], rotate_camera=False)


def reset_animation():
    state["frame"] = 0
    state["playing"] = True
    update_scene(state["frame"], rotate_camera=False)


def quit_app():
    global timer_id

    if state["closed"]:
        return

    state["closed"] = True
    state["playing"] = False

    try:
        iren = pl.iren.interactor
    except Exception:
        iren = None

    try:
        if iren is not None and timer_id is not None:
            iren.DestroyTimer(timer_id)
            timer_id = None
    except Exception:
        pass

    try:
        pl.close()
    except Exception:
        pass

    os._exit(0)


pl.add_key_event("space", toggle_play)
pl.add_key_event("Left", step_backward)
pl.add_key_event("Right", step_forward)
pl.add_key_event("r", reset_animation)
pl.add_key_event("q", quit_app)


# ============================================================
# TIMER CALLBACK
# ============================================================
def timer_callback(obj, event):
    if state["closed"]:
        return

    if not state["playing"]:
        return

    state["frame"] = (state["frame"] + 1) % n_frames
    update_scene(state["frame"], rotate_camera=True)


# initial draw
update_scene(0, rotate_camera=False)

print("Controls:")
print("  space = play/pause")
print("  r     = reset")
print("  left/right = step backward/forward")
print("  q     = quit")
print("  mouse = orbit/pan/zoom")

# Prepare interactor before showing
pl.iren.initialize()
pl.iren.interactor.AddObserver("TimerEvent", timer_callback)
timer_id = pl.iren.interactor.CreateRepeatingTimer(int(1000 / FPS))


def on_exit(obj, event):
    quit_app()


pl.iren.interactor.AddObserver("ExitEvent", on_exit)
if SAVE_GIF:
    export_gif(GIF_NAME, GIF_START_TIME, GIF_DURATION, GIF_FPS)
    raise SystemExit

pl.show(auto_close=False)

# once the window is closed, release references cleanly
timer_id = None
active_actor = None
point_actor = None
point_outline_actor = None
shadow = None
active = None
point_main = None
point_outline = None

floor = None
xwall_min = xwall_max = None
ywall_min = ywall_max = None

floor_grid = None
xwall_min_grid = xwall_max_grid = None
ywall_min_grid = ywall_max_grid = None

floor_actor = None
floor_grid_actor = None
xwall_min_actor = xwall_max_actor = None
ywall_min_actor = ywall_max_actor = None
xwall_min_grid_actor = xwall_max_grid_actor = None
ywall_min_grid_actor = ywall_max_grid_actor = None

pl = None
