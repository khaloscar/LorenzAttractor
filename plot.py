import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ============================================================
# Settings
# ============================================================
FILENAME = "lorenz.txt"

FRAME_STRIDE = 5
ROTATE_VIEW = True

# Main interactive playback
PLAYBACK_FPS = 15

# Saved GIF settings
SAVE_GIF = False
GIF_NAME = "lorenz_clip.gif"

GIF_START_TIME = 25   # approximate start time in simulation seconds
GIF_DURATION = 10.0     # length of clip in simulation seconds
GIF_FPS = 12            # lower fps keeps file size smaller
GIF_CENTER = None           # None -> middle of sim, or set e.g. 22.0

# Good choices: "plasma", "inferno", "viridis", "coolwarm"
CMAP_NAME = "inferno"

# ============================================================
# Load data
# ============================================================
data = np.loadtxt(FILENAME, skiprows=1)
t = data[:, 0]
x = data[:, 1]
y = data[:, 2]
z = data[:, 3]

idx = np.arange(0, len(t), FRAME_STRIDE)
t = t[idx]
x = x[idx]
y = y[idx]
z = z[idx]

n = len(t)
TMAX = t[-1]

# Build 3D segments
points = np.array([x, y, z]).T.reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Explicit colors for each segment
cmap = plt.get_cmap(CMAP_NAME)
segment_colors = cmap(np.linspace(0.0, 1.0, len(segments)))

# ============================================================
# Figure
# ============================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.22)

ax.set_title("Lorenz System")

# Keep limits
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_zlim(np.min(z), np.max(z))

# Keep ticks so the grid/box remains
ax.set_xticks(np.linspace(np.min(x), np.max(x), 6))
ax.set_yticks(np.linspace(np.min(y), np.max(y), 6))
ax.set_zticks(np.linspace(np.min(z), np.max(z), 6))

# Remove axis labels
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")

# Hide tick labels but keep gridlines
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# Hide tick marks
ax.tick_params(axis='x', which='both', length=0, pad=-2)
ax.tick_params(axis='y', which='both', length=0, pad=-2)
ax.tick_params(axis='z', which='both', length=0, pad=-2)

# Keep grid and panes
ax.grid(True)
ax.xaxis.pane.set_alpha(0.08)
ax.yaxis.pane.set_alpha(0.08)
ax.zaxis.pane.set_alpha(0.08)

# Optional lighter grid
ax.xaxis._axinfo["grid"]['linewidth'] = 0.8
ax.yaxis._axinfo["grid"]['linewidth'] = 0.8
ax.zaxis._axinfo["grid"]['linewidth'] = 0.8
ax.xaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.6)
ax.yaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.6)
ax.zaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.6)

# Background full trajectory as shadow
ax.plot(x, y, z, color="gray", alpha=0.18, lw=1.2)

# Colored partial trajectory
lc = Line3DCollection([], linewidth=2.5)
ax.add_collection(lc)

# Current point
point, = ax.plot([], [], [], marker="o", markersize=6, color="red")

# Time label
time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)

# ============================================================
# Slider
# ============================================================
ax_slider = plt.axes([0.15, 0.10, 0.65, 0.03])
slider = Slider(ax_slider, "time", 0.0, TMAX, valinit=0.0)

# ============================================================
# Buttons
# ============================================================
ax_play = plt.axes([0.15, 0.03, 0.12, 0.05])
ax_pause = plt.axes([0.30, 0.03, 0.12, 0.05])
ax_reset = plt.axes([0.45, 0.03, 0.12, 0.05])

btn_play = Button(ax_play, "Play")
btn_pause = Button(ax_pause, "Pause")
btn_reset = Button(ax_reset, "Reset")

is_playing = False
current_time = 0.0

# ============================================================
# Helpers
# ============================================================
def time_to_index(t_now):
    i = np.searchsorted(t, t_now, side="right") - 1
    return max(0, min(i, n - 1))

def draw_time(t_now):
    global current_time
    current_time = float(max(0.0, min(t_now, TMAX)))

    i = time_to_index(current_time)

    if i >= 1:
        seg_now = segments[:i]
        col_now = segment_colors[:i]
        lc.set_segments(seg_now)
        lc.set_color(col_now)
    else:
        lc.set_segments([])
        lc.set_color([])

    point.set_data([x[i]], [y[i]])
    point.set_3d_properties([z[i]])

    time_text.set_text(f"t = {t[i]:.3f}")

    if ROTATE_VIEW:
        azim = 45 + 360.0 * (current_time / TMAX)
        ax.view_init(elev=25, azim=azim)

    fig.canvas.draw_idle()
    return i

# ============================================================
# Slider callback
# ============================================================
def slider_update(val):
    draw_time(slider.val)

slider.on_changed(slider_update)

# ============================================================
# Animation callback
# ============================================================
def update(_):
    global current_time
    if is_playing:
        current_time += 1.0 / PLAYBACK_FPS
        if current_time >= TMAX:
            current_time = TMAX
        draw_time(current_time)
    return lc, point, time_text

ani = FuncAnimation(
    fig,
    update,
    frames=int(np.ceil(TMAX * PLAYBACK_FPS)) + 1,
    interval=1000 / PLAYBACK_FPS,
    blit=False
)

# ============================================================
# Buttons
# ============================================================
def play(_):
    global is_playing
    is_playing = True

def pause(_):
    global is_playing
    is_playing = False

def reset(_):
    global is_playing, current_time
    is_playing = False
    current_time = 0.0
    slider.set_val(0.0)

btn_play.on_clicked(play)
btn_pause.on_clicked(pause)
btn_reset.on_clicked(reset)

draw_time(0.0)

# ============================================================
# Optional GIF saving
# ============================================================
if SAVE_GIF:
    gif_t0 = max(0.0, GIF_START_TIME)
    gif_t1 = min(TMAX, gif_t0 + GIF_DURATION)

    gif_nframes = int(np.ceil((gif_t1 - gif_t0) * GIF_FPS)) + 1

    # ------------------------------------------------------------
    # Separate clean export figure (no buttons, no slider)
    # ------------------------------------------------------------
    fig_gif = plt.figure(figsize=(8, 8))
    ax_gif = fig_gif.add_subplot(111, projection="3d")

    ax_gif.set_title("")

    ax_gif.set_xlim(np.min(x), np.max(x))
    ax_gif.set_ylim(np.min(y), np.max(y))
    ax_gif.set_zlim(np.min(z), np.max(z))

    # Keep grid box, hide numbers/labels
    ax_gif.set_xticks(np.linspace(np.min(x), np.max(x), 6))
    ax_gif.set_yticks(np.linspace(np.min(y), np.max(y), 6))
    ax_gif.set_zticks(np.linspace(np.min(z), np.max(z), 6))

    ax_gif.set_xlabel("")
    ax_gif.set_ylabel("")
    ax_gif.set_zlabel("")

    ax_gif.set_xticklabels([])
    ax_gif.set_yticklabels([])
    ax_gif.set_zticklabels([])

    ax_gif.tick_params(axis='x', which='both', length=0, pad=-2)
    ax_gif.tick_params(axis='y', which='both', length=0, pad=-2)
    ax_gif.tick_params(axis='z', which='both', length=0, pad=-2)

    ax_gif.grid(True)
    ax_gif.xaxis.pane.set_alpha(0.08)
    ax_gif.yaxis.pane.set_alpha(0.08)
    ax_gif.zaxis.pane.set_alpha(0.08)

    ax_gif.xaxis._axinfo["grid"]['linewidth'] = 0.8
    ax_gif.yaxis._axinfo["grid"]['linewidth'] = 0.8
    ax_gif.zaxis._axinfo["grid"]['linewidth'] = 0.8
    ax_gif.xaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.6)
    ax_gif.yaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.6)
    ax_gif.zaxis._axinfo["grid"]['color'] = (0.7, 0.7, 0.7, 0.6)

    # Background shadow
    ax_gif.plot(x, y, z, color="gray", alpha=0.18, lw=1.2)

    # Animated colored trajectory
    lc_gif = Line3DCollection([], linewidth=2.5)
    ax_gif.add_collection(lc_gif)

    point_gif, = ax_gif.plot([], [], [], marker="o", markersize=6, color="red")
    time_text_gif = ax_gif.text2D(0.02, 0.95, "", transform=ax_gif.transAxes)

    def gif_update(frame):
        frac = frame / max(gif_nframes - 1, 1)
        t_now = gif_t0 + frac * (gif_t1 - gif_t0)

        i = np.searchsorted(t, t_now, side="right") - 1
        i = max(0, min(i, n - 1))

        if i >= 1:
            lc_gif.set_segments(segments[:i])
            lc_gif.set_color(segment_colors[:i])
        else:
            lc_gif.set_segments([])
            lc_gif.set_color([])

        point_gif.set_data([x[i]], [y[i]])
        point_gif.set_3d_properties([z[i]])
        time_text_gif.set_text(f"t = {t[i]:.3f}")

        if ROTATE_VIEW:
            azim = 45 + 360.0 * (t_now / TMAX)
            ax_gif.view_init(elev=25, azim=azim)

        return lc_gif, point_gif, time_text_gif

    ani_save = FuncAnimation(
        fig_gif,
        gif_update,
        frames=gif_nframes,
        interval=1000 / GIF_FPS,
        blit=False
    )

    ani_save.save(GIF_NAME, writer=PillowWriter(fps=GIF_FPS))
    print(f"Saved GIF to {GIF_NAME}")

    plt.close(fig_gif)

plt.show()
