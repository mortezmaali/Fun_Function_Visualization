import cv2 as cv
import numpy as np

# ==============================
# Config
# ==============================
FPS = 30
K_START, K_END = -5.0, 100.0
N_FRAMES = int((K_END - K_START) * 6)   # smooth sweep
OUT_FILE = "kx_sweep_x23_fullscreen_k.mp4"

# Plot margins (pixels)
L, R, T, B = 90, 40, 40, 90

# Math domain/range
xmax = np.sqrt(3.0)     # sqrt(3) from sqrt(3 - x^2)
xmin = -xmax
ymin, ymax = -2.0, 3.0  # vertical limits

def world_to_screen(x, y, W, H):
    plot_w, plot_h = W - L - R, H - T - B
    sx = L + (x - xmin) / (xmax - xmin) * plot_w
    sy = T + (ymax - y) / (ymax - ymin) * plot_h
    return np.column_stack([sx, sy]).astype(np.int32)

def draw_axes(img, W, H):
    cv.rectangle(img, (L, T), (W - R, H - B), (180,180,180), 1, cv.LINE_AA)
    # axes
    if xmin <= 0 <= xmax:
        p = world_to_screen(np.array([0.0]), np.array([0.0]), W, H)[0]
        cv.line(img, (p[0], T), (p[0], H - B), (120,120,120), 1, cv.LINE_AA)
    if ymin <= 0 <= ymax:
        p = world_to_screen(np.array([0.0]), np.array([0.0]), W, H)[0]
        cv.line(img, (L, p[1]), (W - R, p[1]), (120,120,120), 1, cv.LINE_AA)
    # ticks
    for xt in np.arange(np.ceil(xmin*2)/2, np.floor(xmax*2)/2 + 0.001, 0.5):
        p = world_to_screen(np.array([xt]), np.array([0]), W, H)[0]
        cv.line(img, (p[0], H - B), (p[0], H - B + 6), (150,150,150), 1, cv.LINE_AA)
        cv.putText(img, f"{xt:g}", (p[0]-10, H - B + 22), cv.FONT_HERSHEY_SIMPLEX, 0.45, (90,90,90), 1, cv.LINE_AA)
    for yt in np.arange(np.ceil(ymin*2)/2, np.floor(ymax*2)/2 + 0.001, 0.5):
        p = world_to_screen(np.array([0]), np.array([yt]), W, H)[0]
        cv.line(img, (L-6, p[1]), (L, p[1]), (150,150,150), 1, cv.LINE_AA)
        cv.putText(img, f"{yt:g}", (L-46, p[1]+4), cv.FONT_HERSHEY_SIMPLEX, 0.45, (90,90,90), 1, cv.LINE_AA)

def overlay_k_panel(img, k, W, H):
    """Draw a semi-transparent panel with the current k value + progress bar."""
    # Panel rect (top-right)
    pad, w, h = 14, 280, 86
    x1, y1 = W - R - w, T + pad
    x2, y2 = W - R - pad, T + pad + h

    panel = img.copy()
    cv.rectangle(panel, (x1, y1), (x2, y2), (255,255,255), -1)
    img[:] = cv.addWeighted(panel, 0.75, img, 0.25, 0)  # 75% white, see-through

    # Text with subtle shadow
    txt = f"k = {k:0.2f}"
    cv.putText(img, txt, (x1 + 16, y1 + 32), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv.LINE_AA)
    cv.putText(img, txt, (x1 + 16, y1 + 32), cv.FONT_HERSHEY_SIMPLEX, 0.9, (20,20,20), 1, cv.LINE_AA)

    # Progress bar
    bar_y1, bar_y2 = y1 + 54, y1 + 70
    cv.rectangle(img, (x1 + 16, bar_y1), (x2 - 16, bar_y2), (180,180,180), 1, cv.LINE_AA)
    t = (k - K_START) / (K_END - K_START)
    t = float(np.clip(t, 0, 1))
    fill_x = int((x2 - 16 - (x1 + 16)) * t)
    cv.rectangle(img, (x1 + 16, bar_y1), (x1 + 16 + fill_x, bar_y2), (30,130,230), -1, cv.LINE_AA)

def make_frame(k, W, H):
    img = np.full((H, W, 3), 255, np.uint8)
    draw_axes(img, W, H)

    # Curve
    x = np.linspace(xmin, xmax, 1600)
    base = np.power(np.abs(x), 2/3)
    rad = 3.0 - x**2
    f = base + 0.9 * np.sin(k * x) * np.sqrt(np.clip(rad, 0, None))

    pts = world_to_screen(x, f, W, H)
    cv.polylines(img, [pts], False, (30,130,230), 2, cv.LINE_AA)

    # Labels
    eq = "f(x) = |x|^(2/3) + 0.9sin(kx)sqrt(3 - x^2)"
    cv.putText(img, eq, (L, T - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv.LINE_AA)

    overlay_k_panel(img, k, W, H)
    return img

def main():
    # Try to detect monitor resolution for true fullscreen
    W, H = 1920, 1080
    try:
        from screeninfo import get_monitors
        m = get_monitors()[0]
        W, H = m.width, m.height
    except Exception:
        pass

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(OUT_FILE, fourcc, FPS, (W, H))

    win = "f(x) animation"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    cv.setWindowProperty(win, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.resizeWindow(win, W, H)

    ks = np.linspace(K_START, K_END, N_FRAMES)
    for k in ks:
        frame = make_frame(k, W, H)
        writer.write(frame)
        cv.imshow(win, frame)
        if (cv.waitKey(int(1000/FPS)) & 0xFF) in (27, ord('q')):
            break

    writer.release()
    cv.destroyAllWindows()
    print(f"Saved: {OUT_FILE}")

if __name__ == "__main__":
    main()
