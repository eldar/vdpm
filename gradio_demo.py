# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import glob
import os
import shutil
import sys
import time
from datetime import datetime

import cv2
import gradio as gr
import matplotlib
import numpy as np
import plotly.graph_objects as go
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from dpm.model import VDPM
from vggt.utils.load_fn import load_and_preprocess_images


MAX_POINTS_PER_FRAME = 50_000
TRAIL_LENGTH = 20
MAX_TRACKS = 150
STATIC_THRESHOLD = 0.025
VIDEO_SAMPLE_HZ = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_cfg_from_cli() -> "omegaconf.DictConfig":
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    overrides = sys.argv[1:]
    with initialize(config_path="configs"):
        return compose(config_name="visualise", overrides=overrides)


def load_model(cfg) -> VDPM:
    model = VDPM(cfg).to(device)

    _URL = "https://huggingface.co/edgarsucar/vdpm/resolve/main/model.pt"
    sd = torch.hub.load_state_dict_from_url(
        _URL,
        file_name="vdpm_model.pt",
        progress=True
    )
    print(model.load_state_dict(sd, strict=True))

    model.eval()
    return model


def require_cuda():
    if device != "cuda":
        raise ValueError("CUDA is not available. Check your environment.")


def gradio_file_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, dict) and "name" in file_obj:
        return file_obj["name"]
    return file_obj


def ensure_nhwc_images(images: np.ndarray) -> np.ndarray:
    if images.ndim == 4 and images.shape[1] == 3:
        return np.transpose(images, (0, 2, 3, 1))
    return images


def compute_scene_bounds(world_points: np.ndarray):
    all_pts = world_points.reshape(-1, 3)
    raw_min = all_pts.min(axis=0)
    raw_max = all_pts.max(axis=0)

    center = 0.5 * (raw_min + raw_max)
    half_extent = 0.5 * (raw_max - raw_min) * 1.05

    if np.all(half_extent < 1e-6):
        half_extent[:] = 1.0
    else:
        half_extent[half_extent < 1e-6] = half_extent.max()

    global_min = center - half_extent
    global_max = center + half_extent

    max_half = half_extent.max()
    aspectratio = {
        "x": float(half_extent[0] / max_half),
        "y": float(half_extent[1] / max_half),
        "z": float(half_extent[2] / max_half),
    }
    return global_min, global_max, aspectratio


def stride_downsample(pts: np.ndarray, cols: np.ndarray, max_points: int):
    n = pts.shape[0]
    if n <= max_points:
        return pts, cols
    step = int(np.ceil(n / max_points))
    idx = np.arange(0, n, step)[:max_points]
    return pts[idx], cols[idx]


# ============================================================
# NEW: Single shared mask function (used by points + tracks)
# ============================================================
def compute_point_mask(
    conf_score: np.ndarray | None,
    cols: np.ndarray,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
) -> np.ndarray:
    """
    conf_score: (N,) or None
    cols: (N,3) uint8
    Returns: (N,) boolean mask
    """
    mask = np.ones(cols.shape[0], dtype=bool)

    # confidence percentile threshold (same semantics as before)
    if conf_score is not None and conf_thres > 0:
        thresh = np.percentile(conf_score, conf_thres)
        mask &= (conf_score >= thresh) & (conf_score > 1e-5)

    # background masks (same as before)
    if mask_black_bg:
        mask &= (cols.sum(axis=1) >= 16)
    if mask_white_bg:
        mask &= ~((cols[:, 0] > 240) & (cols[:, 1] > 240) & (cols[:, 2] > 240))

    return mask


def sample_frame_points(
    world_points: np.ndarray,
    images_nhwc: np.ndarray,
    conf: np.ndarray | None,
    idx: int,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    max_points: int,
):
    i = int(np.clip(idx, 0, world_points.shape[0] - 1))
    pts = world_points[i].reshape(-1, 3)
    cols = (images_nhwc[i].reshape(-1, 3) * 255).astype(np.uint8)

    conf_score = conf[i].reshape(-1) if (conf is not None) else None

    mask = compute_point_mask(
        conf_score=conf_score,
        cols=cols,
        conf_thres=conf_thres,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
    )

    pts = pts[mask]
    cols = cols[mask]

    pts, cols = stride_downsample(pts, cols, max_points)

    if pts.size == 0:
        pts = np.array([[0.0, 0.0, 0.0]])
        cols = np.array([[255, 255, 255]], dtype=np.uint8)

    colors_str = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in cols]
    return pts, colors_str


# ============================================================
# UPDATED: prepare_tracks now applies the SAME masks as points
# ============================================================
def prepare_tracks(
    world_points: np.ndarray,
    images_nhwc: np.ndarray,
    conf: np.ndarray | None,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
):
    S, H, W, _ = world_points.shape
    N = H * W
    if S < 2 or N == 0:
        return None, None, None

    tracks_xyz = world_points.reshape(S, N, 3)

    disp = np.linalg.norm(tracks_xyz - tracks_xyz[0:1], axis=-1)
    dynamic_mask = disp.max(axis=0) > STATIC_THRESHOLD

    # build a per-point confidence score (across time)
    conf_score = None
    if conf is not None:
        conf_flat = conf.reshape(S, N)
        conf_score = conf_flat.mean(axis=0)

    # Use reference-frame colors for background masking (stable, consistent)
    ref_cols = (images_nhwc[0].reshape(-1, 3) * 255).astype(np.uint8)

    point_mask = compute_point_mask(
        conf_score=conf_score,
        cols=ref_cols,
        conf_thres=conf_thres,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
    )

    dynamic_mask &= point_mask

    idx_tracks = np.nonzero(dynamic_mask)[0]
    if idx_tracks.size == 0:
        return None, None, None

    if idx_tracks.size > MAX_TRACKS:
        step = int(np.ceil(idx_tracks.size / MAX_TRACKS))
        idx_tracks = idx_tracks[::step][:MAX_TRACKS]

    tracks_xyz = tracks_xyz[:, idx_tracks, :]
    order = np.argsort(tracks_xyz[0, :, 1])
    tracks_xyz = tracks_xyz[:, order, :]

    num_tracks = tracks_xyz.shape[1]
    cmap = matplotlib.colormaps.get_cmap("hsv")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(num_tracks - 1, 1))

    colorscale = []
    for t in range(num_tracks):
        r, g, b, _ = cmap(norm(t))
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        pos = t / max(num_tracks - 1, 1)
        colorscale.append([pos, f"rgb({r},{g},{b})"])

    track_ids = np.arange(num_tracks, dtype=float)
    return tracks_xyz, colorscale, track_ids


def track_segments_for_frame(tracks_xyz: np.ndarray | None, track_ids: np.ndarray | None, f: int):
    if tracks_xyz is None or track_ids is None or f <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    start_t = max(0, f - TRAIL_LENGTH)
    num_tracks = tracks_xyz.shape[1]

    xs, ys, zs, cs = [], [], [], []
    for j in range(num_tracks):
        seg = tracks_xyz[start_t : f + 1, j, :]
        if seg.shape[0] < 2:
            continue

        xs.extend([seg[:, 0], np.array([np.nan])])
        ys.extend([seg[:, 1], np.array([np.nan])])
        zs.extend([seg[:, 2], np.array([np.nan])])
        cs.append(np.full(seg.shape[0] + 1, track_ids[j], dtype=float))

    x = np.concatenate(xs) if xs else np.array([])
    y = np.concatenate(ys) if ys else np.array([])
    z = np.concatenate(zs) if zs else np.array([])
    c = np.concatenate(cs) if cs else np.array([])

    return x, y, z, c


def build_pointcloud_figure_update(
    data,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
):
    if data is None:
        return go.Figure()

    world_points = data["world_points"]
    conf = data.get("world_points_conf")
    images = ensure_nhwc_images(data["images"])
    S = world_points.shape[0]

    global_min, global_max, aspectratio = compute_scene_bounds(world_points)

    # UPDATED: pass same masks into prepare_tracks
    tracks_xyz, colorscale, track_ids = prepare_tracks(
        world_points=world_points,
        images_nhwc=images,
        conf=conf,
        conf_thres=conf_thres,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
    )
    track_cmax = max(len(track_ids) - 1, 1) if track_ids is not None else 1

    pts_xyz = [None] * S
    pts_cols = [None] * S
    trk_xyz = [None] * S
    trk_c = [None] * S

    for i in range(S):
        pts_i, cols_i = sample_frame_points(
            world_points=world_points,
            images_nhwc=images,
            conf=conf,
            idx=i,
            conf_thres=conf_thres,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            max_points=MAX_POINTS_PER_FRAME,
        )
        pts_xyz[i] = pts_i
        pts_cols[i] = cols_i

        x, y, z, c = track_segments_for_frame(tracks_xyz, track_ids, f=i)
        trk_xyz[i] = (x, y, z)
        trk_c[i] = c

    p0 = pts_xyz[0]
    c0 = pts_cols[0]
    x0, y0, z0 = trk_xyz[0]
    tc0 = trk_c[0]

    scene_cfg = dict(
        xaxis=dict(
            visible=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[float(global_min[0]), float(global_max[0])],
        ),
        yaxis=dict(
            visible=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[float(global_min[1]), float(global_max[1])],
        ),
        zaxis=dict(
            visible=False,
            showbackground=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[float(global_min[2]), float(global_max[2])],
        ),
        aspectmode="manual",
        aspectratio=aspectratio,
        dragmode="orbit",
        camera=dict(
            eye=dict(x=0.0, y=0.0, z=-1.0),
            center=dict(x=0.0, y=0.0, z=0.0),
            up=dict(x=0.0, y=-1.0, z=0.0),
        ),
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=p0[:, 0],
                y=p0[:, 1],
                z=p0[:, 2],
                mode="markers",
                marker=dict(size=2, color=c0),
                showlegend=False,
                name="points",
            ),
            go.Scatter3d(
                x=x0,
                y=y0,
                z=z0,
                mode="lines",
                line=dict(
                    width=2,
                    color=tc0 if (tc0 is not None and tc0.size) else None,
                    colorscale=colorscale if colorscale is not None else None,
                    cmin=0,
                    cmax=track_cmax,
                ),
                hoverinfo="skip",
                showlegend=False,
                name="tracks",
            ),
        ]
    )

    steps = []
    for i in range(S):
        pi = pts_xyz[i]
        ci = pts_cols[i]
        xi, yi, zi = trk_xyz[i]
        ti = trk_c[i]

        steps.append(
            dict(
                method="update",
                label=str(i),
                args=[
                    {
                        "x": [pi[:, 0], xi],
                        "y": [pi[:, 1], yi],
                        "z": [pi[:, 2], zi],
                        "marker.color": [ci, None],
                        "line.color": [None, ti if (ti is not None and len(ti)) else None],
                    },
                    {},
                ],
            )
        )

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Frame: ", "visible": True, "font": {"size": 14}},
            pad={"t": 10},
            len=0.6,
            x=0.2,
            font={"size": 8},
            steps=steps,
        )
    ]

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=scene_cfg,
        sliders=sliders,
        showlegend=False,
        title="Scrub frames with the slider below",
        uirevision="keep-camera",
        height=700,
    )
    return fig


def run_model(target_dir: str, model: VDPM, frame_id_arg=0) -> dict:
    require_cuda()

    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    if not image_names:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        predictions = model.inference(None, images=images.unsqueeze(0))

    pts_list = [pm["pts3d"].detach().cpu().numpy() for pm in predictions["pointmaps"]]
    conf_list = [pm["conf"].detach().cpu().numpy() for pm in predictions["pointmaps"]]

    world_points = np.concatenate(pts_list, axis=0)
    world_points_conf = np.concatenate(conf_list, axis=0)

    try:
        frame_id = int(frame_id_arg)
    except Exception:
        frame_id = 0

    if frame_id >= world_points.shape[0]:
        frame_id = 0

    world_points_s = world_points[:, frame_id, ::2, ::2, :]
    single_mask = world_points_conf[frame_id, frame_id, ::2, ::2]
    world_points_conf_s = np.tile(single_mask[np.newaxis, ...], (world_points.shape[0], 1, 1))

    img_np = images.detach().cpu().numpy()
    img_np = img_np[frame_id : frame_id + 1, :, ::2, ::2]
    img_np = np.repeat(img_np, world_points.shape[0], axis=0)

    torch.cuda.empty_cache()
    return {
        "world_points": world_points_s,
        "world_points_conf": world_points_conf_s,
        "images": img_np,
    }


def handle_uploads(input_video, input_images):
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir_images, exist_ok=True)

    image_paths = []

    if input_images:
        for file_obj in input_images:
            src = gradio_file_path(file_obj)
            if not src:
                continue
            dst = os.path.join(target_dir_images, os.path.basename(src))
            shutil.copy(src, dst)
            image_paths.append(dst)

    if input_video:
        video_path = gradio_file_path(input_video)
        vs = cv2.VideoCapture(video_path)

        fps = float(vs.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_interval = max(int(fps / max(VIDEO_SAMPLE_HZ, 1e-6)), 1)

        count = 0
        frame_num = 0
        try:
            while True:
                gotit, frame = vs.read()
                if not gotit:
                    break
                if count % frame_interval == 0:
                    out_path = os.path.join(target_dir_images, f"{frame_num:06}.png")
                    cv2.imwrite(out_path, frame)
                    image_paths.append(out_path)
                    frame_num += 1
                count += 1
        finally:
            vs.release()

    image_paths.sort()
    print(f"Files copied to {target_dir_images}; took {time.time() - start_time:.3f} seconds")
    return target_dir, image_paths


def update_gallery_on_upload(input_video, input_images):
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


def gradio_reconstruct(
    target_dir,
    conf_thres=50.0,
    mask_black_bg=False,
    mask_white_bg=False,
    frame_id_val=0,
):
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None

    gc.collect()
    torch.cuda.empty_cache()

    target_dir_images = os.path.join(target_dir, "images")
    num_frames = len(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else 0

    with torch.no_grad():
        predictions = run_model(target_dir, model, frame_id_val)

    fig = build_pointcloud_figure_update(predictions, conf_thres, mask_black_bg, mask_white_bg)

    torch.cuda.empty_cache()
    msg = f"Reconstruction Success ({num_frames} frames processed, showing frame {frame_id_val})."
    return fig, msg, predictions


def update_plot(
    target_dir,
    predictions,
    conf_thres,
    mask_black_bg,
    mask_white_bg,
    is_example,
):
    if is_example == "True" or predictions is None:
        return None, "No reconstruction available. Please click the Reconstruct button first."

    fig = build_pointcloud_figure_update(predictions, conf_thres, mask_black_bg, mask_white_bg)
    return fig, "Updated visualization with new settings. Use the slider below the plot to scrub frames."


def clear_fields():
    return None


def update_log():
    return "Loading and Reconstructing..."


def example_pipeline(
    input_video_ex,
    num_images_str,
    input_images_ex,
    conf_thres_val,
    mask_black_bg_val,
    mask_white_bg_val,
    is_example_str,
    frame_id_val,
):
    target_dir, image_paths = handle_uploads(input_video_ex, input_images_ex)
    fig, log_msg, predictions = gradio_reconstruct(
        target_dir,
        conf_thres_val,
        mask_black_bg_val,
        mask_white_bg_val,
        frame_id_val,
    )
    return fig, log_msg, target_dir, predictions, image_paths


colosseum_video = "examples/videos/Colosseum.mp4"
camel_video = "examples/videos/camel.mp4"
tennis_video = "examples/videos/tennis.mp4"
paragliding_video = "examples/videos/paragliding.mp4"
stroller_video = "examples/videos/stroller.mp4"
goldfish_video = "examples/videos/goldfish.mp4"
horse_video = "examples/videos/horse.mp4"
swing_video = "examples/videos/swing.mp4"
car_video = "examples/videos/car.mp4"
figure1_video = "examples/videos/figure1.mp4"
figure2_video = "examples/videos/figure2.mp4"
figure3_video = "examples/videos/figure3.mp4"
tesla_video = "examples/videos/tesla.mp4"
pstudio_video = "examples/videos/pstudio.mp4"

theme = gr.themes.Default(
    primary_hue=gr.themes.colors.slate,
    secondary_hue=gr.themes.colors.zinc,
    neutral_hue=gr.themes.colors.slate,
).set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
    body_background_fill="#FFFFFF",
)

css = """
.custom-log * {
    font-style: italic;
    font-size: 22px !important;
    background-image: linear-gradient(120deg, #1f2937 0%, #4b5563 100%);
    -webkit-background-clip: text;
    background-clip: text;
    font-weight: bold !important;
    color: transparent !important;
    text-align: center !important;
}

.example-log * {
    font-style: italic;
    font-size: 16px !important;
    background-image: linear-gradient(120deg, #1f2937 0%, #4b5563 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}

#my_radio .wrap {
    display: flex;
    flex-wrap: nowrap;
    justify-content: center;
    align-items: center;
}

#my_radio .wrap label {
    display: flex;
    width: 50%;
    justify-content: center;
    align-items: center;
    margin: 0;
    padding: 10px 0;
    box-sizing: border-box;
}
"""

cfg = load_cfg_from_cli()
model = load_model(cfg)

with gr.Blocks(theme=theme, css=css) as demo:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")
    frame_id_state = gr.Textbox(label="frame_id", visible=False, value="0")

    gr.HTML(
        """
        <h1>V-DPM: Video Reconstruction with Dynamic Point Maps</h1>
        <p>
        <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
        <a href="#">Project Page</a>
        </p>
        <div style="font-size: 16px; line-height: 1.5;">
        <p>Upload a video or a set of images to create a dynamic point map reconstruction of a scene or object.</p>
        </div>
        """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")
    predictions_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)
            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=5):
            gr.Markdown("**3D Reconstruction (Point Cloud)**")
            log_output = gr.Markdown(
                "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
            )
            reconstruction_output = gr.Plot(label="3D Point Cloud")

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )

            with gr.Row():
                conf_thres = gr.Slider(0, 100, value=50, step=1, label="Confidence Threshold (%)")
                with gr.Column():
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    examples = [
        [camel_video, "17", None, 15.0, False, False, "True", "8"],
        [horse_video, "18", None, 50.0, False, False, "True", "2"],
        [tennis_video, "11", None, 5.0, False, False, "True", "0"],
        [paragliding_video, "11", None, 5.0, False, False, "True", "0"],
        [stroller_video, "17", None, 10.0, False, False, "True", "8"],
        [goldfish_video, "11", None, 12.0, False, False, "True", "5"],
        [swing_video, "10", None, 40.0, False, False, "True", "4"],
        [car_video, "13", None, 15.0, False, False, "True", "7"],
        [figure1_video, "10", None, 25.0, False, False, "True", "0"],
        [figure2_video, "12", None, 25.0, False, False, "True", "6"],
        [figure3_video, "13", None, 30.0, False, False, "True", "0"],
        [tesla_video, "18", None, 20.0, False, True, "True", "0"],
        [pstudio_video, "12", None, 0.0, False, False, "True", "6"],
    ]

    gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])
    gr.Examples(
        examples=examples,
        inputs=[
            input_video,
            num_images,
            input_images,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            is_example,
            frame_id_state,
        ],
        outputs=[
            reconstruction_output,
            log_output,
            target_dir_output,
            predictions_state,
            image_gallery,
        ],
        fn=example_pipeline,
        cache_examples=False,
        examples_per_page=50,
    )

    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_reconstruct,
        inputs=[
            target_dir_output,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            frame_id_state,
        ],
        outputs=[reconstruction_output, log_output, predictions_state],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]
    )

    for ctrl in (conf_thres, mask_black_bg, mask_white_bg):
        ctrl.change(
            fn=update_plot,
            inputs=[
                target_dir_output,
                predictions_state,
                conf_thres,
                mask_black_bg,
                mask_white_bg,
                is_example,
            ],
            outputs=[reconstruction_output, log_output],
        )

    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)
