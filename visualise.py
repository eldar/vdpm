import time
from typing import List

import cv2
import numpy as np
import viser
import viser.transforms as vt
import hydra
import torch
from torch import Tensor
from jaxtyping import Float
from PIL import Image
from torchvision import transforms as TF
from einops import repeat
import matplotlib

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from dpm.model import VDPM
from util.transforms import transform_points


VIDEO_SAMPLE_HZ = 1.0


def assign_colours(pts3d, colour=[0, 0, 1]):
    num_points = pts3d.shape[0]
    colors = (
        np.tile(np.array([colour]), (num_points, 1)) * 255
    ).astype(np.uint8)
    return colors


def compute_box_edges(corners):
    """
    Compute all edges of a 3D bounding box

    Args:
        corners: torch tensor of shape (8, 3) containing the coordinates of the 8 corners
                 of a 3D bounding box

    Returns:
        edges: torch tensor of shape (12, 2, 3) containing the 12 edges of the box,
               each represented as a pair of 3D coordinates [start_point, end_point]
    """
    # Define the 12 edges of a cube by specifying pairs of corner indices
    edge_indices = torch.tensor([
        # Edges along x-axis
        [0, 1], [2, 3], [4, 5], [6, 7],
        # Edges along y-axis
        [0, 2], [1, 3], [4, 6], [5, 7],
        # Edges along z-axis
        [0, 4], [1, 5], [2, 6], [3, 7]
    ], dtype=torch.long)

    # Initialize edges tensor
    edges = torch.zeros((12, 2, 3), dtype=corners.dtype, device=corners.device)

    # Extract the start and end points for each edge
    for i, (start_idx, end_idx) in enumerate(edge_indices):
        edges[i, 0] = corners[start_idx]  # Start point
        edges[i, 1] = corners[end_idx]    # End point

    colors = torch.tensor([
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [0, 255, 255],    # Cyan
        [255, 0, 255],    # Magenta
        [255, 128, 0],    # Orange
        [128, 0, 255],    # Purple
        [128, 255, 0],    # Lime
        [255, 0, 128],    # Pink
        [0, 128, 255],    # Teal
        [128, 0, 0]       # Maroon
    ], dtype=torch.uint8, device=corners.device)

    return edges, colors


class TrackVisualiser:
    def __init__(self,
         server: viser.ViserServer
    ):
        self._trail_length = 12
        self._server = server

    def remove_static_tracks(self,
        tracks: Float[Tensor, "t n 3"],
        threshold=0.025
    ) -> Float[Tensor, "t n 3"]:
        # delta = tracks[1:] - tracks[[0]]
        delta = tracks[None, ...] - tracks[:, None, ...]
        max_displ = torch.linalg.norm(delta.abs(), dim=-1).amax((0, 1))
        dynamic = max_displ > threshold
        tracks_filtered = tracks[:, dynamic, :]
        return tracks_filtered

    def set_data(self,
        tracks_xyz: Float[Tensor, "t n 3"],
    ):
        # TODO: filter tracks and assign colours
        tracks_xyz = tracks_xyz.numpy()
        print("num actual tracks", tracks_xyz.shape[0])
        num_tracks = min(1000, tracks_xyz.shape[1])
        indices = np.random.choice(tracks_xyz.shape[1], num_tracks, replace=False)
        tracks_xyz = tracks_xyz[:, indices]
        sorted_indices = np.argsort(tracks_xyz[0, ..., 1])
        tracks_xyz = tracks_xyz[:, sorted_indices]
        color_map = matplotlib.colormaps.get_cmap('hsv')
        cmap_norm = matplotlib.colors.Normalize(vmin=0, vmax=tracks_xyz.shape[1] - 1)
        colours = np.zeros((num_tracks, 3), dtype=np.float32)
        for t_idx in range(num_tracks):
            color = color_map(cmap_norm(t_idx))[:3]
            colours[t_idx] = color
        colours = colours[:, None, :].repeat(2, axis=1)

        n_frames = tracks_xyz.shape[0]
        segment_nodes: list[viser.LineSegmentsHandle] = []
        for k in range(1, n_frames):
            segment = tracks_xyz[k-1:k+1].swapaxes(0, 1)
            # colours = (0.0, 0.0, 1.0)
            segment_node = self._server.scene.add_line_segments(
                f"/track_vis/{k}",
                segment,
                colours
            )
            segment_node.visible = False
            segment_nodes.append(segment_node)
        self._segment_nodes = segment_nodes

    def set_current_frame(self, f_idx: int):
        start_idx = max(1, f_idx - self._trail_length + 1)
        for node in self._segment_nodes:
            node.visible = False
        for idx in range(start_idx, f_idx + 1):
            self._segment_nodes[idx-1].visible = True


class ViserViewer:
    def __init__(self, model, device, port=8080):
        self.device = device
        self.model = model
        self.port = port

        self.S = 5 # num_frames
        self.need_update = True
        self.need_sequence_change = False
        self.is_playing = False
        self.last_update_time = time.time()

        self.server = viser.ViserServer(port=self.port)
        self._setup_gui()
        self._setup_event_handlers()

        self._track_visualiser = TrackVisualiser(self.server)

    def _setup_gui(self):
        server = self.server
        server.gui.configure_theme(control_layout="floating", control_width="large", show_logo=False)
        self.seq_selector = server.gui.add_button("Next example")
        self.play_button = server.gui.add_button("Play")
        self.scene_label = server.gui.add_text(
            "Sequence ID",
            initial_value="",
            disabled=True
        )

        self.gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.0005,
            max=0.002,
            step=0.0005,
            initial_value=0.001,
        )

        self.gui_timestep = server.gui.add_slider(
            "Time",
            min=0,
            max=self.S-1,
            step=1,
            initial_value=0,
        )
        self.conf_slider = server.gui.add_slider(
            "Confidence",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=0.3,
        )
        self.prev_timestep = self.gui_timestep.value

        self.rgb0_vis = self.server.gui.add_image(
            np.ones((100,100,3), dtype=np.uint8) * 255,
            label="rgb_0"
        )
        self.rgbt_vis = self.server.gui.add_image(
            np.ones((100,100,3), dtype=np.uint8) * 255,
            label="rgb_t"
        )

    def set_scene_label(self, example_idx):
        seq_idx, frame_idx = self.dataset.idx_to_seq_frame_id(example_idx)
        scene_name = self.dataset.seq_keys()[seq_idx]
        self.scene_label.value = f"{scene_name}_{frame_idx}"
        print("setting scene label", f"{scene_name}_{frame_idx}")

    def _setup_event_handlers(self):
        @self.seq_selector.on_click
        def _(_) -> None:
            """Choose random sequence to display"""
            with self.server.atomic():
                num_scenes = len(self.dataset)
                example_idx = np.random.randint(num_scenes)
                self.set_scene_label(example_idx)
                views = self.dataset[example_idx]
                views = process_example(views, self.device)
                pointmaps, extrinsic, _, gt_extrinsic = compute_predictions(self, model, views)
                self.visualise_reconstruction(views, pointmaps, extrinsic, gt_extrinsic)
            self.server.flush()  # Optional!
            self.need_update = True
            self.need_sequence_change = True

        @self.play_button.on_click
        def _(_) -> None:
            self.is_playing = not self.is_playing
            self.play_button.text = "Pause" if self.is_playing else "Play"

        @self.gui_point_size.on_update
        def _(_):
            for node in self.point_nodes:
                node.point_size = self.gui_point_size.value

        @self.conf_slider.on_update
        def _(_):
            self.need_update = True

        @self.gui_timestep.on_update
        def _(_) -> None:
            """Toggle frame visibility when the timestep slider changes"""
            current_timestep = self.gui_timestep.value
            with self.server.atomic():
                # Toggle visibility.
                self.frame_nodes[current_timestep].visible = True
                self.frame_nodes[self.prev_timestep].visible = False
            self.prev_timestep = current_timestep
            if self._track_visualiser is not None:
                self._track_visualiser.set_current_frame(current_timestep)
            self._update_image_t()
            self.server.flush()  # Optional!

    def continue_loop(self):
        return not self.need_sequence_change

    def set_data(
        self,
        pts3d_v0_t1: Float[Tensor, "s h w 3"],
        confs: Float[Tensor, "h w"],
        img_v0: Float[Tensor, "3 h w"],
        imgs: List[Float[Tensor, "3 h w"]],
        instance_ids,
        panoptic_v0,
        extrinsic,
    ):
        self.S = pts3d_v0_t1.shape[0]
        self.gui_timestep.max = self.S - 1

        self.pts3d_v0_t1 = pts3d_v0_t1
        self.img_v0 = img_v0
        self.imgs = imgs
        self.panoptic_v0 = panoptic_v0
        self.instance_ids = instance_ids
        self.confs = confs
        self.extrinsic = extrinsic # [1, S, 3, 4]

        self.need_update = True
        self.need_sequence_change = False

    def update(self):
        if not self.need_update:
            return
        self._do_update()
        self.need_update = False

    def _do_update(self):
        self.server.scene.reset()

        img_v0 = self.img_v0
        rgb_v0 = (img_v0 * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()

        def get_coloured_pointclouds(pts_img, color=None):
            return {
                "pts3d": pts_img.view(-1, 3),
                "rgb": rgb_v0.reshape(-1, 3) if color is None else color,
                "conf": self.confs.view(-1)
            }

        points3d = dict()
        for s in range(self.S):
            points3d[f"v0_t{s}"] = get_coloured_pointclouds(self.pts3d_v0_t1[s])
        point_size = float(self.gui_point_size.value)

        T = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)

        view_colours = np.array([
            [0, 0, 1],     # blue
            [1, 0, 0],     # red
            [0, 1, 0],     # green
            [1, 1, 0],     # yellow
            [1, 0, 1],     # magenta
            [0, 1, 1],     # cyan
            [0.5, 0, 0],   # dark red
            [0, 0.5, 0],   # dark green
            [0, 0, 0.5],   # dark blue
            [0.5, 0.5, 0]  # olive
        ], np.float32)

        if self.extrinsic is not None:
            extrinsic = self.extrinsic # [1, S, 3, 4]
            S = extrinsic.shape[0]

            T_c2ws = [extrinsic[s] for s in range(S)]

            for v, T_c2w in enumerate(T_c2ws):
                T_c2w = (T @ T_c2w).numpy()
                H, W = img_v0.shape[1:3]
                f_x = 600
                fov = 2 * np.arctan2(W / 2, f_x)
                aspect = W / H
                self.server.scene.add_camera_frustum(
                    f"/frames/t{v}/camera/pred",
                    fov=fov,
                    aspect=aspect,
                    scale=0.1,
                    color=view_colours[0],
                    # image=rgb[::downsample_factor, ::downsample_factor],
                    wxyz=vt.SO3.from_matrix(T_c2w[:3, :3]).wxyz,
                    position=T_c2w[:3, -1],
                )

        for pts in points3d.values():
            pts["pts3d"] = transform_points(T, pts["pts3d"])

        # TODO: choose reference frame
        reference_frame_id = 0

        confs = points3d[f"v0_t{reference_frame_id}"]["conf"]
        thresh = confs[confs.argsort()][int(confs.size()[0] * self.conf_slider.value)].item()
        good_points = (confs > thresh).numpy()

        tracks = torch.stack([points3d[f"v0_t{s}"]["pts3d"] for s in range(self.S)])
        tracks = tracks[:, good_points, :]
        if self._track_visualiser is not None:
            tracks_filtered = self._track_visualiser.remove_static_tracks(tracks)
            self._track_visualiser.set_data(tracks_filtered)

        frame_nodes: list[viser.FrameHandle] = []
        point_nodes: list[viser.PointCloudHandle] = []
        for s in range(self.S):
            v = points3d[f"v0_t{s}"]
            pts3d = v["pts3d"]
            colours = v["rgb"]
            pts3d_ = pts3d.numpy()[good_points, :]
            colours_ = colours if isinstance(colours, tuple) else colours[good_points]
            point_node = self.server.scene.add_point_cloud(
                name=f"/frames/t{s}/xyz",
                points=pts3d_,
                colors=colours_,
                point_size=point_size,
            )
            point_nodes.append(point_node)
            frame_node = self.server.scene.add_frame(f"/frames/t{s}", show_axes=False)
            frame_node.visible = s == self.gui_timestep.value
            frame_nodes.append(frame_node)
        self.point_nodes = point_nodes
        self.frame_nodes = frame_nodes
        # Hide all but the current frame.

        scene_centre = points3d["v0_t0"]["pts3d"].mean(dim=0)

        for client in self.server.get_clients().values():
            camera = client.camera
            camera.look_at = scene_centre

        self.rgb0_vis.image = rgb_v0
        self._update_image_t()

    def _update_image_t(self):
        rgb_vt = (self.imgs[self.gui_timestep.value] * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()
        self.rgbt_vis.image = rgb_vt

    def visualise_reconstruction(self, images, pred, extrinsic):
        S = len(pred)
        pts3d_all = [pr["pts3d"] for pr in pred]
        conf_all = [pr["conf"] for pr in pred]

        #                              tgt_id src_id
        #                                 |     |
        pts3d_v0 = torch.stack([pts3d_all[s][:, 0] for s in range(S)], dim=1)
        pred_dynamic = dict(pts3d=pts3d_v0)

        pred_pts_t1 = pred_dynamic["pts3d"]

        pts3d_t1 = pred_pts_t1[0].detach()

        indices = torch.arange(S).to(torch.int64)
        pts3d_t1 = pts3d_t1[indices]
        confs_t1 = conf_all[0][0, 0]
        if extrinsic is not None:
            extrinsic = extrinsic[indices, ...].cpu()

        H, W = images.shape[-2:]
        imgs = images.cpu()
        img_v0 = images[0] # [3 H W]

        panoptic_1 = torch.zeros((H, W), dtype=torch.uint8, device=self.device)
        valid_instances = []

        self.set_data(
            pts3d_t1.cpu(),
            confs_t1.cpu(),
            img_v0.cpu(),
            imgs,
            valid_instances,
            panoptic_1.cpu(),
            extrinsic,
        )

    def run(self):
        """Run the visualization event loop"""
        while True:
            current_time = time.time()
            if self.is_playing and current_time - self.last_update_time > 0.1:  # 0.5 seconds per frame
                self.gui_timestep.value = (self.gui_timestep.value + 1) % self.S
                self.last_update_time = current_time
            self.update()
            time.sleep(1e-3)

def process_example(views, device):
    tensors = ['img', 'camera_pose', 'T_WV_norm', 'camera_intrinsics', 'pts3d_t0', 'pts3d_t1', 'valid_mask_t0', 'valid_mask_t1'] #, "view_idxs"]
    for view in views:
        # print(view["view_idxs"])
        for name in tensors:
            if name not in view:
                continue
            view[name] = view[name][None, ...]
            if isinstance(view[name], np.ndarray):
                view[name] = torch.from_numpy(view[name])

    for view in views:
        for name in tensors:  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    return views


def compute_predictions(model, images):
    print("model inference started")

    start = time.perf_counter()

    with torch.no_grad():
        result = model.inference(None, images=images.unsqueeze(0))
    print("model inference finished")
    end = time.perf_counter()
    print(f"Execution time: {end - start:.6f} seconds")

    pointmaps = result["pointmaps"]

    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    pose_enc = result["pose_enc"]
    HW = pointmaps[0]["pts3d"].shape[2:4]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, HW)
    extrinsic = extrinsic[0]
    S = extrinsic.shape[0]
    extrinsic_CW = torch.cat([extrinsic.cpu(), repeat(torch.tensor([0, 0, 0, 1]), "c -> s 1 c", s=S)], dim=1)
    extrinsic_WC = torch.linalg.inv(extrinsic_CW)

    return pointmaps, extrinsic_WC, intrinsic


def extract_frames(input_video):
    torch.cuda.empty_cache()

    video_path = input_video
    vs = cv2.VideoCapture(video_path)

    fps = float(vs.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_interval = max(int(fps / max(VIDEO_SAMPLE_HZ, 1e-6)), 1)

    count = 0
    frame_num = 0
    images = []
    try:
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            if count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(frame)
                frame_num += 1
            count += 1
    finally:
        vs.release()

    return images


def preprocess_images(images_np, mode="crop"):
    # Check for empty list
    if len(images_np) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for img_np in images_np:

        # Open image
        img = Image.fromarray(img_np)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(images_np) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def load_model(cfg, device) -> VDPM:
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


@hydra.main(config_path="configs", config_name="visualise")
def main(cfg) -> None:
    device = 'cuda:0'
    torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

    model = load_model(cfg, device)

    viewer = ViserViewer(model, device, cfg.vis.port)

    input_video = cfg.vis.input_video
    frames = extract_frames(input_video)
    images = preprocess_images(frames).to(device)  # (N, 3, H, W)

    pointmaps, extrinsic, _ = compute_predictions(model, images)
    viewer.visualise_reconstruction(images, pointmaps, extrinsic)
    viewer.run()


if __name__ == "__main__":
    main()
