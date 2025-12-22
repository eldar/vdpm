import numpy as np
import PIL.Image
from PIL.PngImagePlugin import PngInfo


def read_depth_from_png(file_path):
    depth = PIL.Image.open(file_path) # Scale the saved image using the metadata
    max_depth = float(depth.info["max_value"])
    min_depth = float(depth.info["min_value"])
    depth = np.array(depth).astype(np.float32)

    # Scale from uint16 range
    depth = (depth / (2 ** 16 - 1)) * (max_depth - min_depth) + min_depth

    # replace the magic constant with positive infinity
    depth[depth == -1.0] = np.inf
    return depth


def write_depth_to_png(outpath_file, depth_pred):
    # MoGe predicts infinity for sky pixels
    depth_pred[np.isinf(depth_pred)] = -1.0

    min_depth = np.min(depth_pred)
    max_depth = np.max(depth_pred)

    # Normalize to [0, 1]
    depth_pred_normalized = (depth_pred - min_depth) / (max_depth - min_depth)

    # Scale to uint16 range
    depth_pred_scaled = np.round(depth_pred_normalized * (2 ** 16 - 1)).astype(np.uint16)

    # Save image with pred and error as uint16
    depth_pred_img = PIL.Image.fromarray(depth_pred_scaled)
    metadata = PngInfo()
    metadata.add_text("max_value", str(max_depth))
    metadata.add_text("min_value", str(min_depth))
    depth_pred_img.save(outpath_file,
                        format="PNG", mode="I", pnginfo=metadata)

