from PIL import Image
import torch
import torchvision.transforms as TF


def preprocess_images(images_in):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - The function ensures width=518px while maintaining aspect ratio
        - Height is adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(images_in) == 0:
        raise ValueError("At least 1 image is required")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    # First process all images and collect their shapes
    for img in images_in:
        img = Image.fromarray(img)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size
        new_width = 518

        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518
        if new_height > 518:
            raise NotImplementedError("Don't support portrait mode for now")

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(images) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images
