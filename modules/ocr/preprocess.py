"""
Image preprocessing for OCR: deskew (straighten rotation), denoise,
and binarize (convert to clean black-and-white) before handing the
image to an OCR engine. Real scanned/photographed documents are often
skewed, noisy, or poorly lit — cleaning them up first typically
improves OCR accuracy significantly.
"""

import cv2
import numpy as np

from modules.core.logging_config import get_logger

logger = get_logger(__name__)


def _deskew(image: np.ndarray) -> np.ndarray:
    """
    Detects and corrects rotation/skew. Uses OpenCV's minAreaRect on
    the thresholded image's non-zero pixels to estimate the dominant
    text angle, then rotates to straighten it.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        # Not enough content to reliably estimate an angle — skip deskewing.
        return image

    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect's angle convention needs this correction to represent
    # actual rotation-from-horizontal rather than its raw [-90, 0) range.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _denoise(image: np.ndarray) -> np.ndarray:
    """Removes small-scale noise (sensor grain, compression artifacts)."""
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def _binarize(image: np.ndarray) -> np.ndarray:
    """
    Converts to clean black-and-white using adaptive thresholding —
    'adaptive' because it computes the threshold per local region
    rather than one global value, handling uneven lighting/shadows
    better than a single fixed threshold would.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15,
    )
    return binary


def preprocess_image(image_path: str, output_path: str | None = None) -> np.ndarray:
    """
    Runs the full preprocessing pipeline: denoise -> deskew -> binarize.
    Returns the processed image array; optionally saves it to disk.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    denoised = _denoise(image)
    deskewed = _deskew(denoised)
    binarized = _binarize(deskewed)

    if output_path:
        cv2.imwrite(output_path, binarized)
        logger.info(f"Saved preprocessed image to {output_path}")

    return binarized