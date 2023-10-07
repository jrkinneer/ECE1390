import numpy as np

def hough_peaks(H, Q):
    """_summary_

    Args:
        img_edge (NDArray): array of a hough transform
        Q ('uint8'): the Q number of local maxima from the img you want to find

    Returns:
        NDArray: (Qx2) array of the indecis in img_edge of the local maxima
    """
    indices = np.argpartition(H.flatten(), -2)[-Q:]
    return np.vstack(np.unravel_index(indices, H.shape)).T