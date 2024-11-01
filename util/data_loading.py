import numpy as np


def arr_to_complex(arr: np.ndarray) -> np.ndarray:
    """
    Converts a 2-row numpy array to a complex-valued array.

    Parameters:
    - arr (np.ndarray): Input array with shape (2, N).

    Returns:
    - np.ndarray: Complex-valued array with shape (N,).

    Raises:
    - ValueError: If the input array does not have two rows.
    """
    if arr.shape[0] != 2:
        raise ValueError("Input array must have two rows")

    # return complex baseband representation: real part + j * imaginary part
    return arr[0, :] + 1j * arr[1, :]


def read_iqdata(file_path: str) -> tuple:
    """
    Reads IQ data from a .npy file and returns the complex-valued signal, recording metadata, and extended metadata.

    Parameters:
    - file_path (str): Path to the .npy file.

    Returns:
    - tuple: Complex-valued signal (np.ndarray) and extended metadata (dict).
    """
    with open(file_path, 'rb') as f:
        iq_data = np.load(f)
        _ = np.load(f)
        extended_metadata = np.load(f, allow_pickle=True)[0]
        # extended metadata is a dictionary, that is why allow_pickle is set to true.

    complex_sig = arr_to_complex(iq_data)
    return complex_sig, extended_metadata
