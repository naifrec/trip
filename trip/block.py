"""
Block based functions.

"""
import math

import numpy as np
from PIL import Image


def get_blocks_from_array(array, block_size):
    """
    Return blocks from an image array.

    Parameters
    ----------
    array : numpy array
        Array with shape (height, width, channels)
    block_size : int
        Block size.

    Returns
    -------
    blocks : numpy array
        Blocks from array with shape (height_blocks, block_size,
        width_blocks, block_size, channels). Note that due to
        automatic trimming it is possible that height_blocks *
        block_size != block_size

    """
    height, width, n_channels = array.shape

    # trim if trailing pixels
    if height % block_size:
        array = array[: block_size * (height // block_size)]
    if width % block_size:
        array = array[:, : block_size * (width // block_size)]
    return array.reshape(
        width // block_size,
        block_size,
        height // block_size,
        block_size,
        n_channels,
    )


def get_array_from_blocks(blocks):
    """
    Return array from an image blocks.

    Parameters
    ----------
    blocks : numpy array
        Blocks from array with shape (height_blocks, block_size,
        width_blocks, block_size, channels).

    Returns
    -------
    array : numpy array
        Array with shape (height, width, channels).

    """
    n_blocks_x, block_size_x, n_blocks_y, block_size_y, n_channels = blocks.shape
    array = blocks.reshape(
        n_blocks_x * block_size_x, n_blocks_y * block_size_y, n_channels
    )
    return array


def block_shuffle(image, channel, block_size):
    """
    Randomly shuffle blocks of a specific channel.

    """
    array = np.asarray(image)
    n_channels = array.shape[-1]

    blocks = get_blocks_from_array(array, block_size)
    n_blocks_x, _, n_blocks_y, _, _ = blocks.shape

    blocks_channel = blocks[..., channel]

    x_coordinates = np.repeat(np.arange(n_blocks_x), n_blocks_y)
    y_coordinates = np.repeat(np.arange(n_blocks_y), n_blocks_x)
    np.random.shuffle(x_coordinates)
    np.random.shuffle(y_coordinates)
    blocks_channel_shuffled = blocks_channel[x_coordinates, :, y_coordinates, :]
    blocks_channel_shuffled = blocks_channel_shuffled.reshape(
        n_blocks_x,
        n_blocks_y,
        block_size,
        block_size,
    )
    blocks_channel_shuffled = blocks_channel_shuffled.transpose(0, 2, 1, 3)
    blocks_channels = [
        blocks[..., i : i + 1] if (i != channel) else blocks_channel_shuffled[..., None]
        for i in range(n_channels)
    ]
    array_corrupted = np.concatenate(blocks_channels, axis=-1)
    array_corrupted = get_array_from_blocks(array_corrupted)
    return Image.fromarray(array_corrupted)


def glitch_repeat_pixels(image, axis, percent_corrupted, margin_corrupted, block_size):
    """
    Glitch repeat pixels along a specific axis.

    """
    array = np.asarray(image)

    blocks = get_blocks_from_array(array, block_size)
    n_blocks_x, _, n_blocks_y, _, _ = blocks.shape

    n_blocks_primary = n_blocks_x if axis == 0 else n_blocks_y
    n_blocks_secondary = n_blocks_y if axis == 0 else n_blocks_x
    start_index_primary = math.ceil(n_blocks_primary * (1 - percent_corrupted))
    end_index_primary = math.ceil(
        n_blocks_primary * ((1 - percent_corrupted) + margin_corrupted)
    )

    start_index_secondary = np.random.randint(low=0, high=n_blocks_secondary)

    start_block_indices = np.random.randint(
        low=start_index_primary,
        high=end_index_primary,
        size=(n_blocks_secondary - start_index_secondary),
    )

    for offset, index in enumerate(start_block_indices):
        if axis == 0:
            value = blocks[index, -2:-1, start_index_secondary + offset, :]
            blocks[index:, :, start_index_secondary + offset, :] = value[None, :, :]
        elif axis == 1:
            value = blocks[start_index_secondary + offset, :, index, -2:-1]
            blocks[start_index_secondary + offset, :, index:, :] = value[:, None, :]

    array_corrupted = get_array_from_blocks(blocks)
    return Image.fromarray(array_corrupted)
