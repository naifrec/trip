# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3.9.7 ('base')
#     language: python
#     name: python3
# ---

# +
import math

import numpy as np
from PIL import Image

from trip.block import (get_blocks_from_array, get_array_from_blocks,
                        block_shuffle, glitch_repeat_pixels)
# -

image = Image.open('../data/amours.jpeg')
image

np.random.seed(1)
block_shuffle(
    glitch_repeat_pixels(
        image, axis=0, block_size=128, percent_corrupted=0.45, margin_corrupted=0.33),
    channel=0, block_size=16,
)


np.random.seed(1)
block_shuffle(
    glitch_repeat_pixels(
        block_shuffle(
            image, channel=2, block_size=16),
        axis=0, block_size=128, percent_corrupted=0.45, margin_corrupted=0.33,
    ),
    channel=1, block_size=64,
)

np.random.seed(1)
glitch_repeat_pixels(
    block_shuffle(
            block_shuffle(
            glitch_repeat_pixels(
                block_shuffle(
                    image, channel=2, block_size=16),
                axis=0, block_size=128, percent_corrupted=0.45, margin_corrupted=0.33,
            ),
            channel=1, block_size=64),
        channel=2, block_size=128,
    ),
    axis=1, block_size=64, percent_corrupted=0.58, margin_corrupted=0.13,
)

np.random.seed(1)
block_shuffle(image, channel=0, block_size=16)


