import sys

from PIL import Image
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

image = Image.open("imageFales/dent.jpg")

new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32
new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

pad = 16
image = image.crop((pad, pad, image.width - pad, image.height - pad))
img = image.convert('L')  # convert image to 8-bit grayscale
WIDTH, HEIGHT = img.size

data = list(img.getdata()) # convert image data to a list of integers

print(len(data))
path = "dataset/dent"
np.savetxt(path, data)