from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import urllib3
import multiprocessing

from PIL import Image
from tqdm import tqdm
from urllib3.util import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def my_resize(im, min_size):
    w, h = im.size
    if w < h:
        new_width = min_size
        new_height = int(min_size * (float(h) / w))

    if h <= w:
        new_height = min_size
        new_width = int(min_size * (float(w) / h))

    return (im.resize((new_width, new_height), Image.ANTIALIAS))


def download_image(fnames_and_urls):
    """
    download image and save its with 90% quality as JPG format
    skip image downloading if image already exists at given path
    :param fnames_and_urls: tuple containing absolute path and url of image
    """
    fname, url = fnames_and_urls
    if not os.path.exists(fname):
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb = my_resize(image_rgb, 256)
        image_rgb.save(fname, format='JPEG', quality=100)


def parse_dataset(_dataset, _outdir, _max=None):
    """
    parse the dataset to create a list of tuple containing absolute path and url of image
    :param _dataset: dataset to parse
    :param _outdir: output directory where data will be saved
    :param _max: maximum images to download (change to download all dataset)
    :return: list of tuple containing absolute path and url of image
    """
    _fnames_urls = []
    with open(dataset, 'r') as f:
        data = json.load(f)
        for image in data["images"]:
            url = image["url"]
            fname = os.path.join(outdir, "{}.jpg".format(image["imageId"]))
            _fnames_urls.append((fname, url))
    # return _fnames_urls[:_max]
    return _fnames_urls


dataset = "ann/train.json"
outdir =  "img/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

# parse json dataset file
fnames_urls = parse_dataset(dataset, outdir)

# download data
pool = multiprocessing.Pool(processes=12)
with tqdm(total=len(fnames_urls)) as progress_bar:
    for _ in pool.imap_unordered(download_image, fnames_urls):
        progress_bar.update(1)

