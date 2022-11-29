#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 15:08:21 2022

@author: edouard.duchesnay@cea.fr
"""
import os
import os.path
import numpy as np
import pandas as pd
import urllib.request
import click

from sklearn.model_selection import train_test_split
from shutil import copyfile, make_archive, unpack_archive, move

try:
    PATH_DATA = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"
    )
except NameError:
    PATH_DATA = "data"

os.makedirs(PATH_DATA, exist_ok=True)

def fetch_data(files, dst, base_url, verbose=1):
    """Fetch dataset.

    Args:
        files (str): file.
        dst (str): destination directory.
        base_url (str): url, examples:


            ftp://ftp.cea.fr/pub/unati/people/educhesnay/data/brain_anatomy_schizophrenia_data
    Returns:
        downloaded ([str, ]): paths to downloaded files.

    """
    downloaded = []
    for file in files:
        src_filename = os.path.join(base_url, file)
        dst_filename = os.path.join(dst, file)
        if not os.path.exists(dst_filename):
            if verbose:
                print("Download: %s" % src_filename)
            urllib.request.urlretrieve(src_filename, dst_filename)
        downloaded.append(dst_filename)
    return downloaded

if __name__ == "__main__":
    fetch_data(files=['train_participants.csv', 'train_rois.csv', 'train_vbm.npz',
                      'validation_participants.csv', 'validation_rois.csv', 'validation_vbm.npz'],
               dst=PATH_DATA,
               base_url='ftp://ftp.cea.fr/pub/unati/people/educhesnay/data/brain_anatomy_schizophrenia_data',
               verbose=1)

    # validation => test
    move(os.path.join(PATH_DATA, 'validation_participants.csv'),
         os.path.join(PATH_DATA, 'test_participants.csv'))
    move(os.path.join(PATH_DATA, 'validation_vbm.npz'),
         os.path.join(PATH_DATA, 'test_vbm.npz'))
    move(os.path.join(PATH_DATA, 'validation_rois.csv'),
         os.path.join(PATH_DATA, 'test_rois.csv'))
