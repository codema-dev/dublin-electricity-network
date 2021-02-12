from os import path
from shutil import unpack_archive


def unzip(filename, extract_dir):

    if path.exists(extract_dir):
        print(f"Skipping as {filename} has already been unzipped..")
    else:
        unpack_archive(filename, extract_dir)