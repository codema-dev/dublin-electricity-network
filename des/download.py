from os import path
import urllib

from tqdm.auto import tqdm


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


def download(url, to_filepath):

    if path.exists(to_filepath):
        print(f"Skipping as {to_filepath} has already been downloaded...")
    else:
        with TqdmUpTo(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=to_filepath,
        ) as t:  # all optional kwargs
            urllib.request.urlretrieve(
                url,
                filename=to_filepath,
                reporthook=t.update_to,
            )
            t.total = t.n