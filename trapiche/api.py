from pathlib import Path
import numpy as np
from .biome2vec import vectorise_sample

class Community2vec:
    def __init__(self):
        pass
    
    def transform(list_of_tax_files):
        """Vectorise one or many samples from taxonomy annotation files.

        Flexible input forms are accepted (all are normalised internally to the
        original list-of-lists of file paths interface):

        1. Single file path (str / Path): that file represents one sample.
        2. Directory path: every *.tsv or *.tsv.gz file inside (non-recursive)
           is treated as belonging to one sample.
        3. Flat list of file paths: all those files together are one sample.
        4. List of lists of file paths (original behaviour): each inner list is a sample.

        Returns
        -------
        np.ndarray
            Array with shape (n_samples, embedding_dim). If no vectors can be
            produced an array with shape (n_samples, 0) is returned (or (0, 0) if
            there are no samples at all).
        """
        self.vectorised_samples = vectorise_sample(list_of_tax_files)
        return self.vectorised_samples
    
    def save(self, path: str | Path) -> None:
        """Save the vectorised samples to a .npy file."""
        if not hasattr(self, 'vectorised_samples'):
            raise ValueError("No vectorised samples to save. Call transform() first.")
        np.save(path, self.vectorised_samples)
