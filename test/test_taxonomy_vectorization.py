import unittest
from unittest.mock import patch

import numpy as np

from trapiche.taxonomy_vectorization import vectorise_samples


class TestVectoriseSamples(unittest.TestCase):
    @patch(
        "trapiche.taxonomy_vectorization.genre_to_taxonomy_vectorization",
        return_value=np.ones(3),
    )
    @patch(
        "trapiche.taxonomy_vectorization.tax_annotations_from_file",
        side_effect=[
            [("p__Bacillota", "g__Bacteroides")],
            [("p__Bacillota", "g__Veillonella")],
        ],
    )
    def test_combines_taxonomy_edges_from_multiple_files(
        self, tax_annotations_from_file, vectorize
    ):
        samples = [
            {
                "sample_taxonomy_paths": ["sample_pr2.mseq", "sample_silva.mseq"],
            }
        ]

        result = vectorise_samples(
            samples,
            model_name="test-model",
            model_version="test-version",
        )

        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(tax_annotations_from_file.call_count, 2)
        vectorize.assert_called_once_with(
            {"Bacteroides", "Veillonella"},
            model_name="test-model",
            model_version="test-version",
            local_model_dir=None,
        )

    @patch(
        "trapiche.taxonomy_vectorization.genre_to_taxonomy_vectorization",
        return_value=np.ones(3),
    )
    @patch(
        "trapiche.taxonomy_vectorization.tax_annotations_from_file",
        side_effect=[
            Exception("boom"),
            [("p__Bacillota", "g__Veillonella")],
        ],
    )
    def test_skips_failing_file_and_keeps_later_files(self, tax_annotations_from_file, vectorize):
        samples = [
            {
                "sample_taxonomy_paths": ["sample_pr2.mseq", "sample_silva.mseq"],
            }
        ]

        result = vectorise_samples(
            samples,
            model_name="test-model",
            model_version="test-version",
        )

        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(tax_annotations_from_file.call_count, 2)
        vectorize.assert_called_once_with(
            {"Veillonella"},
            model_name="test-model",
            model_version="test-version",
            local_model_dir=None,
        )


if __name__ == "__main__":
    unittest.main()
