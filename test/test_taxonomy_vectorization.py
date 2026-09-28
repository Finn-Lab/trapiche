import unittest
from unittest.mock import patch

import numpy as np

from trapiche.taxonomy_vectorization import genus_from_edges_subgraph, vectorise_samples
from trapiche.utils import extract_taxonomic_edges_from_tsv_row, krona_read


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


class TestCandidatusParsing(unittest.TestCase):
    ROW = (
        "251837\t21.0\tsk__Bacteria;k__;p__Candidatus_Melainabacteria;c__;"
        "o__Candidatus_Obscuribacterales;f__;g__Candidatus_Obscuribacter;"
        "s__Candidatus_Obscuribacter_phosphatis"
    )

    def test_candidatus_prefix_stripped_from_underscored_names(self):
        edges = extract_taxonomic_edges_from_tsv_row(self.ROW)

        self.assertIn(("o__Obscuribacterales", "g__Obscuribacter"), edges)
        self.assertFalse(any("Candidatus" in node for edge in edges for node in edge))

    def test_candidatus_genus_resolves_to_real_genus(self):
        edges = krona_read([self.ROW])

        self.assertEqual(genus_from_edges_subgraph(edges), {"Obscuribacter"})

    def test_candidatus_prefix_stripped_from_spaced_names(self):
        edges = extract_taxonomic_edges_from_tsv_row(
            "1\t1.0\tsk__Bacteria;o__Obscuribacterales;g__Candidatus Obscuribacter"
        )

        self.assertIn(("o__Obscuribacterales", "g__Obscuribacter"), edges)


if __name__ == "__main__":
    unittest.main()
