import unittest
from unittest.mock import patch

import numpy as np
import torch


class _Tokenizer:
    def __call__(self, texts, **_kwargs):
        width = max(len(text) for text in texts)
        ids = []
        masks = []
        for text in texts:
            values = [ord(char) % 31 for char in text]
            padding = width - len(values)
            ids.append(values + [0] * padding)
            masks.append([1] * len(values) + [0] * padding)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }


class _Model:
    config = type("Config", (), {"num_labels": 2})()

    def __call__(self, input_ids, attention_mask):
        score = (input_ids * attention_mask).sum(dim=1).float()
        return type("Output", (), {"logits": torch.stack((score, -score), dim=1)})()


class TestTextPredictionBatching(unittest.TestCase):
    def test_batching_preserves_probabilities(self):
        from trapiche.text_prediction import predict_probability

        model = _Model()
        with patch(
            "trapiche.text_prediction.load_text_model",
            return_value=(_Tokenizer(), None, {0: "a", 1: "b"}, model, {}),
        ):
            texts = ["short", "a much longer text", "tiny"]
            one_at_a_time = predict_probability(texts, "model", "1", batch_size=1)
            batched = predict_probability(texts, "model", "1", batch_size=8)

        np.testing.assert_array_equal(one_at_a_time[0], batched[0])
        np.testing.assert_array_equal(one_at_a_time[1], batched[1])

    def test_empty_input_returns_empty_probability_arrays(self):
        from trapiche.text_prediction import predict_probability

        with patch(
            "trapiche.text_prediction.load_text_model",
            return_value=(_Tokenizer(), None, {0: "a", 1: "b"}, _Model(), {}),
        ):
            sigmoid, softmax = predict_probability([], "model", "1")

        self.assertEqual(sigmoid.shape, (0, 2))
        self.assertEqual(softmax.shape, (0, 2))

    def test_predict_preserves_existing_positional_arguments(self):
        from trapiche.text_prediction import predict

        with (
            patch(
                "trapiche.text_prediction.load_text_model",
                return_value=(_Tokenizer(), None, {0: "a", 1: "b"}, _Model(), {}),
            ),
            patch("trapiche.text_prediction.load_biome_herarchy_dict", return_value=({}, {})),
            patch(
                "trapiche.text_prediction.predict_probability",
                return_value=(np.array([[0.9, 0.1]]), np.array([[0.9, 0.1]])),
            ) as predict_probability,
        ):
            # This was the pre-batching positional call shape.
            result = predict(["text"], "model", "1", None, 256, "max", False)

        self.assertEqual(result, [{"a": 0.9}])
        self.assertEqual(predict_probability.call_args.kwargs["batch_size"], 8)
