import torch
import json
import os
import numpy as np
import argparse
from transformers import BertTokenizerFast, BertForSequenceClassification, BertConfig

class TextClassifier:
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the TextClassifier with the model, tokenizer, and configuration.

        :param model_path: Path to the trained model directory.
        :param device: Device to run the model on (auto-detect if None).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model_config = BertConfig.from_pretrained(model_path)
        self.id2label = (
            self.model_config.id2label
            if hasattr(self.model_config, "id2label")
            else {i: f"label_{i}" for i in range(num_classes)}
        )
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, config=self.model_config
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load class-specific thresholds if provided
        self.thresholds = self._load_thresholds(model_path)
        
        # Attribute to cache the NLP model (for sentence splitting)
        self._nlp = None

    def _load_thresholds(self, model_path):
        threshold_path = os.path.join(model_path, "class_thresholds.json")
        if os.path.exists(threshold_path):
            with open(threshold_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def predict_probability(self, texts, max_length=256):
        """
        Perform inference on a list of texts.

        :param texts: List of strings to classify.
        :param max_length: Maximum sequence length for tokenization.
        :return: NumPy array of probabilities with shape (num_samples, num_classes).
        """
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits).cpu().numpy()

        return probabilities

    def _probabilities_to_predictions(self, probabilities, default_threshold=0.01):
        """
        Return a binary mask of predictions based on a decision rule.
        
        :param probabilities: NumPy array of shape (num_samples, num_classes) containing probability scores.
        :param default_threshold:
            - If a numeric value, each class is predicted (mask=1) if its probability is >= threshold 
              (overridden by self.thresholds for specific classes, if available).
            - If "max", only the class with the highest probability is selected (mask=1 for that class).
            - If "top-N" (e.g., "top-3"), the N classes with the highest probabilities are selected.
        :return: NumPy array of shape (num_samples, num_classes) with binary values (1 for predicted, 0 otherwise).
        """
        num_samples, num_classes = probabilities.shape
        mask = np.zeros_like(probabilities, dtype=int)
        
        
        # Build a mapping from class index to its threshold (using class-specific thresholds if provided).
        index_thresholds = {}
        for i in range(num_classes):
            label = self.id2label.get(i, f"label_{i}")
            if self.thresholds:
                index_thresholds[i] = self.thresholds.get(label, default_threshold)
            else:
                index_thresholds[i] = default_threshold

        # Process each sample's probability vector.
        for j, prob in enumerate(probabilities):
            if default_threshold == "max":
                # Only the highest probability class is selected.
                max_index = int(np.argmax(prob))
                mask[j, max_index] = 1

            elif isinstance(default_threshold, str) and default_threshold.startswith("top-"):
                try:
                    top_n = int(default_threshold.split("-")[1])
                except ValueError:
                    raise ValueError("Invalid default_threshold format for top-N. Expected format 'top-N' where N is an integer.")
                # Get indices of the top N probabilities (sorted in descending order).
                top_indices = np.argsort(prob)[::-1][:top_n]
                mask[j, top_indices] = 1

            elif isinstance(default_threshold, (int, float)):
                # For each class, set mask to 1 if probability meets/exceeds the threshold.
                for i, p in enumerate(prob):
                    if p >= index_thresholds[i]:
                        mask[j, i] = 1
            else:
                raise ValueError("default_threshold must be a numeric value, 'max', or a string in the format 'top-N'.")

        return mask

            
    def predict(self, texts, max_length=256, default_threshold=0.01, predict_splitting=False):
        """
        Perform inference on a list of texts.

        :param texts: List of strings to classify.
        :param max_length: Maximum sequence length for tokenization.
        :param default_threshold:
            - If a numeric value, each class is predicted if its probability is >= threshold 
              (overridden by self.thresholds for specific classes, if available).
            - If "max", only the class with the highest probability is selected.
            - If "top-N" (e.g., "top-3"), the N classes with the highest probabilities are selected.
        :param predict_splitting: If True, splits each text into sentences (using a domain-aware splitter)
                                  and aggregates the maximum probability per class across sentences (including the full text).
        :return: List of Lists with the predicted classes.
        """
        if isinstance(texts, str):
            texts = [texts]  # Ensure texts is a list

        if predict_splitting:
            aggregated_probs = []
            for text in texts:
                # Split the text into sentences and include the full text for context.
                splitted_text = self._split_sentences(text) + [text]
                probabilities = self.predict_probability(splitted_text, max_length=max_length)
                # Compute the maximum probability per class across all segments.
                max_probs = np.max(probabilities, axis=0)
                aggregated_probs.append(max_probs)
            self.probabilities = np.array(aggregated_probs)
        else:
            self.probabilities = self.predict_probability(texts, max_length=max_length)

        self.predictions_mask = self._probabilities_to_predictions(self.probabilities, default_threshold=default_threshold)

        self.predictions = []
        for sample_mask in self.predictions_mask:
            sample_preds = []
            for idx, flag in enumerate(sample_mask):
                if flag == 1:
                    sample_preds.append(self.id2label.get(idx, f"label_{idx}"))
            self.predictions.append(sample_preds)
            
        return self.predictions

    def _get_nlp(self):
        """
        Lazy-load and return a spaCy NLP model for sentence splitting. 
        Attempts to load a domain-specific SciSpacy model first; falls back to spaCy's general English model.
        If spaCy is unavailable, returns None.
        """
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
        except ImportError:
            self._nlp = None
            return None

        try:
            # Try loading the SciSpacy model
            self._nlp = spacy.load("en_core_sci_sm")
        except Exception:
            try:
                # Attempt to download the model if it's not available
                spacy.cli.download("en_core_sci_sm")
                self._nlp = spacy.load("en_core_sci_sm")
            except Exception:
                # Fallback to the general English model if download or load fails
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp
    
    def _split_sentences(self, text):
        """
        Split the text into sentences using a domain-aware approach.
        Utilizes a cached spaCy model for efficiency. If spaCy is unavailable, falls back to NLTK.

        :param text: A string representing the text to split.
        :return: List of sentence strings.
        """
        nlp = self._get_nlp()
        if nlp is not None:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            import nltk
            nltk.download("punkt", quiet=True)
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
        return sentences


def main():
    parser = argparse.ArgumentParser(description="Multi-Label Biome Classification of Metagenomic Study Text Descriptions")
    parser.add_argument("--model_path", type=str, default="SantiagoSanchezF/trapiche-biome-classifier", help="Name of HuggingFaceHub model or path to trained model directory. Default: SantiagoSanchezF/trapiche-biome-classifier")
    parser.add_argument("--input_text", type=str, required=True, help="Input text or path to JSON file containing texts")
    parser.add_argument("--output_file", type=str, help="Optional output file to save predictions")
    
    args = parser.parse_args()
    classifier = TextClassifier(args.model_path)
    
    if args.input_text.endswith(".json"):  # Handle batch inference from a file
        with open(args.input_text, "r", encoding="utf-8") as f:
            texts = json.load(f)
    else:
        texts = [args.input_text]
    
    predictions = classifier.predict(texts)
    
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=4)
    else:
        print(json.dumps(predictions, indent=4))


if __name__ == "__main__":
    main()
