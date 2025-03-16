import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List
import torch.nn.utils.rnn as rnn_utils

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # Combine all texts
        all_texts = positive + negative
        
        # Split texts into words and create a unique vocabulary
        all_words = []
        for text in all_texts:
            words = text.split()
            all_words.extend(words)
        
        # Create a lexicographically sorted vocabulary
        unique_words = sorted(set(all_words))
        
        # Create word to index mapping (starting from 1, as 0 is for padding)
        word_to_idx = {word: idx + 1 for idx, word in enumerate(unique_words)}
        
        # Encode each text as a list of indices
        encoded_texts = []
        for text in positive + negative:  # Process positive first, then negative
            words = text.split()
            encoded_text = [float(word_to_idx[word]) for word in words]
            encoded_texts.append(torch.tensor(encoded_text))
        
        # Pad sequences to the same length
        padded_sequences = rnn_utils.pad_sequence(encoded_texts, batch_first=True, padding_value=0.0)
        
        return padded_sequences