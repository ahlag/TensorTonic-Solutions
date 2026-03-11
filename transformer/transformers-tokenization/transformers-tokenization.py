import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]

        self.word_to_id = {}
        self.id_to_word = {}
        
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token

        next_id = len(special_tokens)
        seen = set()

        for text in texts:
            for word in text.split():
                if word not in seen:
                    seen.add(word)
                    self.word_to_id[word] = next_id
                    self.id_to_word[next_id] = word
                    next_id += 1

        self.vocab_size = next_id
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(word, unk_id) for word in text.split()]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        return " ".join(self.id_to_word.get(i, self.unk_token) for i in ids)
