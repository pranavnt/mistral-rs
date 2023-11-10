import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
from typing import List

class CodeTextDataset(Dataset):
    """
    A PyTorch Dataset for loading data from a JSONL file where each line is a JSON object
    with a 'text' field containing code.
    """
    def __init__(self, jsonl_path: str, tokenizer_model_path: str):
        """
        Args:
            jsonl_path (str): The path to the JSONL file containing the data.
            tokenizer_model_path (str): The path to the tokenizer model file.
        """
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer_model_path = tokenizer_model_path
        self.data = self._load_data()
        self.tokenizer = self._load_tokenizer()

    def _load_data(self) -> List[str]:
        """
        Loads the JSONL file and extracts the text from each line.
        
        Returns:
            List[str]: The list of code strings.
        """
        with self.jsonl_path.open('r', encoding='utf-8') as file:
            return [json.loads(line)['text'] for line in file]

    def _load_tokenizer(self):
        """
        Loads the tokenizer model.
        
        Returns:
            The tokenizer model.
        """
        from sentencepiece import SentencePieceProcessor
        tokenizer = SentencePieceProcessor(model_file=str(self.tokenizer_model_path))
        return tokenizer

    def __len__(self) -> int:
        """
        Returns the number of items in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves the dataset item at the specified index and tokenizes it.
        
        Args:
            idx (int): The index of the item.
            
        Returns:
            torch.Tensor: The tokenized code as tensor.
        """
        code_text = self.data[idx]
        encoded_text = self.tokenizer.encode(code_text, out_type=int)
        return torch.tensor(encoded_text, dtype=torch.long)

    def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Collate function to be used by the DataLoader to pad the sequences to the same length.
        
        Args:
            batch (List[torch.Tensor]): A batch of tokenized code tensors.
            
        Returns:
            torch.Tensor: A tensor of the padded batch.
        """
        batch_padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        return batch_padded
