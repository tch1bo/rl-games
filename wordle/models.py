import torch

from common.utils import get_logger
from wordle.env import ALPHABET_LEN, GUESS_WORDS, STATE_DIM, WORD_LEN

logger = get_logger()


class MLPActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(STATE_DIM, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, WORD_LEN * ALPHABET_LEN),
        )

        word_mask = torch.zeros(
            (len(GUESS_WORDS), ALPHABET_LEN * WORD_LEN),
            dtype=torch.float,
            requires_grad=False,
        )
        coords = torch.tensor(
            [
                (word_index, letter_index * ALPHABET_LEN + ord(letter) - ord("a"))
                for word_index, word in enumerate(GUESS_WORDS)
                for letter_index, letter in enumerate(word)
            ],
            dtype=torch.long,
        )
        word_mask[coords[:, 0], coords[:, 1]] = 1.0
        self.register_buffer("_word_mask", word_mask, persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_word_logits(self, state: torch.Tensor) -> torch.Tensor:
        """Returns unnormalized logits for words (result.shape[-1] == len(GUESS_WORDS))"""

        return self.mlp(state) @ self.get_buffer("_word_mask").T


class MLPCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(STATE_DIM, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(state)
