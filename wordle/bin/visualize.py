"""Watch a trained wordle policy play a single game in the terminal.

Usage:
    uv run python -m wordle.bin.visualize --checkpoint out/wordle_reinforce/checkpoint_00290000.pt [WORD]
"""

import random
import time
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg

from wordle.env import ANSWER_WORDS, GUESS_WORDS, MAX_GUESSES, WORD_LEN, WordleEnv
from wordle.models import MLPActor

_ANSWER_INDEX = {word: index for index, word in enumerate(ANSWER_WORDS)}


class CliArgs(BaseSettings):
    word: CliPositionalArg[str | None] = Field(
        default=None,
        description="the target word to guess; chosen randomly from the answer list if omitted",
    )
    checkpoint: Path = Field(
        description="path to a checkpoint saved by wordle.bin.train"
    )
    top_k: int = Field(
        default=5,
        ge=0,
        description="number of top candidate words (with probabilities) to show before each guess",
    )
    delay: float = Field(
        default=0.7, ge=0.0, description="seconds to pause between guesses"
    )
    device: Literal["cpu", "cuda"] = "cpu"

    @field_validator("word")
    @classmethod
    def _validate_word(cls, word: str | None) -> str | None:
        if word is None:
            return None
        word = word.strip().lower()
        if word not in _ANSWER_INDEX:
            raise ValueError(f"'{word}' is not a valid {WORD_LEN}-letter answer word")
        return word


def _load_actor(checkpoint: Path, device: str) -> MLPActor:
    state = torch.load(checkpoint, map_location=device)
    # Older checkpoints saved the policy under "model", newer ones under "actor"
    actor_state = state.get("actor", state.get("model"))
    if actor_state is None:
        raise ValueError(f"{checkpoint} does not contain an actor state dict")
    actor = MLPActor()
    actor.load_state_dict(actor_state)
    actor.to(device)
    actor.eval()
    return actor


def _clear_screen() -> None:
    print("\x1b[2J\x1b[H", end="")


def _render(
    env: WordleEnv, args: CliArgs, step: int, top: list[tuple[str, float]]
) -> None:
    _clear_screen()
    print(f"checkpoint: {args.checkpoint}")
    print(f"guess {min(step, MAX_GUESSES)}/{MAX_GUESSES}")
    print()
    env.print()
    if top:
        print()
        print("model's top candidates:")
        for word, prob in top:
            bar = "█" * int(round(prob * 30))
            print(f"  {word.upper()}  {prob:6.1%}  \x1b[2m{bar}\x1b[0m")
    print()


def main() -> None:
    args = CliApp.run(CliArgs)
    actor = _load_actor(args.checkpoint, args.device)

    if args.word is None:
        env = WordleEnv.create(random.randrange(len(ANSWER_WORDS)))
    else:
        env = WordleEnv.create(_ANSWER_INDEX[args.word])

    _render(env, args, step=0, top=[])
    with torch.no_grad():
        while not env.is_over():
            state = torch.tensor(
                [env.encode_state()], device=args.device, dtype=torch.float
            )
            logits = actor.get_word_logits(state)[0]
            guess = int(logits.argmax().item())

            probs = torch.softmax(logits, dim=0)
            k = min(args.top_k, len(GUESS_WORDS))
            top_probs, top_ids = probs.topk(k)
            top = [
                (GUESS_WORDS[i], p)
                for i, p in zip(top_ids.tolist(), top_probs.tolist(), strict=True)
            ]

            time.sleep(args.delay)
            env.step(guess)
            _render(env, args, step=env.guesses_made(), top=top)

    if env.solved:
        print(f"Solved '{env.target}' in {env.guesses_made()}/{MAX_GUESSES} guesses.")
    else:
        print(f"Failed to solve. The word was '{env.target}'.")


if __name__ == "__main__":
    main()
