from dataclasses import dataclass
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"

ANSWER_WORDS: tuple[str, ...] = tuple((_DATA_DIR / "answers.txt").read_text().split())
GUESS_WORDS: tuple[str, ...] = tuple((_DATA_DIR / "guesses.txt").read_text().split())

WORD_LEN = 5
MAX_GUESSES = 6
ALPHABET_LEN = 26
STATE_DIM = WORD_LEN * ALPHABET_LEN + 2 * ALPHABET_LEN + 1


@dataclass(slots=True)
class GuessResult:
    green: list[int]
    yellow: list[int]
    gray: list[int]

    def is_win(self) -> bool:
        return len(self.green) == WORD_LEN


@dataclass(slots=True)
class WordleEnv:
    target: str
    history: list[tuple[str, GuessResult]]
    solved: bool

    @staticmethod
    def create(start_word_index: int) -> "WordleEnv":
        if not 0 <= start_word_index < len(ANSWER_WORDS):
            raise ValueError(
                f"invalid start word index {start_word_index}, must be in [0, {len(ANSWER_WORDS)})"
            )
        return WordleEnv(
            target=ANSWER_WORDS[start_word_index], history=[], solved=False
        )

    def step(self, guess_index: int) -> GuessResult:
        if self.is_over():
            raise RuntimeError("the game is already over")
        if not 0 <= guess_index < len(GUESS_WORDS):
            raise ValueError(
                f"invalid guess index {guess_index}, must be in [0, {len(GUESS_WORDS)})"
            )
        guess = GUESS_WORDS[guess_index]
        result = _score_guess(self.target, guess)
        if result.is_win():
            self.solved = True
        self.history.append((guess, result))
        return result

    def is_over(self) -> bool:
        return self.solved or len(self.history) >= MAX_GUESSES

    def guesses_made(self) -> int:
        return len(self.history)

    def encode_state(self) -> list[int]:
        possible = [[1] * ALPHABET_LEN for _ in range(WORD_LEN)]
        min_count = [0] * ALPHABET_LEN
        max_count = [WORD_LEN] * ALPHABET_LEN
        for guess, result in self.history:
            letters = [ord(ch) - ord("a") for ch in guess]
            colored = [0] * ALPHABET_LEN
            for i in result.green + result.yellow:
                colored[letters[i]] += 1
            for i in result.green:
                for c in range(ALPHABET_LEN):
                    if c != letters[i]:
                        possible[i][c] = 0
            for i in result.yellow:
                possible[i][letters[i]] = 0
            has_gray = [False] * ALPHABET_LEN
            for i in result.gray:
                possible[i][letters[i]] = 0
                has_gray[letters[i]] = True
            for c in range(ALPHABET_LEN):
                min_count[c] = max(min_count[c], colored[c])
                if has_gray[c]:
                    max_count[c] = min(max_count[c], colored[c])
        for c in range(ALPHABET_LEN):
            if max_count[c] == 0:
                for i in range(WORD_LEN):
                    possible[i][c] = 0
        state = [value for row in possible for value in row]
        state.extend(min_count)
        state.extend(max_count)
        state.append(MAX_GUESSES - len(self.history))
        return state

    def print(self) -> None:
        for guess, result in self.history:
            colors = ["\x1b[100;97m"] * WORD_LEN
            for i in result.green:
                colors[i] = "\x1b[42;30m"
            for i in result.yellow:
                colors[i] = "\x1b[43;30m"
            print(
                "".join(
                    f"{colors[i]} {guess[i].upper()} \x1b[0m" for i in range(WORD_LEN)
                )
            )
        for _ in range(len(self.history), MAX_GUESSES):
            print("\x1b[2m · \x1b[0m" * WORD_LEN)

    def to_markdown(self) -> str:
        return "\n".join(
            [
                f"# Target: {self.target}, solved={self.solved}",
                "Guesses:",
            ]
            + [
                f" - {guess} (g:{r.green}, y:{r.yellow}, gr: {r.gray})"
                for guess, r in self.history
            ]
        )


def _score_guess(target: str, guess: str) -> GuessResult:
    remaining = [0] * ALPHABET_LEN
    green: list[int] = []
    yellow: list[int] = []
    gray: list[int] = []
    for i in range(WORD_LEN):
        if guess[i] != target[i]:
            remaining[ord(target[i]) - ord("a")] += 1
    for i in range(WORD_LEN):
        if guess[i] == target[i]:
            green.append(i)
            continue
        c = ord(guess[i]) - ord("a")
        if remaining[c] > 0:
            remaining[c] -= 1
            yellow.append(i)
        else:
            gray.append(i)
    return GuessResult(green=green, yellow=yellow, gray=gray)
