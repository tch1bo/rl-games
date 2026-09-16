import random

from wordle.env import ANSWER_WORDS, GUESS_WORDS, MAX_GUESSES, WORD_LEN, WordleEnv

_GUESS_INDEX = {word: index for index, word in enumerate(GUESS_WORDS)}


def main() -> None:
    env = WordleEnv.create(random.randrange(len(ANSWER_WORDS)))
    print(f"Guess the {WORD_LEN}-letter word. You have {MAX_GUESSES} tries.")
    env.print()
    while not env.is_over():
        try:
            line = input("> ")
        except EOFError:
            print()
            return
        word = line.strip().lower()
        if len(word) != WORD_LEN or not (word.isascii() and word.isalpha()):
            print(f"Please enter a {WORD_LEN}-letter word.")
            continue
        index = _GUESS_INDEX.get(word)
        if index is None:
            print(f"'{word}' is not in the word list.")
            continue
        env.step(index)
        env.print()
    if env.solved:
        print(f"You won in {env.guesses_made()}/{MAX_GUESSES} guesses!")
    else:
        print(f"You lost. The word was '{env.target}'.")


if __name__ == "__main__":
    main()
