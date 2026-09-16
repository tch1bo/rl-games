from wordle.env import (
    ALPHABET_LEN,
    ANSWER_WORDS,
    GUESS_WORDS,
    MAX_GUESSES,
    STATE_DIM,
    WORD_LEN,
    WordleEnv,
)


def _state_after(target: str, guess: str) -> list[int]:
    env = WordleEnv.create(ANSWER_WORDS.index(target))
    env.step(GUESS_WORDS.index(guess))
    return env.encode_state()


def test_gray_duplicate_sets_exact_count() -> None:
    state = _state_after("abate", "speed")
    e = ord("e") - ord("a")
    s = ord("s") - ord("a")
    min_count = state[WORD_LEN * ALPHABET_LEN : WORD_LEN * ALPHABET_LEN + ALPHABET_LEN]
    max_count = state[WORD_LEN * ALPHABET_LEN + ALPHABET_LEN : STATE_DIM - 1]
    assert min_count[e] == 1
    assert max_count[e] == 1
    assert min_count[s] == 0
    assert max_count[s] == 0
    assert state[2 * ALPHABET_LEN + e] == 0
    assert state[3 * ALPHABET_LEN + e] == 0
    assert state[4 * ALPHABET_LEN + e] == 1
    assert state[s] == 0
    assert state[ALPHABET_LEN + s] == 0
    assert state[STATE_DIM - 1] == MAX_GUESSES - 1
    assert len(state) == STATE_DIM


def test_green_makes_column_one_hot() -> None:
    state = _state_after("abate", "aback")
    for c in range(ALPHABET_LEN):
        assert state[c] == int(c == 0)
        assert state[ALPHABET_LEN + c] == int(c == 1)
