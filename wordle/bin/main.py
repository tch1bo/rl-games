from wordle.env import WordleEnv


def main() -> None:
    env = WordleEnv.create(0)
    result = env.step(0)
    print(result)
    env.print()


if __name__ == "__main__":
    main()
