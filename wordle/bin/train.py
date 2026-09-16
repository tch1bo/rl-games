from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
import tqdm
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp
from torch.distributions.categorical import Categorical
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from common.utils import get_logger, save_checkpoint
from wordle.env import ANSWER_WORDS, MAX_GUESSES, WordleEnv
from wordle.models import MLPActor, MLPCritic

logger = get_logger()

# TODO(chibo) - ideas
#   1. compare REINFORCE vs REINFORCE-with-baseline
#   2. curriculum learning - remember hard states, focus on these at some point


class CliArgs(BaseSettings):
    num_steps: int = Field(
        default=10000, description="number of gradient update steps to perform"
    )
    games_per_step: int = Field(
        default=512,
        description="the number of games (seeds) to play per step. Every game is played from the start to the end",
    )
    max_lr: float = 1e-5
    out_dir: Path = Field(
        default_factory=lambda: Path(
            f"out/wordle_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    )
    lr_warmup_steps: int = 100
    seed: int = 42
    save_every: int = 10000
    max_checkpoints: int = 10

    device: Literal["cpu", "cuda"] = "cuda"
    algorithm: Literal["reinforce", "reinforce-baseline"] = "reinforce"


def _make_optimizer_and_scheduler(
    args: CliArgs, model: torch.nn.Module
) -> tuple[Optimizer, LambdaLR]:
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.max_lr, fused=True)

    def lr_lambda(step: int):
        return min(1.0, (step + 1) / args.lr_warmup_steps)

    return (optimizer, LambdaLR(optimizer, lr_lambda))


def _reward(env: WordleEnv, step: int) -> float:
    if step >= len(env.history):
        raise ValueError(
            f"requested reward for step {step} for an env with {len(env.history)} steps"
        )
    step_rewards = [1 if h[1].is_win() else -1 for h in env.history]
    # Add a big boost if the game was solved
    return sum(step_rewards[step:]) + (10 if env.solved else 0)


def _sample_trajectories(envs: list[WordleEnv]) -> list[str]:
    random_words: list[int] = [
        1722,  # shelf
        1048,  # ivory
        1073,  # khaki
        43,  # ahead
        839,  # gaunt
        407,  # clerk
        1006,  # hunky
        1464,  # press
        408,  # click
        1553,  # realm
    ]

    return [envs[idx].to_markdown() for idx in random_words]


def main() -> None:
    args = CliApp.run(CliArgs)
    torch.manual_seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(args.model_dump_json(indent=2))
    writer = SummaryWriter(log_dir=args.out_dir)

    actor = MLPActor().to(args.device)
    actor_optimizer, actor_lr_scheduler = _make_optimizer_and_scheduler(args, actor)

    use_critic = args.algorithm == "reinforce-baseline"
    if use_critic:
        critic = MLPCritic().to(args.device)
        critic_optimizer, critic_lr_scheduler = _make_optimizer_and_scheduler(
            args, critic
        )
    else:
        critic, critic_optimizer, critic_lr_scheduler = None, None, None

    logger.info("starting training", log_dir=str(args.out_dir.absolute()))
    for step in tqdm.trange(args.num_steps, desc="training"):
        start_word_ids: list[int] = torch.randperm(len(ANSWER_WORDS)).tolist()[
            : args.games_per_step
        ]
        envs: dict[str, WordleEnv] = {
            env.target: env for swi in start_word_ids if (env := WordleEnv.create(swi))
        }
        # the log_probs for every step of every env
        step_log_probs: list[torch.Tensor] = []
        # the state values for every step of every env
        state_values: list[torch.Tensor] = []
        env_targets: list[list[str]] = []
        num_different_starting_words: int = 0
        entropies: list[torch.Tensor] = []

        for guess in range(MAX_GUESSES):
            active_envs = [env for env in envs.values() if not env.is_over()]
            if not active_envs:
                break
            states = torch.tensor(
                [env.encode_state() for env in active_envs],
                device=args.device,
                dtype=torch.float,
            )

            # Sample the actor to get the next word for each env
            # `word_logits` is (games_per_step, len(GUESS_WORDS))
            word_logits = actor.get_word_logits(states)
            # from https://docs.pytorch.org/docs/2.14/distributions.html#score-function
            distr = Categorical(logits=word_logits)
            words = distr.sample()
            entropies.append(distr.entropy().detach())

            # Get the state values for each env
            if use_critic:
                assert critic is not None
                state_values.append(critic(states))

            # Step the environments
            words_list = words.tolist()
            for env, word in zip(active_envs, words_list, strict=True):
                env.step(word)
            if guess == 0:
                num_different_starting_words = len(set(words_list))

            step_log_probs.append(distr.log_prob(words))

            # Record which environments were used at this step
            env_targets.append([env.target for env in active_envs])

        # Compute the rewards for every step of every env
        rewards = torch.cat(
            [
                torch.tensor(
                    [_reward(envs[target], step) for target in targets],
                    dtype=torch.float,
                    device=args.device,
                )
                for step, targets in enumerate(env_targets)
            ]
        )

        # If using a state function for a baseline, update the rewards and backprop the value func
        if use_critic:
            assert state_values
            sv = torch.cat(state_values).reshape(-1)
            critic_loss = torch.nn.functional.mse_loss(sv, rewards, reduction="mean")

            assert critic_optimizer is not None
            assert critic_lr_scheduler is not None

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            critic_lr_scheduler.step()

            writer.add_scalar("train/critic_loss", float(critic_loss.item()), step)
            rewards = rewards - sv.detach()

        # Compute and normalize the loss (the loss is negated, because we're doing gradient descent)
        log_probs = torch.cat(step_log_probs)
        loss = -(log_probs * rewards).mean()

        # Backprop the actor
        actor_optimizer.zero_grad()
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), max_norm=float("inf")
            ).item()
        )
        actor_optimizer.step()
        actor_lr_scheduler.step()

        # Log stuff
        entropy = torch.cat(entropies).mean()
        writer.add_scalar("train/actor_grad_norm", grad_norm, step)
        writer.add_scalar("train/actor_loss", float(loss.item()), step)
        writer.add_scalar(
            "train/ratio_solved_envs",
            sum(int(env.solved) for env in envs.values()) / len(envs),
            step,
        )
        writer.add_scalar("train/mean_num_moves", log_probs.shape[0] / len(envs), step)
        writer.add_scalar(
            "train/num_different_starting_words", num_different_starting_words, step
        )
        writer.add_scalar("train/entropy", entropy, step)

        if step % args.save_every == 0:
            # Save the checkpoint
            state_dict = {
                "step": step,
                "actor": actor.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "actor_scheduler": actor_lr_scheduler.state_dict(),
            }
            if use_critic:
                assert critic is not None
                assert critic_optimizer is not None
                assert critic_lr_scheduler is not None
                state_dict["critic"] = critic.state_dict()
                state_dict["critic_optimizer"] = critic_optimizer.state_dict()
                state_dict["critic_lr_scheduler"] = critic_lr_scheduler.state_dict()
            save_checkpoint(
                args.out_dir,
                state_dict,
                step=step,
                max_checkpoints=args.max_checkpoints,
            )

            # Run evals
            eval_envs = [WordleEnv.create(swi) for swi in range(0, len(ANSWER_WORDS))]
            with torch.no_grad():
                for guess in range(MAX_GUESSES):
                    active_envs = [env for env in eval_envs if not env.is_over()]
                    if not active_envs:
                        break
                    states = torch.tensor(
                        [env.encode_state() for env in active_envs],
                        device=args.device,
                        dtype=torch.float,
                    )
                    # `word_logits` is (games_per_step, len(GUESS_WORDS))
                    word_logits = actor.get_word_logits(states)
                    words = word_logits.argmax(dim=1)
                    for env, word in zip(active_envs, words.tolist(), strict=True):
                        env.step(word)

            ratio_solved = sum(int(env.solved) for env in eval_envs) / len(eval_envs)
            mean_moves = sum(len(env.history) for env in eval_envs) / len(eval_envs)
            logger.info(
                "eval done", step=step, ratio_solved=ratio_solved, mean_moves=mean_moves
            )
            writer.add_scalar("eval/ratio_solved", ratio_solved, step)
            writer.add_scalar("eval/mean_moves", mean_moves, step)

            for i, md in enumerate(_sample_trajectories(eval_envs)):
                writer.add_text(f"eval/trajectory_{i}", md, step)


if __name__ == "__main__":
    main()
