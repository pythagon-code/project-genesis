# Othello Policy-Gradient Self‑Play (with Target Critic)
# -----------------------------------------------------
# Features:
# - Uses your Othello game logic (minimally adapted) as the environment
# - Actor-Critic with masked policy over legal moves
# - Self-play: the agent plays both sides; observations are from the current player's perspective
# - Target value network for stable updates (soft τ-update) -> NOW HARD UPDATE
# - Entropy regularization with decay for exploration
# - GAE (Generalized Advantage Estimation) for more stable policy updates
# - Optional data augmentation via board symmetries
# - A more robust neural network architecture
# - PyTorch implementation, single-file
# - ADDED: Hard update of target critic
# - ADDED: Epsilon-greedy exploration strategy
# -----------------------------------------------------

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# =====================
# Othello game logic (from user code, lightly wrapped)
# =====================

empty = 2
white = 0
black = 1

initial_pos = [[empty] * 8 for _ in range(8)]
initial_pos[3][3] = initial_pos[4][4] = white
initial_pos[3][4] = initial_pos[4][3] = black

piece_to_emoji = {empty: "🔲", black: "⚫", white: "⚪"}


def initializeBoard():
    return np.array(initial_pos, dtype=np.int64)


def isTrapped(board, turn, r, c, r_incr, c_incr):
    r += r_incr
    c += c_incr
    if not (0 <= r < 8 and 0 <= c < 8) or board[r, c] != 1 - turn:
        return False
    r += r_incr
    c += c_incr
    while 0 <= r < 8 and 0 <= c < 8:
        if board[r, c] == empty:
            return False
        elif board[r, c] == turn:
            return True
        r += r_incr
        c += c_incr
    return False


def generateMoves(board, turn):
    moves = []
    full = True
    for r in range(8):
        for c in range(8):
            if board[r, c] == empty:
                if (
                        isTrapped(board, turn, r, c, 1, 0)
                        or isTrapped(board, turn, r, c, -1, 0)
                        or isTrapped(board, turn, r, c, 0, 1)
                        or isTrapped(board, turn, r, c, 0, -1)
                        or isTrapped(board, turn, r, c, 1, 1)
                        or isTrapped(board, turn, r, c, 1, -1)
                        or isTrapped(board, turn, r, c, -1, 1)
                        or isTrapped(board, turn, r, c, -1, -1)
                ):
                    moves.append((r, c))
                full = False
    return None if full else moves


def flipPieces(board, turn, r, c, r_incr, c_incr):
    if isTrapped(board, turn, r, c, r_incr, c_incr):
        r += r_incr
        c += c_incr
        while 0 <= r < 8 and 0 <= c < 8:
            if board[r, c] == 1 - turn:
                board[r, c] = turn
                r += r_incr
                c += c_incr
            else:
                break


def playMove(board, turn, move):
    r, c = move
    board[r, c] = turn
    flipPieces(board, turn, r, c, 1, 0)
    flipPieces(board, turn, r, c, -1, 0)
    flipPieces(board, turn, r, c, 0, 1)
    flipPieces(board, turn, r, c, 0, -1)
    flipPieces(board, turn, r, c, 1, 1)
    flipPieces(board, turn, r, c, 1, -1)
    flipPieces(board, turn, r, c, -1, 1)
    flipPieces(board, turn, r, c, -1, -1)


def getResult(board):
    return int(np.sum(board == white) - np.sum(board == black))


# =====================
# Environment wrapper
# =====================

@dataclass
class Step:
    obs: torch.Tensor
    legal_mask: torch.Tensor
    action: int
    logp: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool
    player: int


class OthelloEnv:
    def __init__(self, augment: bool = True):
        self.augment = augment
        self.reset()

    @staticmethod
    def _obs_from_board(board: np.ndarray, player: int) -> np.ndarray:
        me = (board == player).astype(np.float32)
        opp = (board == (1 - player)).astype(np.float32)
        emp = (board == empty).astype(np.float32)
        obs = np.stack([me, opp, emp], axis=0)
        return obs

    @staticmethod
    def _legal_mask(board: np.ndarray, player: int) -> np.ndarray:
        moves = generateMoves(board, player)
        mask = np.zeros(64, dtype=np.float32)
        if moves is None:
            return mask
        for (r, c) in moves:
            mask[r * 8 + c] = 1.0
        return mask

    def reset(self):
        self.board = initializeBoard()
        self.player = black
        self.done = False
        return self._get_obs()

    def _get_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        obs = self._obs_from_board(self.board, self.player)
        mask = self._legal_mask(self.board, self.player)
        if self.augment:
            k = random.randint(0, 3)
            obs = np.rot90(obs, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask.reshape(8, 8), k=k).reshape(-1)
            if random.random() < 0.5:
                obs = np.flip(obs, axis=1).copy()
                mask = np.flip(mask.reshape(8, 8), axis=0).reshape(-1)
            if random.random() < 0.5:
                obs = np.flip(obs, axis=2).copy()
                mask = np.flip(mask.reshape(8, 8), axis=1).reshape(-1)
        return obs, mask

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Step called on finished game")
        r, c = divmod(int(action), 8)
        moves = generateMoves(self.board, self.player)
        if moves is None:
            self.done = True
            result = getResult(self.board)
            return (self._obs_from_board(self.board, self.player),
                    self._legal_mask(self.board, self.player), 0.0, True, result)
        if (r, c) not in moves:
            self.done = True
            result = -64
            return (self._obs_from_board(self.board, self.player),
                    self._legal_mask(self.board, self.player), -1.0, True, result)

        playMove(self.board, self.player, (r, c))
        self.player = 1 - self.player

        for _ in range(2):
            moves = generateMoves(self.board, self.player)
            if moves is None:
                self.done = True
                result = getResult(self.board)
                return (self._obs_from_board(self.board, 1 - self.player),
                        self._legal_mask(self.board, 1 - self.player), 0.0, True, result)
            if len(moves) == 0:
                self.player = 1 - self.player
            else:
                break
        return (*self._get_obs(), 0.0, False, None)


# =====================
# Model: Actor-Critic with Residual Blocks
# =====================
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(3)]  # Using 3 residual blocks
        )
        self.shared_head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )
        self.head_fc = nn.Linear(128 * 8 * 8, 512)

        self.policy_head = nn.Linear(512, 64)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.initial_block(x)
        x = self.res_blocks(x)
        x = self.shared_head(x)
        x = F.relu(self.head_fc(x))

        logits = self.policy_head(x)
        v = self.value_head(x).squeeze(-1)
        return logits, v


# =====================
# Training utilities
# =====================

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2.5e-4
    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_episodes: int = 32
    total_episodes: int = 16000
    target_update_freq: int = 500  # ADDED: Target network hard update frequency
    epsilon_start: float = 0.5  # ADDED: Epsilon-greedy exploration start
    epsilon_end: float = 0.01  # ADDED: Epsilon-greedy exploration end
    epsilon_decay_episodes: int = 10000  # ADDED: Epsilon decay period
    seed: int = 43
    augment: bool = True


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Tuple[
    torch.distributions.Categorical, torch.Tensor]:
    VERY_LOW = -1e9
    masked_logits = torch.where(mask > 0.0, logits, torch.full_like(logits, VERY_LOW))
    probs = F.softmax(masked_logits, dim=-1)
    dist = torch.distributions.Categorical(probs=probs)
    return dist, probs


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.env = OthelloEnv(augment=cfg.augment)
        self.actor_critic = ActorCritic().to(cfg.device)
        self.critic_target = ActorCritic().to(cfg.device)
        self.critic_target.load_state_dict(self.actor_critic.state_dict())
        self.optim = optim.Adam(self.actor_critic.parameters(), lr=cfg.lr)
        self.total_steps_in_training = 0
        self.episodes_run = 0

    def run_episode(self) -> Tuple[List[Step], torch.Tensor, torch.Tensor]:
        # Calculate current epsilon
        epsilon = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
                  math.exp(-1.0 * self.episodes_run / self.cfg.epsilon_decay_episodes)

        obs_np, mask_np = self.env.reset()
        steps: List[Step] = []
        done = False
        terminal_info = None

        while not done:
            player_id = self.env.player
            obs = torch.tensor(obs_np, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
            mask = torch.tensor(mask_np.copy(), dtype=torch.float32, device=self.cfg.device).unsqueeze(0)

            with torch.no_grad():
                logits, value = self.actor_critic(obs)
                dist, _ = masked_categorical(logits, mask)

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                # Epsilon-random exploration
                legal_moves_indices = torch.nonzero(mask).squeeze().cpu().numpy()
                action = int(np.random.choice(legal_moves_indices))
            else:
                # Policy-based action (sample from distribution)
                action = int(dist.sample().item())

            logp = dist.log_prob(torch.tensor(action, device=self.cfg.device).unsqueeze(0))

            next_obs_np, next_mask_np, reward, done, terminal_info = self.env.step(action)

            steps.append(
                Step(
                    obs=obs.squeeze(0).detach(),
                    legal_mask=mask.squeeze(0).detach(),
                    action=action,
                    logp=logp.squeeze(0).detach(),
                    value=value.squeeze(0).detach(),
                    reward=float(reward),
                    done=done,
                    player=player_id,
                )
            )
            obs_np, mask_np = next_obs_np, next_mask_np

        self.episodes_run += 1

        if terminal_info == -64:
            last_player = steps[-1].player
            final_rewards = {last_player: -1.0, 1 - last_player: 1.0}
        else:
            result = getResult(self.env.board)
            if result > 0:
                final_rewards = {white: 1.0, black: -1.0}
            elif result < 0:
                final_rewards = {black: 1.0, white: -1.0}
            else:
                final_rewards = {black: 0.0, white: 0.0}

        for step in steps:
            step.reward = final_rewards[step.player]

        returns: List[float] = []
        advantages: List[float] = []
        with torch.no_grad():
            gae = 0.0
            last_v = 0.0
            for t in reversed(range(len(steps))):
                r = steps[t].reward

                if t < len(steps) - 1:
                    next_obs = steps[t + 1].obs.unsqueeze(0).to(self.cfg.device)
                    _, next_v = self.critic_target(next_obs)
                    next_v = next_v.item()
                else:
                    next_v = 0.0

                td_delta = r + self.cfg.gamma * next_v - steps[t].value.item()
                gae = td_delta + self.cfg.gamma * self.cfg.gae_lambda * gae
                advantages.insert(0, gae)

                G = r + self.cfg.gamma * last_v
                returns.insert(0, G)
                last_v = G

        return steps, torch.tensor(returns, dtype=torch.float32, device=self.cfg.device), torch.tensor(advantages,
                                                                                                       dtype=torch.float32,
                                                                                                       device=self.cfg.device)

    def train(self):
        cfg = self.cfg
        ep = 0
        while ep < cfg.total_episodes:
            all_obs = []
            all_masks = []
            all_actions = []
            batch_returns = []
            batch_advs = []

            for _ in range(cfg.batch_episodes):
                steps, returns, advs = self.run_episode()
                ep += 1
                for s, R, A in zip(steps, returns, advs):
                    all_obs.append(s.obs)
                    all_masks.append(s.legal_mask)
                    all_actions.append(s.action)
                    batch_returns.append(R)
                    batch_advs.append(A)
                    self.total_steps_in_training += 1

            # Hard update of target critic
            if self.total_steps_in_training % cfg.target_update_freq == 0:
                self.critic_target.load_state_dict(self.actor_critic.state_dict())
                print(f"--- Hard update of target network at step {self.total_steps_in_training} ---")

            # Decay entropy coefficient linearly
            current_entropy_coef = cfg.entropy_coef_start - (cfg.entropy_coef_start - cfg.entropy_coef_end) * (
                        ep / cfg.total_episodes)

            obs = torch.stack(all_obs, dim=0).to(cfg.device)
            masks = torch.stack(all_masks, dim=0).to(cfg.device)
            actions = torch.tensor(all_actions, dtype=torch.int64, device=cfg.device)
            returns_t = torch.stack(batch_returns).to(cfg.device)
            advs_t = torch.stack(batch_advs).to(cfg.device)
            advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

            logits, values = self.actor_critic(obs)
            dist, _ = masked_categorical(logits, masks)
            logps = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            policy_loss = -(logps * advs_t.detach()).mean()
            value_loss = F.mse_loss(values, returns_t)
            loss = policy_loss + cfg.value_coef * value_loss - current_entropy_coef * entropy

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), cfg.max_grad_norm)
            self.optim.step()

            if ep % (cfg.batch_episodes * 2) == 0:
                epsilon = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * \
                          math.exp(-1.0 * ep / self.cfg.epsilon_decay_episodes)
                print(
                    f"Episode {ep:5d} | loss={loss.item():.4f} | π_loss={policy_loss.item():.4f} | V_loss={value_loss.item():.4f} | H={entropy.item():.4f} | ε={epsilon:.4f}")

        torch.save(self.actor_critic.state_dict(), "othello_actor_critic.pt")
        print("Saved model to othello_actor_critic.pt")


# =====================
# Simple evaluation vs. random
# =====================

def eval_vs_random(n_games: int = 100, seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    cfg = Config()
    env = OthelloEnv(augment=False)
    net = ActorCritic()
    net.load_state_dict(torch.load("othello_actor_critic.pt", map_location="cpu"))
    net.eval()

    def pick_action(obs_np, mask_np):
        with torch.no_grad():
            obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)
            logits, _ = net(obs)
            dist, _ = masked_categorical(logits, mask)
            return int(dist.probs.argmax(dim=-1).item())

    wins = ties = losses = 0
    for _ in range(n_games):
        obs_np, mask_np = env.reset()
        done = False
        agent_plays_black = random.random() < 0.5
        while not done:
            current_player = env.player
            if (current_player == black and agent_plays_black) or \
                    (current_player == white and not agent_plays_black):
                action = pick_action(obs_np, mask_np)
            else:
                legal_idx = np.flatnonzero(mask_np > 0)
                if len(legal_idx) == 0:
                    action = -1
                else:
                    action = int(np.random.choice(legal_idx))

            obs_np, mask_np, reward, done, info = env.step(action)

        res = getResult(env.board)
        agent_color = black if agent_plays_black else white
        if res == 0:
            ties += 1
        elif (res > 0 and agent_color == white) or (res < 0 and agent_color == black):
            wins += 1
        else:
            losses += 1

    print(f"Eval vs Random over {n_games} games: W {wins} / T {ties} / L {losses}")


# =====================
# Entry
# =====================

if __name__ == "__main__":
    cfg = Config()
    print(cfg)
    trainer = Trainer(cfg)
    trainer.train()
    eval_vs_random(100)