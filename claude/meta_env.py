"""RL^2 обёртка над MarsRoverVecEnv.

Ключевое наблюдение по среде: биом/механики/сид рельефа фиксируются на весь
*trial* (episodes_per_trial = 4) и меняются только при trial_start. То есть это
классическая meta-RL постановка: скрытая задача z ~ p(z) постоянна внутри трайла,
и агент должен вывести её из опыта первых эпизодов и адаптироваться.

Поэтому обучаем рекуррентную политику (RL^2, Duan et al. 2016):
  - скрытое состояние GRU НЕ обнуляется на границе эпизодов внутри трайла,
    обнуляется только на trial_start;
  - наблюдение расширяется историей: предыдущее действие (one-hot), предыдущая
    награда, флаг done, номер эпизода в трайле.
Так градиент PPO оптимизирует суммарную награду ЗА ТРАЙЛ, что напрямую
поощряет исследование в первом эпизоде ради быстрой адаптации в следующих.
"""

from __future__ import annotations

import numpy as np

from mars_rover_env import MarsRoverVecEnv
from mars_rover_env.actions import ACTION_MACROS


class MetaVecEnv:
    def __init__(
        self,
        num_envs: int,
        *,
        seed: int = 0,
        biome_split: int = 1,
        config_path: str | None = None,
        macros: tuple[int, ...] = ACTION_MACROS,
    ) -> None:
        self.native = MarsRoverVecEnv(num_envs, config_path=config_path, biome_split=biome_split)
        self.macros = np.asarray(macros, dtype=np.int32)
        self.num_envs = self.native.num_envs
        self.num_actions = len(self.macros)
        self.episodes_per_trial = self.native.episodes_per_trial
        self.raw_obs_dim = self.native.obs_dim
        # obs + one-hot(prev action) + prev reward + done + one-hot(episode in trial)
        self.obs_dim = self.raw_obs_dim + self.num_actions + 2 + self.episodes_per_trial
        self._rng = np.random.default_rng(seed)
        self._base_seed = int(seed)

        self._prev_action = np.zeros(self.num_envs, dtype=np.int64)
        self._prev_reward = np.zeros(self.num_envs, dtype=np.float32)
        self._prev_done = np.zeros(self.num_envs, dtype=np.float32)
        self._ep_idx = np.zeros(self.num_envs, dtype=np.int64)
        # диагностика
        self.ep_return = np.zeros(self.num_envs, dtype=np.float64)
        self.ep_len = np.zeros(self.num_envs, dtype=np.int64)

    def _augment(self, raw: np.ndarray) -> np.ndarray:
        n = self.num_envs
        out = np.zeros((n, self.obs_dim), dtype=np.float32)
        out[:, : self.raw_obs_dim] = raw
        k = self.raw_obs_dim
        out[np.arange(n), k + self._prev_action] = 1.0
        k += self.num_actions
        out[:, k] = np.clip(self._prev_reward / 10.0, -5.0, 5.0)
        out[:, k + 1] = self._prev_done
        k += 2
        out[np.arange(n), k + self._ep_idx] = 1.0
        return out

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        raw = self.native.reset(self._base_seed).copy()
        self._prev_action[:] = 0
        self._prev_reward[:] = 0.0
        self._prev_done[:] = 0.0
        self._ep_idx[:] = 0
        self.ep_return[:] = 0.0
        self.ep_len[:] = 0
        trial_start = np.ones(self.num_envs, dtype=bool)
        return self._augment(raw), trial_start

    def step(self, actions: np.ndarray):
        """Возвращает (obs, reward, done, trial_start, infos).

        trial_start[i]=True означает, что НОВОЕ наблюдение obs[i] относится к
        новому трайлу -> скрытое состояние нужно обнулить.
        """
        actions = np.asarray(actions, dtype=np.int64)
        raw, reward, term, trunc, _ = self.native.step(self.macros[actions])
        reward = reward.copy()
        done = np.logical_or(term, trunc)
        self.ep_return += reward
        self.ep_len += 1

        infos = []
        trial_start = np.zeros(self.num_envs, dtype=bool)
        for i in np.flatnonzero(done):
            ts = bool(self.native.next_trial_start[i])
            infos.append(
                {
                    "env": int(i),
                    "return": float(self.ep_return[i]),
                    "length": int(self.ep_len[i]),
                    "episode_in_trial": int(self._ep_idx[i]),
                    "progress": float(raw[i, 0]),
                    "success": bool(term[i] and raw[i, 0] >= 0.999),
                }
            )
            seed = int(self._rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            self.native.reset_at(int(i), seed, trial_start=ts)
            trial_start[i] = ts
            self._ep_idx[i] = 0 if ts else self._ep_idx[i] + 1
            self._ep_idx[i] = min(self._ep_idx[i], self.episodes_per_trial - 1)
            self.ep_return[i] = 0.0
            self.ep_len[i] = 0

        self._prev_action[:] = actions
        self._prev_reward[:] = reward
        self._prev_done[:] = done.astype(np.float32)
        return self._augment(raw), reward, done, trial_start, infos
