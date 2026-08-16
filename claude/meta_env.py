"""RL^2 обёртка над MarsRoverVecEnv (среда v11 — бюджет времени на трайл).

Правила среды (см. `Env::reset` / `Env::step` в environment/cpp/src/env.cpp):

* Трайл = фиксированный бюджет `trial_time_limit = 120 c` -> 7200 шагов при
  dt = 1/60. Бюджет тикает НЕПРЕРЫВНО и не сбрасывается при гибели ровера.
* `episodes_per_trial = 0` — попыток внутри трайла сколько угодно: переворот,
  застревание, разряд просто перезапускают ровер с x = 0 в ТОМ ЖЕ мире
  (`trial_mechanic_seed_` сохраняется), а таймер продолжает идти.
* Новый мир (биом, зоны, рельеф) выдаётся только когда бюджет исчерпан:
  `trial_exhausted()` -> truncated, и следующий reset идёт с trial_start = True.

Отсюда постановка: **скрытая задача постоянна внутри трайла, а цена ошибки —
потраченные секунды**. Значит нужно (а) выводить тип мира из опыта и (б) делать
это за минимальное число секунд, потому что каждая неудачная попытка съедает
общий бюджет. Ровно это и оптимизирует RL^2: GRU-память живёт весь трайл и
обнуляется только на trial_start, а PPO максимизирует сумму наград за трайл.

Наблюдение расширяется: prev_action (one-hot), prev_reward, done, доля
оставшегося времени трайла, номер попытки, лучший достигнутый x в трайле.
"""

from __future__ import annotations

import numpy as np

from mars_rover_env import MarsRoverVecEnv
from mars_rover_env.actions import ACTION_MACROS

MAX_ATTEMPT_BUCKETS = 6


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
        self.raw_obs_dim = self.native.obs_dim
        # бюджет трайла в шагах — повторяем Env::trial_step_budget()
        self.trial_budget = int(self.native.core.trial_step_budget(0))
        if self.trial_budget <= 0:
            raise RuntimeError("В конфиге не задан trial_time_limit — среда не v11")
        # obs + one-hot(prev action) + prev_reward + done + time_left + attempt one-hot + best_x
        self.obs_dim = self.raw_obs_dim + self.num_actions + 3 + MAX_ATTEMPT_BUCKETS + 1
        self._rng = np.random.default_rng(seed)
        self._base_seed = int(seed)

        n = self.num_envs
        self._prev_action = np.zeros(n, dtype=np.int64)
        self._prev_reward = np.zeros(n, dtype=np.float32)
        self._prev_done = np.zeros(n, dtype=np.float32)
        self._steps_used = np.zeros(n, dtype=np.int64)
        self._attempt = np.zeros(n, dtype=np.int64)
        self._best_x = np.zeros(n, dtype=np.float32)
        # статистика попыток / трайлов
        self.ep_return = np.zeros(n, dtype=np.float64)
        self.ep_len = np.zeros(n, dtype=np.int64)
        self._trial_return = np.zeros(n, dtype=np.float64)
        self._trial_best = np.zeros(n, dtype=np.float32)
        self._trial_sum_progress = np.zeros(n, dtype=np.float64)

    def _augment(self, raw: np.ndarray) -> np.ndarray:
        n = self.num_envs
        rows = np.arange(n)
        out = np.zeros((n, self.obs_dim), dtype=np.float32)
        out[:, : self.raw_obs_dim] = raw
        k = self.raw_obs_dim
        out[rows, k + self._prev_action] = 1.0
        k += self.num_actions
        out[:, k] = np.clip(self._prev_reward / 10.0, -5.0, 5.0)
        out[:, k + 1] = self._prev_done
        out[:, k + 2] = 1.0 - self._steps_used / self.trial_budget  # сколько времени осталось
        k += 3
        bucket = np.minimum(self._attempt, MAX_ATTEMPT_BUCKETS - 1)
        out[rows, k + bucket] = 1.0
        k += MAX_ATTEMPT_BUCKETS
        out[:, k] = self._best_x
        return out

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        raw = self.native.reset(self._base_seed).copy()
        for arr in (
            self._prev_action,
            self._prev_reward,
            self._prev_done,
            self._steps_used,
            self._attempt,
            self._best_x,
            self.ep_return,
            self.ep_len,
            self._trial_return,
            self._trial_best,
            self._trial_sum_progress,
        ):
            arr[:] = 0
        return self._augment(raw), np.ones(self.num_envs, dtype=bool)

    def step(self, actions: np.ndarray):
        """(obs, reward, done, trial_start, infos).

        `trial_start[i]` = True -> obs[i] уже из НОВОГО мира, память сбросить.
        """
        actions = np.asarray(actions, dtype=np.int64)
        raw, reward, term, trunc, _ = self.native.step(self.macros[actions])
        reward = reward.copy()
        done = np.logical_or(term, trunc)
        self._steps_used += 1
        self.ep_return += reward
        self.ep_len += 1
        self._trial_return += reward
        progress = raw[:, 0].copy()  # x / finish_x
        np.maximum(self._best_x, progress, out=self._best_x)

        infos = []
        trial_start = np.zeros(self.num_envs, dtype=bool)
        for i in np.flatnonzero(done):
            exhausted = self._steps_used[i] >= self.trial_budget
            self._trial_best[i] = self._best_x[i]
            self._trial_sum_progress[i] += progress[i]
            info = {
                "env": int(i),
                "attempt": int(self._attempt[i]),
                "return": float(self.ep_return[i]),
                "length": int(self.ep_len[i]),
                "progress": float(progress[i]),
                # метров в секунду за попытку: честная мера качества езды,
                # не искажённая обрывом последней попытки по концу бюджета
                "speed_mps": float(progress[i] * 800.0 / max(1e-6, self.ep_len[i] / 60.0)),
                "success": bool(term[i] and progress[i] >= 0.999),
                "trial_end": bool(exhausted),
            }
            if exhausted:
                info["trial"] = {
                    "return": float(self._trial_return[i]),
                    "best_progress": float(self._trial_best[i]),
                    "sum_progress": float(self._trial_sum_progress[i]),
                    "attempts": int(self._attempt[i]) + 1,
                }
            infos.append(info)

            seed = int(self._rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            self.native.reset_at(int(i), seed, trial_start=bool(exhausted))
            trial_start[i] = exhausted
            if exhausted:
                self._steps_used[i] = 0
                self._attempt[i] = 0
                self._best_x[i] = 0.0
                self._trial_return[i] = 0.0
                self._trial_sum_progress[i] = 0.0
            else:
                self._attempt[i] += 1
            self.ep_return[i] = 0.0
            self.ep_len[i] = 0

        self._prev_action[:] = actions
        self._prev_reward[:] = reward
        self._prev_done[:] = done.astype(np.float32)
        return self._augment(raw), reward, done, trial_start, infos
