# -*- encoding: "utf-8" -*-

# import warnings
import os
import time
from dataclasses import dataclass
from datetime import datetime
import gymnasium as gym
import gym_anytrading  # noqa, for "stock-v0"
from data_preprocessing import DataPreprocessing
from stable_baselines3 import PPO

# warnings.filterwarnings("ignore")


@dataclass
class RLStrategy:

    rl_model: object = PPO
    policy: str = "MlpPolicy"
    env_id: str = "stocks-v0"
    timestamp: datetime = datetime.now().strftime("%Y%m%d")  # for record
    RLTIMESTAMPS: int = 50000
    window_size: int = 60
    episodes: int = 500

    def __post_init__(self):

        self.collect_data()
        self._set_dirs()

    def _print(self, str) -> None:

        print()
        print("#", "-" * (len(str)), "#")
        print(f"# {str} #")
        print("#", "-" * (len(str)), "#", "\n")

    def _dec(func):

        def wrap(self, *args, **kwargs):
            begin = time.perf_counter()
            # self._print(f"Processing: {func.__name__}")
            res = func(self, *args, **kwargs)
            self._print(
                f"{func.__name__} finished in "
                + f"{round(time.perf_counter() - begin, 4)}s")

            return res

        return wrap

    def collect_data(
            self,
            start: datetime = datetime(2000, 1, 1),
            end: datetime = datetime(2024, 1, 1),
            period: str = "60d",
            interval: str = "2m"
    ):

        symbol = "AAPL"
        cla = DataPreprocessing(symbol)
        df = cla.download_historical_data(start, end)
        self.df = cla.run(df)
        print(f"Data set size: {df.shape}")

    def _set_dirs(self):

        self.model_dir = rf"./models/{self.rl_model.__name__}"
        self.log_dir = rf"./logs/{self.rl_model.__name__}"

        os.makedirs(rf"{self.log_dir}", exist_ok=True)
        os.makedirs(rf"{self.model_dir}", exist_ok=True)

    def train_again(
            self,
            last_steps: int
    ):

        # def env_maker():
        #     return gym.make(
        #         self.env_id,
        #         df=self.df,
        #         frame_bound=(60, self.df.shape[0]),
        #         window_size=self.window_size,
        #     )

        # env = DummyVecEnv([env_maker])

        env = gym.make(
            self.env_id,
            df=self.df,
            frame_bound=(60, self.df.shape[0]),
            window_size=self.window_size,
        )
        model = self.rl_model.load(
            rf"./{self.model_dir}/{last_steps}.zip",
            env=env,
        )
        for i in range(self.episodes):
            tb_log_name = \
                f"{self.model_name(self.rl_model)}" \
                + f"-{self.RLTIMESTAMPS*i + last_steps}"
            model.learn(
                total_timesteps=self.RLTIMESTAMPS,
                reset_num_timesteps=False,
                tb_log_name=tb_log_name
            )
            model.save(rf"{self.model_dir}/{self.RLTIMESTAMPS*i + last_steps}")

        return model

    def train(self):

        env = gym.make(
            self.env_id,
            df=self.df,
            frame_bound=(60, self.df.shape[0]),
            window_size=self.window_size,
        )
        model = self.rl_model(
            self.policy, env, verbose=1, tensorboard_log=self.log_dir
        )
        for i in range(self.episodes):
            model.learn(
                total_timesteps=self.RLTIMESTAMPS,
                reset_num_timesteps=False,
                tb_log_name=f"{self.rl_model.__name__}-{self.RLTIMESTAMPS*i}",
            )
            model.save(rf"{self.model_dir}/{self.RLTIMESTAMPS*i}")

        return model


if __name__ == "__main__":

    rl = RLStrategy()
    model = rl.train()  # best: 43520000; worst: 20000
