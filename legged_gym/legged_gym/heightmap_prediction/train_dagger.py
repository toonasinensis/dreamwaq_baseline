"""Trains student policies using DAgger."""
from absl import app
from absl import flags

from datetime import datetime
from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import os
from rsl_rl.runners import OnPolicyRunner
from torch.utils.tensorboard import SummaryWriter
from legged_gym.heightmap_prediction.depth_backbone import DepthOnlyFCBackbone58x87 ,RecurrentDepthBackbone
from legged_gym.heightmap_prediction.replay_buffer import ReplayBuffer
from legged_gym.heightmap_prediction.lstm_heightmap_predictor import LSTMHeightmapPredictor



def main(args):
  # del argv  # unused
  # config = FLAGS.config
  env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
  # override some parameters for testing
  env_cfg.env.num_envs = min(env_cfg.env.num_envs, 200)
  env_cfg.terrain.num_rows = 11
  env_cfg.terrain.num_cols = 10
  env_cfg.terrain.curriculum = True
  env_cfg.terrain.max_init_terrain_level = 9
  env_cfg.noise.add_noise = False
  env_cfg.domain_rand.randomize_friction = False
  env_cfg.domain_rand.push_robots = False
  env_cfg.domain_rand.disturbance = False
  env_cfg.domain_rand.randomize_payload_mass = False
  env_cfg.commands.heading_command = False
  # prepare environment
    
  env_cfg.env.episode_length_s = 20 # 2分钟
  env_cfg.commands.resampling_time = 20 # 2分钟更新一次命令
  
  env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

  #TODO:need to add random:
  env.focus = False
  env.commands[:, 0] = 1.
  env.commands[:, 1] = 0
  env.commands[:, 2] = 0
  env.commands[:, 2] = 0

  obs = env.get_observations()

# Setup logging TODO: need to add 
  log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', "height_est")

  logdir = os.path.join(log_root,
                        datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
  if not os.path.exists(logdir):
    os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir, flush_secs=10)

  train_cfg.runner.resume = True
  ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
  policy = ppo_runner.get_inference_policy_wo(device=env.device)


  # Initialize heightmap predictor
  heightmap_predictor = LSTMHeightmapPredictor(
      dim_output=32,
      vertical_res=58,
      horizontal_res=87,
      dim_n_proprio_hist_in=env.cfg.env.num_observations_wo_height,
  ).to(env.device)
  # Initialize replay buffer
  replay_buffer = ReplayBuffer(env, device=env.device)
  use_save_data = False
  if not use_save_data:
    rewards,  _ = replay_buffer.collect_data(
        policy, heightmap_predictor=None, num_steps=5000)
    
    print(f"[Initial Rollout] Average Reward: {np.mean(rewards)}, ")
    replay_buffer.save(os.path.join(logdir, "replay_buffer.pt"))
  else:
    replay_buffer_path =  "/home/tian/Desktop/deep/dreamwaq_baseline/legged_gym/logs/height_est/2024_12_11_13_23_29/replay_buffer.pt"
    replay_buffer.load(replay_buffer_path)
  for step in range(30):
    # Train student policy
    loss = heightmap_predictor.train_on_data(replay_buffer,
                                             batch_size=4,
                                             num_steps=1500)
    heightmap_predictor.save(os.path.join(logdir, f"model_{step}.pt"))
    # Collect more data
    rewards, _ = replay_buffer.collect_data(
        policy,
        heightmap_predictor=heightmap_predictor,
        num_steps=5000)
    replay_buffer.save(os.path.join(logdir, f"replay_buffer{step}.pt"))

    # Log Rewards and Terrain Levels
    print(f"[Dagger step {step}] Average Reward: {np.mean(rewards)}, ")
    writer.add_scalar("Rollout/average_reward", np.mean(rewards), step)
    # writer.add_scalar("Rollout/average_cycles", np.mean(cycles), step)
    writer.add_scalar("Heightmap Training/MSE", loss, step)


if __name__ == "__main__":
  args = get_args()

  main(args)
