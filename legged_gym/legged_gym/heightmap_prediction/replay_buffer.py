"""Replay Buffer to store collected trajectories."""
import collections
from typing import Optional

from isaacgym.torch_utils import to_torch  # pylint: disable=unused-import
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class DelayedImageRecorder:
  """Record and outputs delayed depth image."""
  def __init__(self, min_delay=3, max_delay=7):
    self._min_delay = min_delay
    self._max_delay = max_delay
    self.reset()

  def record_new_image(self, image):
    self._image_buffer.append(image)
    while len(self._image_buffer) < self._current_delay_steps:
      self._image_buffer.append(image)

  def get_image(self):
    return self._image_buffer[0]

  def reset(self):
    self._current_delay_steps = np.random.randint(self._min_delay,
                                                  self._max_delay)
    self._image_buffer = collections.deque(maxlen=self._current_delay_steps)


class ReplayBuffer:
  """Replay buffer to store collected trajectories."""
  def __init__(self, env, device: str, min_delay: int = 1, max_delay: int = 3):
    self._env = env
    self._device = device
    self._base_states = []
    # self._foot_positions = []
    self._height_maps = []
    self._depth_imgs = []
    self._steps_count = 0
    self._reward_sums = torch.zeros(env.num_envs, device=self._device)
    self._num_envs = env.num_envs
    self._image_recorders = [
        DelayedImageRecorder(min_delay=min_delay, max_delay=max_delay)
        for _ in range(self._num_envs)
    ]

    self._env.reset()

  def _record_new_traj(self, base_states, height_maps,
                       depth_imgs, start_idx, end_idx, env_id):
    # Input: batched states: list of [num_envs x dim_state]
    # Select the env_id, from start_idx to end_idx
    if start_idx == end_idx:
      return

    self._steps_count += (end_idx - start_idx)
    self._base_states.append(
        torch.stack([
            base_state[env_id] for base_state in base_states[start_idx:end_idx]
        ],
                    dim=0))
    self._height_maps.append(
        torch.stack([
            height_map[env_id] for height_map in height_maps[start_idx:end_idx]
        ],
                    dim=0))
    self._depth_imgs.append(
        torch.stack(
            [depth_img[env_id] for depth_img in depth_imgs[start_idx:end_idx]],
            dim=0))

  def collect_data(self, policy,
                   heightmap_predictor: Optional[torch.nn.Module],
                   num_steps: int):
    self._env.reset()
    policy.reset()
    policy.eval()
    for recorder in self._image_recorders:
      recorder.reset()

    steps_count = 0
    infos = []
    pbar = tqdm(total=num_steps, desc="Collecting Data", leave=True)
    # Initialize trajectory buffers
    base_states = []
    height_maps = []
    depth_imgs = []
    sum_rewards = []
    cycle_counts = []

    start_indices = torch.zeros(self._num_envs,
                                device=self._device,
                                dtype=torch.int64)
    start_count = self._steps_count
    with torch.no_grad():
      # while steps_count * self._num_envs < num_steps:
      while self._steps_count - start_count < num_steps:
        # pbar.update(self._num_envs)
        steps_count += 1
        curr_imgs = []
        # print(self._env.get_depth().size())
        for env_id in range(self._num_envs):
          self._image_recorders[env_id].record_new_image(
              to_torch(self._env.get_depth()[env_id,:],
                       device=self._device))
          curr_imgs.append(self._image_recorders[env_id].get_image())
        curr_imgs = torch.stack(curr_imgs, dim=0).clone()
        self._env.commands[:, 0] = 1
        self._env.commands[:, 1] = 0
        self._env.commands[:, 2] = 0
        self._env.commands[:, 3] = 0
        if heightmap_predictor is None:
          # print(self._env.commands)
          obs = self._env.get_observations()
          action = policy.act_inference(obs.detach())
        else:
          height_predicted = heightmap_predictor.forward(
              depth_image=curr_imgs
              ,proprio_hist = self._env.get_observations()[:,:-187])
          
          obs = self._env.get_observations()
          # height_latent = policy.height_mlp(obs.detach()[:,-187:])
          action = policy.act_inference_wo_height(obs.detach()[:,:-187],height_predicted)

        base_states.append(self._env.get_observations()[:,:-187])
        self._env.get_observations()[:,-187:]
        height_latent = policy.height_mlp(self._env.get_observations()[:,-187:])
        height_maps.append(height_latent)
        depth_imgs.append(curr_imgs)
        # print(curr_imgs)

        if heightmap_predictor is None:
          _, _, reward, dones, info, _,_= self._env.step(action)
        else:
          _, _, reward, dones, info, _,_ = self._env.step(action)
        self._reward_sums += reward.clone()
        if dones.any():
          policy.reset(dones)
          if heightmap_predictor is not None:
            heightmap_predictor.reset(dones)
          # cycle_counts.append(info["episode"]["cycle_count"].cpu().numpy())
          done_idx = dones.nonzero(as_tuple=False).flatten()
          sum_rewards.extend(list(self._reward_sums[done_idx].cpu().numpy()))
          self._reward_sums[done_idx] = 0
          for env_id in done_idx:
            self._image_recorders[env_id].reset()
            self._record_new_traj(base_states, height_maps,
                                  depth_imgs, start_indices[env_id].item(),
                                  steps_count, env_id)
            pbar.update(steps_count - start_indices[env_id].item())
            start_indices[env_id] = steps_count

    pbar.close()
    return sum_rewards, infos

  def to_dataloader(self, batch_size: int):
    dataset = TensorDataset(torch.concatenate(self._base_states, dim=0),
                            torch.concatenate(self._height_maps, dim=0),
                            torch.concatenate(self._depth_imgs, dim=0))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

  def _prepare_padded_sequence(self, traj_indices):
    traj_lengths = [self._base_states[idx].shape[0] for idx in traj_indices]
    max_length = max(traj_lengths)
    num_trajs = len(traj_indices)
    # print("self._depth_imgs[0].shape",self._depth_imgs[0].size())
    base_states = torch.zeros(
        (max_length, num_trajs, self._base_states[0].shape[1]),
        device=self._device)
    height_maps = torch.zeros(
        (max_length, num_trajs, self._height_maps[0].shape[1]),
        device=self._device)
    depth_imgs = torch.zeros(
        (max_length, num_trajs, self._depth_imgs[0].shape[1],
         self._depth_imgs[0].shape[2]),
        device=self._device)
    masks = torch.zeros((max_length, num_trajs),
                        dtype=torch.bool,
                        device=self._device)
    for output_idx, traj_idx in enumerate(traj_indices):
      base_states[:traj_lengths[output_idx],
                  output_idx] = self._base_states[traj_idx]
      height_maps[:traj_lengths[output_idx],
                  output_idx] = self._height_maps[traj_idx]
      depth_imgs[:traj_lengths[output_idx],
                 output_idx] = self._depth_imgs[traj_idx]
      masks[:traj_lengths[output_idx], output_idx] = 1

    return dict(base_states=base_states,
                height_maps=height_maps,
                depth_imgs=depth_imgs,
                masks=masks)

  def to_recurrent_generator(self, batch_size: int):
    num_trajs = len(self._base_states)
    traj_indices = np.arange(num_trajs)
    traj_indices = np.random.permutation(traj_indices)
    for start_idx in range(0, num_trajs, batch_size):
      end_idx = np.minimum(start_idx + batch_size, num_trajs)#prevent from leaving the last trajs
      yield self._prepare_padded_sequence(traj_indices[start_idx:end_idx])

  def save(self, rb_dir):
    torch.save(
        dict(base_states=self._base_states,
             height_maps=self._height_maps,
             depth_imgs=self._depth_imgs), rb_dir)
    print(f"Replay buffer saved to: {rb_dir}")

  def load(self, rb_dir):
    traj = torch.load(rb_dir)
    self._base_states = traj["base_states"]
    self._height_maps = traj["height_maps"]
    self._depth_imgs = traj["depth_imgs"]
    # print(" self._depth_imgs: ", self._depth_imgs.size())
    # print(" self._height_maps: ", self._height_maps.size())
  @property
  def num_trajs(self):
    return len(self._base_states)
