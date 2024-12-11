import torch
import torch.nn as nn
import sys
import torchvision
from tqdm import tqdm
import numpy as np
class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone,n_proprio, env_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + n_proprio, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(512, 32),
                                last_activation
                            )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

    def train_on_data(self, replay_buffer, batch_size: int, num_steps: int):
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        criterion = nn.MSELoss(reduction='none')


        # print("replay_buffer.num_trajs: ",replay_buffer.num_trajs)
        num_epoches = int(
            num_steps / max(replay_buffer.num_trajs / batch_size, 1)) + 1
        pbar = tqdm(range(num_epoches), desc="Training")
        for _ in pbar:
            losses = []
            dataloader = replay_buffer.to_recurrent_generator(batch_size=batch_size)
            for batch in dataloader:
                optimizer.zero_grad()
                output = self.forward(batch['depth_imgs'],batch['base_states'])
                loss = criterion(output, batch['height_maps'])
                loss = torch.sum(loss, dim=-1) / batch['height_maps'].shape[-1]
                loss = (loss * batch['masks']).sum() / batch['masks'].sum()
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

        pbar.set_postfix({f"Avg Loss": f"{np.mean(losses):.4f}"})
        return np.mean(losses)

    def save(self, model_dir):
        torch.save(self.state_dict(), model_dir)  # pylint: disable=missing-kwoa
        print(f"Predictor saved to: {model_dir}")

    def load(self, model_dir):
        self.load_state_dict(torch.load(model_dir, map_location="cpu"))

class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, scandots_output_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent
    


    