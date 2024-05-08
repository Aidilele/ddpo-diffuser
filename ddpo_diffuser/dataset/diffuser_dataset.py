import torch
import os
import numpy as np


class DeciDiffuserDataset:
    """Implementation of DecisionDiffuser dataset."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 256,
        gpu_threshold: int = 1024,
        device: torch.device = torch.device('cpu'),
        horizon: int = 64,
        discount: float = 0.99,
        # returns_scale: int = 1000,
        include_returns: bool = True,
        include_constraints: bool = True,
        include_skills: bool = True,
    ) -> None:
        """Initialize for Class DeciDiffuserDataset."""
        # super().__init__(
        #     dataset_name=dataset_name,
        #     batch_size=batch_size,
        #     gpu_threshold=gpu_threshold,
        #     device=device,
        # )

        self._default_datasets_dir = './datasets/'
        file_path = os.path.join(self._default_datasets_dir, f'{dataset_name}.npz')
        data = np.load(file_path)
        self._batch_size = batch_size
        self._gpu_threshold = gpu_threshold
        self._pre_transfer = False

        # Determine whether to use GPU or not
        for field in data.files:
            self.__setattr__(field, torch.from_numpy(data[field]).to(device=device))
        self._device = device
        self._length = len(self.obs)

        if self._name_to_metadata[dataset_name].episode_length is None:
            self.episode_length = torch.where(self.done == 1)[0][0].item() + 1
        else:
            self.episode_length = self._name_to_metadata[dataset_name].episode_length
        self.num_trajs = len(self.obs) // self.episode_length
        assert horizon <= self.episode_length, 'Horizon is not allowed large than episode length.'
        self.horizon = horizon
        self.discount = discount
        self.discounts = self.discount ** torch.arange(self.episode_length, device=device)
        self.include_returns = include_returns
        # self.returns_scale = returns_scale
        self.include_constraints = include_constraints
        self.include_skills = include_skills
        rewards = self.reward.view(-1, self.episode_length)
        returns = torch.zeros_like(rewards)
        # self.returns = (self.reward * self.discounts.repeat(self.num_trajs)).view(-1, self.episode_length)
        # for i in range(rewards.shape[0]):
        for start in range(rewards.shape[1]):
            returns[:, start] = (
                rewards[:, start:] * self.discounts[: (self.episode_length - start)]
            ).sum(dim=1)
        self.returns = returns.view(-1)
        self.returns_scale = self.returns.max() - self.returns.min()
        self.returns = 2 * (self.returns - self.returns.min()) / self.returns_scale - 1

    def get_returns(self, indices: torch.Tensor) -> torch.Tensor:
        """Get returns tensor for training."""
        return self.returns[indices].view(-1, 1)
        # return self.reward[indices].view(-1, self.horizon, 1).mean(dim=1)


    def get_constraints(self, indices: torch.Tensor) -> torch.Tensor:
        """Get constraints tensor for training."""
        # constraint = self.constraint[indices].view(-1, self.horizon, 2).mean(dim=1)
        # constraint[torch.where(constraint > 0.5)] = 1.0
        # constraint[torch.where(constraint <= 0.5)] = 0.0
        # # constraint = self.constraint[indices].view(-1, 2)
        # constraint[:,0]= self.cost[indices].view(-1, self.horizon, 1).mean(dim=1)
        return self.cost[indices].view(-1, self.horizon, 1).mean(dim=1)
        # constraint = constraint.repeat(1,2)

    def get_skills(self, indices: torch.Tensor) -> torch.Tensor:
        """Get skills tensor for training."""
        return self.skill[indices].view(-1, 2)

    def sample(
        self,
    ) -> tuple:
        """Sample a batch of data from the dataset."""
        # indices = torch.randint(low=0, high=len(self), size=(self._batch_size,), dtype=torch.int64)

        traj_indices = torch.randint(low=0, high=self.num_trajs, size=(self._batch_size,))
        traj_start_indices = torch.randint(
            low=0,
            high=self.episode_length - self.horizon,
            size=(self._batch_size,),
        )
        indices_start = traj_indices * self.episode_length + traj_start_indices
        # traj_indices * self.episode_length + traj_start_indices + self.horizon
        # batch_returns = self.returns[indices_start].view(-1, 1)

        indices = indices_start.view(-1, 1).repeat(1, self.horizon).view(-1) + torch.arange(
            self.horizon,
        ).repeat(self._batch_size)

        batch_obs = self.obs[indices].view(self._batch_size, self.horizon, -1)
        batch_action = self.action[indices].view(self._batch_size, self.horizon, -1)
        batch_trajectories = torch.cat([batch_action, batch_obs], dim=-1)

        if not self._pre_transfer:
            batch_trajectories = batch_trajectories.to(device=self._device)
            # batch_returns = batch_returns.to(device=self._device)
            # for key, value in batch_conditions.items():
            #     batch_conditions[key] = batch_conditions[key].to(device=self._device)

        sample_batch = [batch_trajectories]
        if self.include_returns:
            sample_batch.append(self.get_returns(indices))
        if self.include_constraints:
            sample_batch.append(self.get_constraints(indices))
        if self.include_skills:
            sample_batch.append(self.get_skills(indices_start))

        return tuple(sample_batch)
