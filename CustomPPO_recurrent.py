# Author: Ahmet Ege Yilmaz
# Year: 2021
# The code in this file is a customized version of stable_baselines3's PPO implementation, which makes training of Echo State Networks possible.

from faulthandler import disable
import time
from numpy.core.numeric import roll
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import GymEnv, Schedule, TensorDict, TrainFreq, TrainFrequencyUnit
import warnings
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
import torch
from torch.nn import functional as F
import gym
from gym import spaces
import numpy as np
from typing import Any, Dict, Generator, List, Optional, Union,Type, Tuple
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)

from tqdm import trange,tqdm

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: gym.spaces.Space,
                        action_space: gym.spaces.Space, 
                        lr_schedule: Schedule, 
                        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None, 
                        activation_fn: Type[torch.nn.Module] = torch.nn.Tanh,
                        ortho_init: bool = True, 
                        use_sde: bool = False, 
                        log_std_init: float = 0, 
                        full_std: bool = True, 
                        sde_net_arch: Optional[List[int]] = None, 
                        use_expln: bool = False, 
                        squash_output: bool = False, 
                        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                        features_extractor_kwargs: Optional[Dict[str, Any]] = None, 
                        normalize_images: bool = True, 
                        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                        optimizer_kwargs: Optional[Dict[str, Any]] = None):

        super().__init__(observation_space, 
                        action_space, 
                        lr_schedule, 
                        net_arch, 
                        activation_fn, 
                        ortho_init, 
                        use_sde, 
                        log_std_init, 
                        full_std, 
                        sde_net_arch, 
                        use_expln, 
                        squash_output, 
                        features_extractor_class, 
                        features_extractor_kwargs, 
                        normalize_images, 
                        optimizer_class, 
                        optimizer_kwargs)

    def forward(self, obs, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde=latent_sde)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_latent(self, obs):
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Features for sde
        latent_sde = latent_pi
        if self.sde_features_extractor is not None:
            latent_sde = self.sde_features_extractor(features)
        return latent_pi, latent_vf, latent_sde

    def evaluate_actions(self, obs, actions):
        latent_pi, latent_vf, latent_sde = self._get_latent(obs)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

class CustomRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        
        super(CustomRolloutBuffer, self).__init__(buffer_size=buffer_size,observation_space=observation_space,action_space=action_space,
        device=device,gae_lambda=gae_lambda,gamma=gamma,n_envs=n_envs)
    
    def get(self, batch_size: Optional[int] = None, mute_tqdm=0) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        # indices = np.random.permutation(self.buffer_size * self.n_envs)
        # indices.sort() #MY MODIFICATION
        #MY MODIFICATION
        env_inds = np.random.permutation(self.n_envs)
        # Prepare the data
        if not self.generator_ready:

            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            # observations (2048, 3, 4)
            # observations (6144, 4)
            # actions (2048, 3, 1)
            # actions (6144, 1)
            # values (2048, 3)
            # values (6144, 1)
            # log_probs (2048, 3)
            # log_probs (6144, 1)
            # advantages (2048, 3)
            # advantages (6144, 1)
            # returns (2048, 3)
            # returns (6144, 1)
            
            # MY MODIFICATION
            # for tensor in _tensor_names:
            #     self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True


        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # MY MODIFICATION
        # start_idx =  0
        # while start_idx < self.buffer_size:                           #    MY MODIFICATION
        # while start_idx < self.buffer_size * self.n_envs:                        # MY MODIFICATION
            # yield self._get_samples(indices[start_idx : start_idx + batch_size],env_inds[env_start_idx])# , (res_memory[:,indices[start_idx : start_idx + batch_size]] for res_memory in self._reservoir_layers_memory)
        
        # MY MODIFICATION
        # her env'e bi kere ugra ve batch size kadar replay memory passle

        env_start_idx = 0
        pbar = tqdm(total = self.n_envs,disable=mute_tqdm)
        while env_start_idx < self.n_envs:
            pbar.set_description(f'Collecting replay from Env{env_inds[env_start_idx]}...')
            start_idx = np.random.randint(self.buffer_size-batch_size+1)
            yield self._get_samples(np.arange(start_idx,start_idx+batch_size,dtype=int),env_inds[env_start_idx])
            env_start_idx+=1
            pbar.update(1)
        pbar.close()
                                                # MY MODIFICATION
    def _get_samples(self, batch_inds: np.ndarray, env_ind) -> RolloutBufferSamples:
        data = (                  # MY MODIFICATION
            self.observations[batch_inds,env_ind],
            self.actions[batch_inds,env_ind],
            self.values[batch_inds,env_ind].flatten(),
            self.log_probs[batch_inds,env_ind].flatten(),
            self.advantages[batch_inds,env_ind].flatten(),
            self.returns[batch_inds,env_ind].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class CustomOnPolicyAlgorithm(OnPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        policy_base: Type[BasePolicy] = CustomActorCriticPolicy,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None
        ,T=10
        ,mute_tqdm = 0):
        
        # MY MODIFICATION
        # self._T = T
        self.__mute_tqdm = mute_tqdm
        ###########
        super(CustomOnPolicyAlgorithm,self).__init__(
            policy = policy
            ,env = env
            ,learning_rate = learning_rate
            ,n_steps = n_steps
            ,gamma = gamma
            ,gae_lambda = gae_lambda
            ,ent_coef = ent_coef
            ,vf_coef = vf_coef
            ,max_grad_norm = max_grad_norm
            ,use_sde = use_sde
            ,sde_sample_freq = sde_sample_freq
            ,policy_base = policy_base
            ,tensorboard_log = tensorboard_log
            ,create_eval_env = create_eval_env
            ,monitor_wrapper = monitor_wrapper
            ,policy_kwargs = policy_kwargs
            ,verbose = verbose
            ,seed = seed
            ,device = device
            ,_init_setup_model = _init_setup_model
            ,supported_action_spaces = supported_action_spaces
        )

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else CustomRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # MY MODIFICATION
        # self.rollout_buffer._reservoir_layers_memory = None

        # self.rollout_buffer_reservoir = buffer_cls(
        #     self.n_steps*self._T,
        #     self.observation_space,
        #     self.action_space,
        #     self.device,
        #     gamma=self.gamma,
        #     gae_lambda=self.gae_lambda,
        #     n_envs=self.n_envs)

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        mute_tqdm=0
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # self.rollout_buffer_reservoir.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        # MY MODIFICATION  
        # if hasattr(self.policy,'has_reservoirs'):
        # self.rollout_buffer._reservoir_layers_memory = [torch.zeros(res.resSize,n_rollout_steps) for res in self.policy.mlp_extractor.reservoirs]
        # for res in self.policy.mlp_extractor.reservoirs:
        #     assert not res.reservoir_layer.requires_grad
        self.policy.collapse_reservoirs(env.num_envs)
        # for res in self.policy.mlp_extractor.reservoirs:
        #     assert res.reservoir_layer.sum()==0,res.reservoir_layer.sum()
            # assert res.reservoir_layer.shape == (env.num_envs,res.resSize,1),res.reservoir_layer.shape
            # assert res._layer_mode == 'ensemble',res._layer_mode
                                # MY MODIFICATION             
        # while n_steps < n_rollout_steps * self._T:

        # MY MODIFICATION
        pbar = tqdm(total = n_rollout_steps,disable=mute_tqdm)
        if not mute_tqdm:
            print('Creating replay memory...')
        while n_steps < n_rollout_steps:
            pbar.set_description(f'Play No: {n_steps+1}')
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # MY MODIFICATION
                # check = self.policy.mlp_extractor.res1.reservoir_layer.clone()
                ######
                actions, values, log_probs = self.policy.forward(obs_tensor)
                # MY MODIFICATION
                # assert not torch.all(check ==self.policy.mlp_extractor.res1.reservoir_layer)
                ######
            actions = actions.cpu().numpy().ravel()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            # MY MODIFICATION
            # for res in self.policy.mlp_extractor.reservoirs:
            #     assert not res.reservoir_layer.requires_grad
            self.policy.reset_reservoirs(dones=dones)
            # for res in self.policy.mlp_extractor.reservoirs:
            #     assert res.reservoir_layer[dones,:,:].sum()==0,res.reservoir_layer[dones,:,:].sum()
                # assert res.reservoir_layer.shape == (env.num_envs,res.resSize,1),res.reservoir_layer.shape
                # assert res._layer_mode == 'ensemble',res._layer_mode

            # MY MODIFICATION
            # for i,res in enumerate(self.policy.mlp_extractor.reservoirs):
            #     self.rollout_buffer._reservoir_layers_memory[i][:,n_steps] = res.reservoir_layer.ravel()

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # MY MODIFICATION
            # self.rollout_buffer_reservoir.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            # if (n_steps-1)%self._T == self._T - 1:
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            ############
            self._last_obs = new_obs
            self._last_episode_starts = dones
            # MY MODIFICATION
            pbar.update(1)
        # MY MODIFICATION
        pbar.close()

        with torch.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()
        # MY MODIFICATION
        # if hasattr(self.policy,'has_reservoirs'):
        # for res in self.policy.mlp_extractor.reservoirs:
        #     assert not res.reservoir_layer.requires_grad
        self.policy.expand_reservoirs()
        # for res in self.policy.mlp_extractor.reservoirs:
        #     assert res.reservoir_layer.sum()==0,res.reservoir_layer.sum()
            # assert res.reservoir_layer.shape == (res.resSize,self.policy.mlp_extractor.batch_size),res.reservoir_layer.shape
            # assert res._layer_mode == 'batch',res._layer_mode

        return True

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps,mute_tqdm=self.__mute_tqdm)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self


class CustomPPO(CustomOnPolicyAlgorithm):

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        T=10,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        mute_tqdm=0
    ):

        super(CustomPPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            )
            ,T=T
            ,mute_tqdm=mute_tqdm
        )

        # MY MODIFICATION
        # self._T = T
        self.__mute_tqdm = mute_tqdm
        ############

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
            batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(CustomPPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs

        for epoch in (pbar := trange(self.n_epochs,disable=self.__mute_tqdm)):
            pbar.set_description(f"PPO Epoch: {epoch+1}...")

            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            # MY MODIFICATION
            # indices = np.random.permutation(self.rollout_buffer.buffer_size * self.n_envs)
            # indices.sort()

            # indices_res = np.random.permutation(self.rollout_buffer_reservoir.buffer_size * self.n_envs)
            # indices_res.sort()
            # for rollout_data , rollout_data_reservoir in zip(self.rollout_buffer.get(self.batch_size,indices),self.rollout_buffer_reservoir.get(self.batch_size*self._T,indices_res)):
            
            # MY MODIFICATION
            # for rollout_data , res_memories in self.rollout_buffer.get(self.batch_size,indices):
            for rollout_data in self.rollout_buffer.get(self.batch_size,mute_tqdm=self.__mute_tqdm):
                #MY MODIFICATON
                self.policy.reset_reservoirs()

                # print([*map(lambda x: x.shape,rollout_data)])
                # MY MODIFICATION
                # with torch.no_grad():
                #     for res, memo in zip(self.policy.mlp_extractor.reservoirs,res_memories):
                #         res.reservoir_layer[:,:] = memo
                # res_obs = rollout_data_reservoir.observations.reshape(self.batch_size,self._T,-1)
                # assert torch.all(res_obs[:,-1,:] == rollout_data.observations),[res_obs[0],rollout_data.observations]
                                                # MY MODIFICATION
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                # MY MODIFICATION
                # self.policy.mlp_extractor.res1.reset_reservoir_layer()
                # for i in range(self._T):
                    # self.policy.mlp_extractor.res1.update_reservoir_layer(res_obs[:,i,:].T)

                # print(rollout_data.observations.shape,actions.shape) # torch.Size([64, 3, 4]) torch.Size([64, 3, 1])
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                # self.policy.reset_reservoirs()
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "CustomPPO":

        return super(CustomPPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )