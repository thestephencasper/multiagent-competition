"""Copied and lightly modified from Gym version 0.9.3.

Uses old MuJoCo 1.3.1.

This is needed for backwards compatibility so as not to break policies trained in this environment.
"""

import os

from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

import mujoco_py


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim: mujoco_py.MjSim = mujoco_py.MjSim(self.model)
        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):

        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer.autoscale()
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        state_array = self.sim.get_state().flatten()

        # See https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjsimstate.pyx
        idx_time = 0
        idx_qpos = idx_time + 1
        idx_qvel = idx_qpos + self.sim.model.nq
        end_qvel = idx_qvel + self.sim.model.nv

        state_array[idx_qpos:idx_qvel] = qpos  # Indices 1 and 2 correspond to the respective values, override the values with passed args
        state_array[idx_qvel:end_qvel] = qvel
        self.sim.set_state_from_flattened(state_array)
        # I assume this isn't necessary and is done in forward()?
        # self.sim._compute_subtree()  # pylint: disable=W0212
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self._get_viewer().finish()
                self.viewer = None
            return

        if mode == 'rgb_array':
            self._get_viewer().render()
            data, width, height = self._get_viewer().get_image()
            return np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()
            # self._get_viewer().loop_once()

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            # self.viewer.start()
            # self.viewer.set_model(self.model)
            # self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.sim.data.com_subtree[idx]

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.body_comvels[idx]

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.sim.data.xmat[idx].reshape((3, 3))

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
