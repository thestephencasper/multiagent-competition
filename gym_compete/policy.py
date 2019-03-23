"""Abstract policy class and some concrete implementations."""

import copy

from gym.spaces import Box
import numpy as np
from stable_baselines.a2c.utils import seq_to_batch
from stable_baselines.common.distributions import DiagGaussianProbabilityDistribution
from stable_baselines.common.policies import ActorCriticPolicy
import tensorflow as tf


class RunningMeanStd(object):
    def __init__(self, scope="running", reuse=False, epsilon=1e-2, shape=()):
        with tf.variable_scope(scope, reuse=reuse):
            self._sum = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="sum", trainable=False)
            self._sumsq = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(epsilon),
                name="sumsq", trainable=False)
            self._count = tf.get_variable(
                dtype=tf.float32,
                shape=(),
                initializer=tf.constant_initializer(epsilon),
                name="count", trainable=False)
            self.shape = shape

            self.mean = tf.to_float(self._sum / self._count)
            var_est = tf.to_float(self._sumsq / self._count) - tf.square(self.mean)
            self.std = tf.sqrt(tf.maximum(var_est, 1e-2))


def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret


def switch(condition, if_exp, else_exp):
    x_shape = copy.copy(if_exp.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: if_exp,
                lambda: else_exp)
    x.set_shape(x_shape)
    return x


class GymCompetePolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 reuse=False, normalize=False):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                         reuse=reuse, scale=False)
        self.normalized = normalize

    def restore(self, params):
        with self.sess.graph.as_default():
            var_list = self.get_serializable_variables()
            shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
            total_size = np.sum([int(np.prod(shape)) for shape in shapes])
            theta = tf.placeholder(tf.float32, [total_size])

            start = 0
            assigns = []
            for (shape, v) in zip(shapes, var_list):
                size = int(np.prod(shape))
                assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
                start += size

            op = tf.group(*assigns)
            self.sess.run(op, {theta: params})

    def get_serializable_variables(self):
        vars = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
        vars = [x for x in vars if 'Adam' not in x.name]
        return vars

    def get_trainable_variables(self):
        return self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class MlpPolicyValue(GymCompetePolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, hiddens, scope="input",
                 reuse=False, normalize=False):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                         reuse=reuse, normalize=normalize)
        self.initial_state = None
        with self.sess.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                self.scope = tf.get_variable_scope().name

                assert isinstance(ob_space, Box)

                self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")

                if self.normalized:
                    if self.normalized != 'ob':
                        self.ret_rms = RunningMeanStd(scope="retfilter")
                    self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

                obz = self.processed_obs
                if self.normalized:
                    obz = tf.clip_by_value((self.processed_obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

                last_out = obz
                for i, hid_size in enumerate(hiddens):
                    last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i + 1)))
                self.value_fn = dense(last_out, 1, "vffinal")
                if self.normalized and self.normalized != 'ob':
                    self.value_fn = self.value_fn * self.ret_rms.std + self.ret_rms.mean  # raw = not standardized

                last_out = obz
                for i, hid_size in enumerate(hiddens):
                    last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i + 1)))
                mean = dense(last_out, ac_space.shape[0], "polfinal")
                logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]], initializer=tf.zeros_initializer())

                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                self.proba_distribution = DiagGaussianProbabilityDistribution(pdparam)
                self.sampled_action = switch(self.stochastic_ph,
                                             self.proba_distribution.sample(),
                                             self.proba_distribution.mode())
                self.policy = mean
                self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        outputs = [self.sampled_action, self._value, self.neglogp]
        a, v, neglogp = self.sess.run(outputs, {
            self.obs_ph: obs,
            self.stochastic_ph: not deterministic,
        })
        return a, v, None, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        value = self.sess.run(self._value, {self.obs_ph: obs})
        return value


class LSTMPolicy(GymCompetePolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, hiddens, scope="input",
                 reuse=False, normalize=False):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                         reuse=reuse, normalize=normalize)
        with self.sess.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                self.scope = tf.get_variable_scope().name

                assert isinstance(ob_space, Box)

                self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
                # We don't use masks_ph, but Stable Baselines needs it to exist
                self.masks_ph = tf.placeholder(dtype=tf.float32, shape=[n_batch], name="masks")

                # Observation Normalization
                if self.normalized:
                    if self.normalized != 'ob':
                        self.ret_rms = RunningMeanStd(scope="retfilter")
                    self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

                obz = self.obs_ph
                if self.normalized:
                    obz = tf.clip_by_value((self.obs_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

                num_lstm = hiddens[-1]
                self.states_ph = tf.placeholder(tf.float32, [self.n_env, 4, num_lstm], name="lstmpv_ch")
                self.state_out = []
                states = tf.transpose(self.states_ph, (1, 0, 2))
                self.zero_state = np.zeros((4, num_lstm), dtype=np.float32)

                def lstm(start, suffix):
                    # Feed forward
                    ff_out = obz
                    for hidden in hiddens[:-1]:
                        ff_out = tf.contrib.layers.fully_connected(ff_out, hidden)

                    # Batch->Seq
                    input_seq = tf.reshape(ff_out, [self.n_env, n_steps, -1])
                    input_seq = tf.transpose(input_seq, (1, 0, 2))
                    masks = tf.reshape(self.masks_ph, [self.n_env, n_steps, 1])

                    # RNN
                    inputs_ta = tf.TensorArray(dtype=tf.float32, size=n_steps)
                    inputs_ta = inputs_ta.unstack(input_seq)

                    cell = tf.contrib.rnn.BasicLSTMCell(num_lstm, reuse=reuse)
                    initial_state = tf.contrib.rnn.LSTMStateTuple(states[start], states[start + 1])

                    def loop_fn(time, cell_output, cell_state, loop_state):
                        emit_output = cell_output

                        elements_finished = time >= n_steps
                        finished = tf.reduce_all(elements_finished)

                        # TODO: use masks
                        mask = tf.cond(finished,
                                       lambda: tf.zeros([self.n_env, 1], dtype=tf.float32),
                                       lambda: masks[:, time, :])
                        next_cell_state = cell_state or initial_state
                        next_cell_state = tf.contrib.rnn.LSTMStateTuple(next_cell_state.c * (1 - mask),
                                                                        next_cell_state.h * (1 - mask))

                        next_input = tf.cond(
                            finished,
                            lambda: tf.zeros([self.n_env, ff_out.shape[-1]],
                                             dtype=tf.float32),
                            lambda: inputs_ta.read(time))
                        next_loop_state = None
                        return (elements_finished, next_input, next_cell_state,
                                emit_output, next_loop_state)

                    outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn,
                                                               parallel_iterations=1,
                                                               scope=f'lstm{suffix}')
                    last_out = outputs_ta.stack()
                    last_out = seq_to_batch(last_out)
                    self.state_out.append(final_state)

                    return last_out

                value_out = lstm(0, 'v')
                self.value_fn = tf.contrib.layers.fully_connected(value_out, 1, activation_fn=None)
                if self.normalized and self.normalized != 'ob':
                    self.value_fn = self.value_fn * self.ret_rms.std + self.ret_rms.mean  # raw = not standardized

                mean = lstm(2, 'p')
                mean = tf.contrib.layers.fully_connected(mean, ac_space.shape[0],
                                                         activation_fn=None)
                logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]],
                                         initializer=tf.zeros_initializer())

                mean = tf.reshape(mean, [n_batch] + list(ac_space.shape))
                logstd = tf.reshape(logstd, ac_space.shape)

                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                self.proba_distribution = DiagGaussianProbabilityDistribution(pdparam)
                self.sampled_action = switch(self.stochastic_ph,
                                             self.proba_distribution.sample(),
                                             self.proba_distribution.mode())
                self.policy = mean

                self.initial_state = np.tile(self.zero_state, (self.n_env, 1, 1))

                for p in self.get_trainable_variables():
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.square(p)))

                self._setup_init()

    def _make_feed_dict(self, obs, state, mask):
        return {
            self.obs_ph: obs,
            self.states_ph: state,
            self.masks_ph: mask,
        }

    def step(self, obs, state=None, mask=None, deterministic=False):
        outputs = [self.sampled_action, self._value, self.state_out, self.neglogp]
        feed_dict = self._make_feed_dict(obs, state, mask)
        feed_dict[self.stochastic_ph] = not deterministic
        a, v, s, neglogp = self.sess.run(outputs, feed_dict)
        state = []
        for x in s:
            state.append(x.c)
            state.append(x.h)
        state = np.array(state)
        state = np.transpose(state, (1, 0, 2))
        return a, v, state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, self._make_feed_dict(obs, state, mask))

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, self._make_feed_dict(obs, state, mask))

