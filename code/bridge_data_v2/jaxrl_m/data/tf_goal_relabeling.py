"""
Contains goal relabeling and reward logic written in TensorFlow.

Each relabeling function takes a trajectory with keys "observations",
"next_observations", and "terminals". It returns a new trajectory with the added
keys "goals", "rewards", and "masks". Keep in mind that "observations" and
"next_observations" may themselves be dictionaries, and "goals" must match their
structure.

"masks" determines when the next Q-value is masked out. Typically this is NOT(terminals).
"""

import tensorflow as tf


def uniform(traj, *, reached_proportion):
    """
    Relabels with a true uniform distribution over future states. With
    probability reached_proportion, observations[i] gets a goal
    equal to next_observations[i].  In this case, the reward is 0. Otherwise,
    observations[i] gets a goal sampled uniformly from the set
    next_observations[i + 1:], and the reward is -1.
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # select a random future index for each transition i in the range [i + 1, traj_len)
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len) + 1, tf.float32)
    high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # TODO(kvablack): don't know how I got an out-of-bounds during training,
    # could not reproduce, trying to patch it for now
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    # select a random proportion of transitions to relabel with the next observation
    goal_reached_mask = tf.random.uniform([traj_len]) < reached_proportion

    # the last transition must be goal-reaching
    goal_reached_mask = tf.logical_or(
        goal_reached_mask, tf.range(traj_len) == traj_len - 1
    )

    # make goal-reaching transitions have an offset of 0
    goal_idxs = tf.where(goal_reached_mask, tf.range(traj_len), goal_idxs)

    # select goals
    traj["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        traj["next_observations"],
    )

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

    # add masks
    traj["masks"] = tf.logical_not(traj["terminals"])

    return traj


def last_state_upweighted(traj, *, reached_proportion):
    """
    A weird relabeling scheme where the last state gets upweighted. For each
    transition i, a uniform random number is generated in the range [i + 1, i +
    traj_len). It then gets clipped to be less than traj_len. Therefore, the
    first transition (i = 0) gets a goal sampled uniformly from the future, but
    for i > 0 the last state gets more and more upweighted.
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # select a random future index for each transition
    offsets = tf.random.uniform(
        [traj_len],
        minval=1,
        maxval=traj_len,
        dtype=tf.int32,
    )

    # select random transitions to relabel as goal-reaching
    goal_reached_mask = tf.random.uniform([traj_len]) < reached_proportion
    # last transition is always goal-reaching
    goal_reached_mask = tf.logical_or(
        goal_reached_mask, tf.range(traj_len) == traj_len - 1
    )

    # the goal will come from the current transition if the goal was reached
    offsets = tf.where(goal_reached_mask, 0, offsets)

    # convert from relative to absolute indices
    indices = tf.range(traj_len) + offsets

    # clamp out of bounds indices to the last transition
    indices = tf.minimum(indices, traj_len - 1)

    # select goals
    traj["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, indices),
        traj["next_observations"],
    )

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

    # add masks
    traj["masks"] = tf.logical_not(traj["terminals"])

    return traj


def geometric(traj, *, reached_proportion, discount):
    """
    Relabels with a geometric distribution over future states. With
    probability reached_proportion, observations[i] gets a goal
    equal to next_observations[i].  In this case, the reward is 0. Otherwise,
    observations[i] gets a goal sampled geometrically from the set
    next_observations[i + 1:], and the reward is -1.
    """
    traj_len = tf.shape(traj["terminals"])[0]

    # geometrically select a future index for each transition i in the range [i + 1, traj_len)
    arange = tf.range(traj_len)
    is_future_mask = tf.cast(arange[:, None] < arange[None], tf.float32)
    d = discount ** tf.cast(arange[None] - arange[:, None], tf.float32)

    probs = is_future_mask * d
    # The indexing changes the shape from [seq_len, 1] to [seq_len]
    goal_idxs = tf.random.categorical(
        logits=tf.math.log(probs), num_samples=1, dtype=tf.int32
    )[:, 0]

    # select a random proportion of transitions to relabel with the next observation
    goal_reached_mask = tf.random.uniform([traj_len]) < reached_proportion

    # the last transition must be goal-reaching
    goal_reached_mask = tf.logical_or(
        goal_reached_mask, tf.range(traj_len) == traj_len - 1
    )

    # make goal-reaching transitions have an offset of 0
    goal_idxs = tf.where(goal_reached_mask, tf.range(traj_len), goal_idxs)

    # select goals
    traj["goals"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        traj["next_observations"],
    )

    # reward is 0 for goal-reaching transitions, -1 otherwise
    traj["rewards"] = tf.cast(tf.where(goal_reached_mask, 0, -1), tf.int32)

    # add masks
    traj["masks"] = tf.logical_not(traj["terminals"])

    return traj


GOAL_RELABELING_FUNCTIONS = {
    "uniform": uniform,
    "last_state_upweighted": last_state_upweighted,
    "geometric": geometric,
}
