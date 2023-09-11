from functools import partial
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.common.common import shard_batch
from jaxrl_m.data.bridge_dataset import BridgeDataset, glob_to_path_list
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders
from jaxrl_m.agents import agents
import tensorflow as tf

import tqdm
import jax
import jax.numpy as jnp
from absl import app, flags, logging
from ml_collections import config_flags
import numpy as np
from flax.training import checkpoints
import os

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)


def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": "jaxrl_m_bridgedata",
            "exp_descriptor": FLAGS.name,
        }
    )
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        debug=FLAGS.debug,
    )

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    # load datasets
    assert type(FLAGS.bridgedata_config.include[0]) == list
    task_paths = [
        glob_to_path_list(
            path, prefix=FLAGS.config.data_path, exclude=FLAGS.bridgedata_config.exclude
        )
        for path in FLAGS.bridgedata_config.include
    ]

    train_paths = [
        [os.path.join(path, "train/out.tfrecord") for path in sub_list]
        for sub_list in task_paths
    ]
    val_paths = [
        [os.path.join(path, "val/out.tfrecord") for path in sub_list]
        for sub_list in task_paths
    ]

    obs_horizon = FLAGS.config.get("obs_horizon")

    train_data = BridgeDataset(
        train_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        train=True,
        action_metadata=FLAGS.bridgedata_config.action_metadata,
        sample_weights=FLAGS.bridgedata_config.sample_weights,
        obs_horizon=obs_horizon,
        **FLAGS.config.dataset_kwargs,
    )
    val_data = BridgeDataset(
        val_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        action_metadata=FLAGS.bridgedata_config.action_metadata,
        train=False,
        obs_horizon=obs_horizon,
        **FLAGS.config.dataset_kwargs,
    )
    train_data_iter = train_data.get_iterator()

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['observations']['image'].shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['observations']['image'].shape[0] // num_devices}"
    )

    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    example_batch = shard_batch(example_batch, sharding)

    # define encoder
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # initialize agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **FLAGS.config.agent_kwargs,
    )
    if FLAGS.config.resume_path is not None:
        agent = checkpoints.restore_checkpoint(FLAGS.config.resume_path, target=agent)
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    timer = Timer()
    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps))):
        timer.tick("total")

        timer.tick("dataset")
        batch = shard_batch(next(train_data_iter), sharding)
        timer.tock("dataset")

        timer.tick("train")
        agent, update_info = agent.update(batch)
        timer.tock("train")

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            timer.tick("val")
            metrics = []
            for batch in val_data.get_iterator():
                rng, val_rng = jax.random.split(rng)
                metrics.append(agent.get_debug_metrics(batch, seed=val_rng))
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_logger.log({"validation": metrics}, step=i)
            timer.tock("val")

        if (i + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, agent, step=i + 1, keep=1e6
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_logger.log({"training": update_info}, step=i)

            wandb_logger.log({"timer": timer.get_average_times()}, step=i)


if __name__ == "__main__":
    app.run(main)
