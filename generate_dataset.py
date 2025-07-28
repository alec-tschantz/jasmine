from dataclasses import dataclass
from pathlib import Path

import numpy as np
import gymnasium as gym
import tyro


@dataclass
class Args:
    num_episodes: int = 1000
    output_dir: str = "data/episodes"
    min_episode_length: int = 50
    max_steps_per_episode: int = 1000


args = tyro.cli(Args)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

metadata = []
i = 0
while i < args.num_episodes:
    seed = np.random.randint(0, 10000)

    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    frames = []
    for step in range(args.max_steps_per_episode):

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        frames.append(frame)

        if terminated or truncated:
            break

    env.close()

    if len(frames) >= args.min_episode_length:
        episode_array = np.stack(frames, axis=0).astype(np.uint8)
        ep_path = output_dir / f"episode_{i}.npy"
        np.save(ep_path, episode_array)

        metadata.append({"path": str(ep_path), "length": len(frames)})
        print(f"Episode {i} completed, length: {len(frames)}")
        i += 1
    else:
        print(f"Episode too short ({len(frames)}), retrying...")

meta_path = output_dir / "metadata.npy"
np.save(meta_path, metadata)
print(f"Dataset generated with {len(metadata)} valid episodes")
