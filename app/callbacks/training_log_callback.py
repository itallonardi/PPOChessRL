import os
import csv
from stable_baselines3.common.callbacks import BaseCallback


def approximate_rating_from_acpl(acpl: float) -> int:
    """
    A simple heuristic to convert ACPL into an approximate rating.
    The logic here is arbitrary; adjust as needed.
    For example:
      rating = 3000 - 15 * acpl
    We'll clamp it to a range [100, 3000] for safety.
    """
    raw = 3000 - 15 * acpl
    return int(max(100, min(3000, raw)))


class ChessRatingLogger(BaseCallback):
    """
    Callback that logs the approximate rating of the agent
    based on ACPL after each episode.
    Saves results to CSV: training_rating_log.csv
    """

    def __init__(self, log_path="training_rating_log.csv", verbose=0):
        super(ChessRatingLogger, self).__init__(verbose)
        self.log_path = log_path

        # If the file does not exist, create with header
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "acpl", "estimated_rating"])

        self.episode_count = 0

    def _on_step(self) -> bool:
        # We do nothing special at each step
        return True

    def _on_episode_end(self):
        """
        Called at the end of each episode. We read info['episode_acpl'] from the
        'self.locals' or 'self.model.env' if available, then log it.
        """
        # 'self.locals["infos"]' is a list of dict for each step in the final episode
        # but the last step typically contains the final info with 'episode_acpl'.
        infos = self.locals.get("infos", [])
        if not infos:
            return

        # Try to see if the last info has 'episode_acpl'
        last_info = infos[-1]
        if "episode_acpl" in last_info:
            acpl = last_info["episode_acpl"]
            rating_estimate = approximate_rating_from_acpl(acpl)

            self.episode_count += 1
            # Escreve no CSV
            with open(self.log_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [self.episode_count, f"{acpl:.2f}", rating_estimate])

            if self.verbose > 0:
                print(
                    f"[ChessRatingLogger] Episode #{self.episode_count} ACPL: {acpl:.2f}, rating ~ {rating_estimate}")
