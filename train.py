"""Training entry point; configure experiments with YAML."""
from terminal_learner.train import train
from terminal_learner.model.loop import Trainer
from terminal_learner.cli import main, parse_args

if __name__ == "__main__":
    main()
