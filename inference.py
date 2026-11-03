"""Compatibility entry point; inference implementation lives in terminal_learner."""
from terminal_learner.session import AgentSession
from terminal_learner.inference import main

if __name__ == "__main__":
    main()
