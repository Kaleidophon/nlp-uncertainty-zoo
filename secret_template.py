"""
Module defining personal information that should not be comitted to a repo.

When cloning the repo, rename secret_template.py to secret.py.
"""

# Country code for emission tracking
COUNTRY_CODE: str = ...

# Telegram info for knockknock
TELEGRAM_CHAT_ID: int = ...
TELEGRAM_API_TOKEN: str = ...

# Weights & Biases API key
WANDB_API_KEY = "eb1d299f4c3a1646708af0244fca671abbf3d5e2"
