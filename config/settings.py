"""
Central configuration for the orchestrator system.
Edit these values to match your local setup.
"""

# ─── Local LLM Endpoints ────────────────────────────────────────────────────
#
# Nyx supports both LM Studio and Ollama, as long as the backend exposes
# an OpenAI-compatible chat API. For Ollama, the local default is usually:
# http://localhost:11434/v1
#
# SINGLE_MODEL_MODE: run planner + executor against the same model/instance.
# This is the simplest setup if you are running only one local model.

SINGLE_MODEL_MODE = True

PLANNER_API_URL = "http://localhost:11434/v1"      # Ollama-compatible API
EXECUTOR_API_URL = "http://localhost:11434/v1" if SINGLE_MODEL_MODE else "http://localhost:11434/v1"
LMSTUDIO_API_TOKEN = ""  # set only if your local provider requires an auth token

# ─── Model Names ────────────────────────────────────────────────────────────
# Must match the model name shown by your local backend.
# Examples for Ollama: "llama3.2", "qwen2.5:7b", "mistral:7b"

PLANNER_MODEL = "qwen2.5-coder:14b"
EXECUTOR_MODEL = PLANNER_MODEL if SINGLE_MODEL_MODE else "qwen2.5-coder:14b"

# ─── LM Studio Request Settings ─────────────────────────────────────────────

PLANNER_TEMPERATURE = 0.4
EXECUTOR_TEMPERATURE = 0.2
REQUEST_TIMEOUT = 120                              # seconds per API call
MAX_TOKENS = 32768

# ─── Orchestrator Loop Settings ─────────────────────────────────────────────

MAX_ITERATIONS = 50                               # hard safety cap
MAX_TASK_RETRIES = 3                              # retries before marking FAILED
RESUME_PREVIOUS_SESSION = False                   # set True only if you intentionally want stale PENDING/RUNNING tasks restored

# ─── Terminal (tmux) Executor Settings ──────────────────────────────────────

TMUX_SESSION_NAME = "nyx"           # session created when not launched inside tmux
MAX_SHELL_STEPS_PER_TASK = 12      # commands the executor may run before giving up
SHELL_COMMAND_TIMEOUT = 120        # seconds to wait for a single command

# ─── Optional SSH execution ────────────────────────────────────────────────

SSH_DEFAULT_USER = "root"
SSH_DEFAULT_OPTIONS = (
    "HostKeyAlgorithms=+ssh-rsa",
    "PubkeyAcceptedAlgorithms=+ssh-rsa",
)
SSH_CONNECT_TIMEOUT = 60

# ─── Database ────────────────────────────────────────────────────────────────

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TASK_DB_PATH = os.path.join(DATA_DIR, "task_history.db")
CONTROL_DB_PATH = os.path.join(DATA_DIR, "orchestrator_control.db")
HTML_REPORT_PATH = os.path.join(DATA_DIR, "nyx_report.html")

# ─── Logging ─────────────────────────────────────────────────────────────────

LOG_LEVEL = "INFO"                                # DEBUG | INFO | WARNING | ERROR
LOG_TO_FILE = True
LOG_FILE = os.path.join(DATA_DIR, "orchestrator.log")