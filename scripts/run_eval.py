"""Run the baseline or the agent on the evaluation set. See `agentic_rag.cli`.

    python scripts/run_eval.py --system agent --provider openai --model <model>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rag.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
