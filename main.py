# main.py
import os
import logging
import warnings


def setup_quiet_console():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "1")
    logging.disable(logging.CRITICAL)


if __name__ == "__main__":
    setup_quiet_console()

    from src.train import run_logreg_c_sweep
    from src.config import MLFLOW_UI_HOST, MLFLOW_UI_PORT, AUTO_START_MLFLOW_UI
    from src.mlflow_ui import start_mlflow_ui

    run_logreg_c_sweep(
        c_values=[0.05, 0.2, 0.85, 1.0, 1.15, 5.0, 20.0],
        solver="liblinear",
        run_prefix="lab1_logreg_C",
        run_group="lab1_c_sweep",
    )

    if AUTO_START_MLFLOW_UI:
        start_mlflow_ui(
            host=MLFLOW_UI_HOST,
            port=MLFLOW_UI_PORT,
            open_browser=True,
            quiet=True,
            wait=True,
        )