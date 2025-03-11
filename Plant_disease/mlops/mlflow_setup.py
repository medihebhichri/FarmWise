"""
MLflow server setup and utilities for the Plant Disease Detection MLOps system.
Run this script to start a local MLflow server.
"""

import os
import subprocess
import argparse
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mlflow_setup")


def setup_mlflow_directories():
    """Set up required directories for MLflow."""
    # Create directories
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("mlflow-artifacts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logger.info("MLflow directories created successfully")


def start_mlflow_server(host="127.0.0.1", port=5000, backend_store="mlruns",
                        artifact_store="mlflow-artifacts"):
    """
    Start a local MLflow tracking server.

    Args:
        host (str): Host to run the server on
        port (int): Port to run the server on
        backend_store (str): Path to the backend store
        artifact_store (str): Path to the artifact store
    """
    # Ensure directories exist
    setup_mlflow_directories()

    # Build command
    cmd = [
        "mlflow", "server",
        "--host", host,
        "--port", str(port),
        "--backend-store-uri", backend_store,
        "--default-artifact-root", artifact_store
    ]

    logger.info(f"Starting MLflow server with command: {' '.join(cmd)}")

    try:
        # Run the server
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # Wait a bit to ensure server starts
        time.sleep(2)

        if process.poll() is None:
            logger.info(f"MLflow server is running at http://{host}:{port}")
            logger.info("Press Ctrl+C to stop the server")

            # Stream logs until interrupted
            try:
                for line in process.stdout:
                    print(line, end="")
            except KeyboardInterrupt:
                logger.info("Shutting down MLflow server...")
                process.terminate()
                process.wait()
                logger.info("MLflow server stopped")
        else:
            # If process terminated, print output
            stdout, _ = process.communicate()
            logger.error(f"MLflow server failed to start:\n{stdout}")

    except Exception as e:
        logger.error(f"Error starting MLflow server: {e}")


def check_mlflow_experiments():
    """Check and display existing MLflow experiments."""
    try:
        import mlflow

        # Set tracking URI to local mlruns directory
        mlflow.set_tracking_uri("file:./mlruns")

        # Get all experiments
        experiments = mlflow.search_experiments()

        print("\n=== MLflow Experiments ===")
        if not experiments:
            print("No experiments found.")
        else:
            for exp in experiments:
                print(f"ID: {exp.experiment_id}, Name: {exp.name}")

                # Get all runs for this experiment
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
                print(f"  Total runs: {len(runs)}")

                # Print run summary if runs exist
                if len(runs) > 0:
                    print("  Latest runs:")
                    for idx, (_, run) in enumerate(runs.sort_values("start_time", ascending=False).head(3).iterrows()):
                        print(f"    - {run['run_id'][:8]}... ({run['start_time']})")

        print("=======================")

    except Exception as e:
        logger.error(f"Error checking MLflow experiments: {e}")


def main():
    """Main function to run the MLflow setup script."""
    parser = argparse.ArgumentParser(description="MLflow Server Management")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--check-experiments", action="store_true", help="Check existing experiments")

    args = parser.parse_args()

    if args.check_experiments:
        check_mlflow_experiments()
    else:
        start_mlflow_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()