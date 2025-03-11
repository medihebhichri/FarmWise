import os
import pytest
import tempfile
import yaml
import json
from unittest.mock import patch, MagicMock, mock_open

# Import the PlantDiseaseDetector class
import sys

sys.path.append(".")
from plant_disease_detector import PlantDiseaseDetector


class TestPlantDiseaseDetector:
    """Tests for the PlantDiseaseDetector class."""

    @pytest.fixture
    def mock_config(self):
        """Fixture for test configuration."""
        return {
            "model": {
                "path": "test_model.pt",
                "confidence_threshold": 0.25,
                "version": "test-v1.0"
            },
            "output": {
                "directory": "test_results",
                "save_detections": True
            },
            "inference": {
                "batch_size": 1,
                "image_size": 640
            }
        }

    @pytest.fixture
    def mock_config_file(self, mock_config, tmp_path):
        """Create a temporary config file for testing."""
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(mock_config, f)
        return str(config_path)

    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.create_experiment')
    @patch('mlflow.set_experiment')
    @patch('logging.FileHandler')
    @patch('logging.StreamHandler')
    def test_initialization(self, mock_stream, mock_file, mock_set_exp,
                            mock_create_exp, mock_get_exp, mock_tracking,
                            mock_config_file):
        """Test detector initialization."""
        # Mock mlflow.get_experiment_by_name to return None
        mock_get_exp.return_value = None

        # Initialize detector with test config
        detector = PlantDiseaseDetector(config_path=mock_config_file)

        # Check that MLflow was properly set up
        mock_tracking.assert_called_once()
        mock_get_exp.assert_called_once()
        mock_create_exp.assert_called_once()
        mock_set_exp.assert_called_once()

        # Check config was loaded correctly
        assert detector.config["model"]["path"] == "test_model.pt"
        assert detector.config["model"]["version"] == "test-v1.0"

        # Check device setup
        assert "cuda" in str(detector.device) or "cpu" in str(detector.device)

    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_dict')
    @patch('mlflow.log_artifact')
    def test_load_model(self, mock_log_artifact, mock_log_dict, mock_log_param,
                        mock_start_run, mock_set_exp, mock_get_exp, mock_tracking):
        """Test model loading."""
        # Create mock run context
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context

        # Mock YOLO class
        with patch('plant_disease_detector.YOLO') as mock_yolo:
            # Set up mock model
            mock_model = MagicMock()
            mock_model.names = {0: "healthy", 1: "disease_1", 2: "disease_2"}
            mock_yolo.return_value = mock_model

            # Create detector with temporary config
            with tempfile.NamedTemporaryFile(mode='w') as temp_config:
                yaml.dump({
                    "model": {"path": "test_model.pt", "version": "test-v1"}
                }, temp_config)
                temp_config.flush()

                detector = PlantDiseaseDetector(config_path=temp_config.name)

                # Test model loading
                detector.load_model("test_path.pt")

                # Check YOLO was called correctly
                mock_yolo.assert_called_once_with("test_path.pt")

                # Check model was moved to device
                mock_model.to.assert_called_once()

                # Verify MLflow logging
                mock_start_run.assert_called_once()
                assert mock_log_param.call_count >= 3  # At least model_path, version, device
                mock_log_dict.assert_called_once()
                mock_log_artifact.assert_called_once_with("test_path.pt")

    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('mlflow.log_param')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_dict')
    def test_predict(self, mock_log_dict, mock_log_metric, mock_log_param,
                     mock_start_run, mock_set_exp, mock_get_exp, mock_tracking):
        """Test model prediction."""
        # Create mock run context
        mock_run_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run_context

        # Create temp directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create detector with temp config
            with tempfile.NamedTemporaryFile(mode='w') as temp_config:
                config = {
                    "model": {"path": "test_model.pt", "version": "test-v1", "confidence_threshold": 0.5},
                    "output": {"directory": temp_dir}
                }
                yaml.dump(config, temp_config)
                temp_config.flush()

                detector = PlantDiseaseDetector(config_path=temp_config.name)

                # Set up mock model
                mock_model = MagicMock()
                mock_model.names = {0: "healthy", 1: "disease_1"}

                # Set up mock results
                mock_box = MagicMock()
                mock_box.cls = [0]
                mock_box.conf = [0.95]

                mock_result = MagicMock()
                mock_result.boxes = [mock_box]

                mock_model.predict.return_value = [mock_result]
                detector.model = mock_model
                detector.class_names = mock_model.names

                # Test prediction
                results = detector.predict("test_image.jpg", run_name="test_run")

                # Check model was called correctly
                mock_model.predict.assert_called_once()
                args, kwargs = mock_model.predict.call_args
                assert kwargs["conf"] == 0.5
                assert kwargs["save"] is True

                # Verify MLflow logging
                mock_start_run.assert_called_once_with(run_name="test_run")
                assert mock_log_param.call_count >= 2  # At least source_type and source_path
                assert mock_log_metric.call_count >= 1  # At least inference_time
                mock_log_dict.assert_called_once()  # class_distribution

                # Check results are returned
                assert results == [mock_result]

    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.get_experiment_by_name')
    @patch('mlflow.set_experiment')
    @patch('os.path.exists')
    def test_export_metrics(self, mock_exists, mock_set_exp, mock_get_exp, mock_tracking):
        """Test metrics export functionality."""
        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "test_id"
        mock_get_exp.return_value = mock_experiment

        # Mock runs data
        mock_runs_data = {
            "run_id": ["run1", "run2"],
            "start_time": ["2023-01-01", "2023-01-02"],
            "metrics.inference_time": [0.1, 0.2],
            "metrics.detection_count": [5, 10],
            "metrics.average_confidence": [0.9, 0.8],
            "params.source_type": ["file", "file"],
            "params.confidence_threshold": [0.25, 0.3]
        }

        import pandas as pd
        mock_runs = pd.DataFrame(mock_runs_data)

        # Mock mlflow.search_runs to return our dataframe
        with patch('mlflow.search_runs', return_value=mock_runs):
            # Mock open to capture file writing
            mock_file = mock_open()
            with patch('builtins.open', mock_file):
                # Create detector
                with tempfile.NamedTemporaryFile(mode='w') as temp_config:
                    temp_config.write("{}")
                    temp_config.flush()

                    detector = PlantDiseaseDetector(config_path=temp_config.name)

                    # Test export_metrics
                    metrics = detector.export_metrics("test_metrics.json")

                    # Check metrics format
                    assert metrics["total_runs"] == 2
                    assert len(metrics["inference_times"]) == 2
                    assert metrics["avg_inference_time"] == 0.15
                    assert metrics["avg_detection_count"] == 7.5
                    assert metrics["overall_avg_confidence"] == 0.85

                    # Verify file was written
                    mock_file.assert_called_once_with('test_metrics.json', 'w')
                    handle = mock_file()
                    handle.write.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-v", "test_detector.py"])