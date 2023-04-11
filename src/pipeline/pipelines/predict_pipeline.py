from timeit import default_timer as timer
from typing import Any

import pandas as pd

import config.general_settings as cfg

from src.pipeline.pipeline import Pipeline, PredictPipelineParameters
from src.pipeline.steps.calculate_predicted_risk_rates import CalculatePredictedRiskRatesStep
from src.pipeline.steps.calculate_statistical_risk_rates_step import CalculateStatisticalRiskRatesStep
from src.pipeline.steps.create_forecasted_datasets_step import CreateForecastedDatasetsStep
from src.pipeline.steps.create_results_output_dataset_step import CreateResultsOutputDatasetStep
from src.pipeline.steps.load_model_step import LoadModelStep
from src.pipeline.steps.predict_step import PredictStep
from src.pipeline.steps.read_prediction_data_step import ReadPredictionDataStep


class PredictPipeline(Pipeline):
    """The PredictPipeline entity"""

    def run(self, pipeline_parameters: PredictPipelineParameters) -> Any:
        """
        Run pipeline, according to the sequential steps defined
        :param pipeline_parameters: The parameters for each run of the pipeline
        """
        start = timer()

        read_data_step = ReadPredictionDataStep()

        load_model_step = LoadModelStep()

        create_forecasted_datasets_step = CreateForecastedDatasetsStep(
            load_model_step.step_output.scaler,
            load_model_step.step_output.forecaster,
            pd.DataFrame(),
            read_data_step.step_output.dataset
        )

        predict_step = PredictStep(
            load_model_step.step_output.classifier,
            create_forecasted_datasets_step.step_output.scaled_X_test,
            create_forecasted_datasets_step.step_output.forecasted_scaled_X_tests
        )

        calculate_predicted_risk_rates_step = CalculatePredictedRiskRatesStep(
            read_data_step.step_output.dataset,
            predict_step.step_output.predicted_dataset,
            load_model_step.step_output.thresholds
        )

        calculate_statistical_risk_rates_step = CalculateStatisticalRiskRatesStep(
            read_data_step.step_output.dataset,
            read_data_step.step_output.dataset,
            create_forecasted_datasets_step.step_output.forecasted_X_tests
        )

        create_output_dataset_step = CreateResultsOutputDatasetStep(
            read_data_step.step_output.dataset,
            predict_step.step_output.predicted_dataset,
            calculate_predicted_risk_rates_step.step_output.predicted_risk_rates_dataset,
            calculate_statistical_risk_rates_step.step_output.statistical_risk_rates_dataset
        )

        create_output_dataset_step.step_output.output_dataset.to_csv(cfg.DATA_GENERATED_DIR + "predict_output_dataset.csv")

        end = timer()

        self.save_elapsed_time(start, end)
