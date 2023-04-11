from timeit import default_timer as timer
from typing import Any

import config.general_settings as cfg

from src.pipeline.pipeline import FitPipelineParameters, Pipeline
from src.pipeline.steps.calculate_evaluation_measures_step import CalculateEvaluationMeasuresStep
from src.pipeline.steps.calculate_predicted_risk_rates import CalculatePredictedRiskRatesStep
from src.pipeline.steps.calculate_statistical_risk_rates_step import CalculateStatisticalRiskRatesStep
from src.pipeline.steps.create_forecasted_datasets_step import CreateForecastedDatasetsStep
from src.pipeline.steps.create_results_output_dataset_step import CreateResultsOutputDatasetStep
from src.pipeline.steps.create_training_assets_step import CreateTrainingAssetsStep
from src.pipeline.steps.deploy_model_step import DeployModelStep
from src.pipeline.steps.predict_step import PredictStep
from src.pipeline.steps.read_prepare_training_data_step import ReadPrepareTrainingDataStep
from src.pipeline.steps.retrain_step import RetrainStep
from src.pipeline.steps.select_sampler_classifier_step import SelectSamplerClassifierStep
from src.pipeline.steps.select_scaler_forecaster_step import SelectScalerForecasterStep
from src.pipeline.steps.select_thresholds_step import SelectThresholdsStep


class FitPipeline(Pipeline):
    """The FitPipeline entity"""

    def run(self, pipeline_parameters: FitPipelineParameters) -> Any:
        """
        Run pipeline, according to the sequential steps defined
        :param pipeline_parameters: The parameters for each run of the pipeline
        :return: the evaluation metrics dataset, the scaler, the classifier and the forecaster
        """
        start = timer()

        prepare_data_step = ReadPrepareTrainingDataStep()

        create_assets_step = CreateTrainingAssetsStep(
            prepare_data_step.step_output.dataset
        )

        select_scaler_forecaster_step = SelectScalerForecasterStep(
            pipeline_parameters.scaling_methods,
            pipeline_parameters.forecasting_algorithms,
            create_assets_step.step_output.X_train,
            create_assets_step.step_output.X_validation
        )

        select_sampler_classifier_step = SelectSamplerClassifierStep(
            pipeline_parameters.sampling_methods,
            pipeline_parameters.classification_algorithms,
            select_scaler_forecaster_step.step_output.scaled_X_train,
            create_assets_step.step_output.y_train,
            select_scaler_forecaster_step.step_output.scaled_X_validation,
            create_assets_step.step_output.y_validation
        )

        # First time, train vs validation
        create_forecasted_datasets_step = CreateForecastedDatasetsStep(
            select_scaler_forecaster_step.step_output.scaler,
            select_scaler_forecaster_step.step_output.forecaster,
            select_scaler_forecaster_step.step_output.scaled_X_train,
            create_assets_step.step_output.X_validation
        )

        # First time, train vs validation
        predict_step = PredictStep(
            select_sampler_classifier_step.step_output.classifier,
            create_forecasted_datasets_step.step_output.scaled_X_test,
            create_forecasted_datasets_step.step_output.forecasted_scaled_X_tests
        )

        select_thresholds_step = SelectThresholdsStep(
            prepare_data_step.step_output.dataset,
            predict_step.step_output.predicted_dataset
        )

        retrain_step = RetrainStep(
            select_scaler_forecaster_step.step_output.scaler,
            select_scaler_forecaster_step.step_output.forecaster,
            select_sampler_classifier_step.step_output.sampler,
            select_sampler_classifier_step.step_output.classifier,
            create_assets_step.step_output.X_train_validation,
            create_assets_step.step_output.y_train_validation
        )

        # Second time, train_validation vs test
        create_forecasted_datasets_step = CreateForecastedDatasetsStep(
            retrain_step.step_output.scaler,
            retrain_step.step_output.forecaster,
            retrain_step.step_output.scaled_X_train_validation,
            create_assets_step.step_output.X_test
        )

        # Second time, train_validation vs test
        predict_step = PredictStep(
            retrain_step.step_output.classifier,
            create_forecasted_datasets_step.step_output.scaled_X_test,
            create_forecasted_datasets_step.step_output.forecasted_scaled_X_tests
        )

        calculate_predicted_risk_rates_step = CalculatePredictedRiskRatesStep(
            prepare_data_step.step_output.dataset,
            predict_step.step_output.predicted_dataset,
            select_thresholds_step.step_output.thresholds
        )

        calculate_statistical_risk_rates_step = CalculateStatisticalRiskRatesStep(
            prepare_data_step.step_output.dataset,
            create_assets_step.step_output.X_test,
            create_forecasted_datasets_step.step_output.forecasted_X_tests
        )

        create_output_dataset_step = CreateResultsOutputDatasetStep(
            prepare_data_step.step_output.dataset,
            predict_step.step_output.predicted_dataset,
            calculate_predicted_risk_rates_step.step_output.predicted_risk_rates_dataset,
            calculate_statistical_risk_rates_step.step_output.statistical_risk_rates_dataset
        )

        create_output_dataset_step.step_output.output_dataset.to_csv(cfg.DATA_GENERATED_DIR + "fit_output_dataset.csv")

        calculate_evaluation_measures_step = CalculateEvaluationMeasuresStep(
            create_output_dataset_step.step_output.output_dataset
        )

        DeployModelStep(
            retrain_step.step_output.scaler,
            retrain_step.step_output.forecaster,
            retrain_step.step_output.sampler,
            retrain_step.step_output.classifier,
            select_thresholds_step.step_output.thresholds
        )

        end = timer()

        self.save_elapsed_time(start, end)

        return calculate_evaluation_measures_step
