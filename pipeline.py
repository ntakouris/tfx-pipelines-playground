# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
# Modifications Copyright 2020 Theodoros Ntakouris
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Chicago taxi example using TFX."""

import os
import requests, json
import tensorflow as tf
from typing import List, Text, Optional, Any

import absl
from absl import logging
import tensorflow_model_analysis as tfma

from tfx.components import CsvExampleGen, Evaluator, ExampleValidator, Pusher, ResolverNode, \
  SchemaGen, StatisticsGen, Trainer, Transform, InfraValidator
from tfx.components.tuner.component import Tuner
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import pusher_pb2, trainer_pb2, infra_validator_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.utils.dsl_utils import external_input

from tfx.dsl.component.experimental.annotations import InputArtifact, Parameter, OutputDict
from tfx.dsl.component.experimental.decorators import component
from tfx.types.standard_artifacts import Model, ExampleAnomalies, ModelEvaluation

_pipeline_name = 'chicago_taxi_pipeline'

_taxi_root = os.path.dirname(__file__)
_data_root = os.path.join(_taxi_root, 'data', 'big_tipper_label')

_module_file = os.path.join(_taxi_root, 'model.py')

_tfx_root = os.path.join(os.path.dirname(__file__), 'tfx')
_pipeline_root = os.path.join(_tfx_root, _pipeline_name)

_metadata_path = os.path.join(_taxi_root, 'metadata.db')

_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]

_eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='big_tipper')],
    slicing_specs=[
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=['trip_start_hour'])
    ],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            tfma.MetricConfig(
                class_name='BinaryAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value': 0.5}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value': -1e-10})))
        ])
    ],
)


@component
def SlackNotifierComponent(
    example_anomalies: InputArtifact[ExampleAnomalies],
    model: InputArtifact[Model],
    model_evaluation: InputArtifact[ModelEvaluation],
    slack_url: Parameter[Text] = None
) -> None:
    logging.info(f'Running SlackNotifierComponent for slack_url: {slack_url}')
    example_anomalies_uri = example_anomalies.uri  # /anomalies.pbtxt
    model_uri = model.uri  # /logs or /serving_model_dir. For TFX >= 0.22 use ModelRun for logs
    model_eval_uri = model_evaluation.uri  # /metrics or /plots or /validations

    if slack_url is None:
        logging.warn(
            'SlackNotifierComponent: slack_url is none. Skipping component execution'
        )
        return None

    slack_data = {
        'text':
        f':spaghetti: \n anomalies: {example_anomalies_uri} \n model: {model_uri} \n eval results: {model_eval_uri} \n'
    }

    response = requests.post(slack_url,
                             data=json.dumps(slack_data),
                             headers={'Content-Type': 'application/json'})

    if response.status_code != 200:
        logging.warn(f'Status Code: {response.status_code} \n {response.text}')

def _create_pipeline(pipeline_name: Text,
                     pipeline_root: Text,
                     data_root: Text,
                     module_file: Text,
                     serving_model_dir: Text,
                     metadata_path: Text,
                     beam_pipeline_args: List[Text],
                     slack_url: Text = None,
                     enable_cache: bool = True) -> pipeline.Pipeline:
    examples = external_input(data_root)

    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = CsvExampleGen(input=examples)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

    # Generates schema based on statistics files.
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
                           infer_feature_shape=True)

    # or schema_gen = ImporterNode(
    # instance_name='schema_importer',
    # source_uri=<uri>,
    # artifact_type=standard_artifacts.Schema)

    # Performs anomaly detection based on statistics and data schema.
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema'])

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(examples=example_gen.outputs['examples'],
                          schema=schema_gen.outputs['schema'],
                          module_file=module_file)

    tuner = Tuner(module_file=module_file,
                  examples=transform.outputs['transformed_examples'],
                  transform_graph=transform.outputs['transform_graph'],
                  train_args=trainer_pb2.TrainArgs(num_steps=20),
                  eval_args=trainer_pb2.EvalArgs(num_steps=5))

    trainer = Trainer(
        module_file=module_file,
        hyperparameters=tuner.outputs['best_hyperparameters'],
        custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
        transformed_examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=2000),
        eval_args=trainer_pb2.EvalArgs(num_steps=150))

    # Get the latest blessed model for model validation.
    model_resolver = ResolverNode(
        instance_name='latest_blessed_model_resolver',
        resolver_class=latest_blessed_model_resolver.
        LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing))

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        # Change threshold will be ignored if there is no baseline (first run).
        eval_config=_eval_config)

    # Performs infra validation of a candidate model to prevent unservable model
    # from being pushed.

    # Setup docker first and then uncomment these lines

    # infra_validator = InfraValidator(
    #     model=trainer.outputs['model'],
    #     examples=example_gen.outputs['examples'],
    #     serving_spec=infra_validator_pb2.ServingSpec(
    #         tensorflow_serving=infra_validator_pb2.TensorFlowServing(
    #             tags=['latest']),
    #         local_docker=infra_validator_pb2.LocalDockerConfig()),
    #     request_spec=infra_validator_pb2.RequestSpec(
    #         tensorflow_serving=infra_validator_pb2.TensorFlowServingRequestSpec()),
    #     validation_spec=infra_validator_pb2.ValidationSpec(
    #       max_loading_time_seconds=20,
    #       num_tries=2
    #   )
    # )

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        # infra_blessing=infra_validator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    slack_component = SlackNotifierComponent(
      example_anomalies=example_validator.outputs['anomalies'],
      model=trainer.outputs['model'],
      model_evaluation=evaluator.outputs['evaluation'],
      slack_url=slack_url
    )

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            tuner,
            trainer,
            model_resolver,
            evaluator,
            # infra_validator,
            pusher,
            slack_component
        ],
        enable_cache=enable_cache,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path),
        beam_pipeline_args=beam_pipeline_args)


# To run this pipeline from the python CLI:
#   $python pipeline.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)

  BeamDagRunner().run(
      _create_pipeline(pipeline_name=_pipeline_name,
                        pipeline_root=_pipeline_root,
                        serving_model_dir=os.path.join(
                            os.path.dirname(__file__), 'models_for_serving'),
                        data_root=_data_root,
                        module_file=_module_file,
                        metadata_path=_metadata_path,
                        slack_url=os.environ.get('SLACK_URL'),
                        beam_pipeline_args=_beam_pipeline_args,
                        enable_cache=True))
