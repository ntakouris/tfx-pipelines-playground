# Lint as: python2, python3
# Copyright 2020 Theodoros Ntakouris
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
"""Chicago taxi example using Fluent TFX."""

import os
from typing import List, Text

import absl
import fluent_tfx as ftfx

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

_pipeline_name = 'chicago_taxi_pipeline'

_taxi_root = os.path.dirname(__file__)
_data_root = os.path.join(_taxi_root, 'data', 'big_tipper_label')

_module_file = os.path.join(_taxi_root, 'model.py')

_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]

_eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(signature_name='serving_default', label_key='big_tipper')],
      slicing_specs=[
          tfma.SlicingSpec(),
          tfma.SlicingSpec(feature_keys=['trip_start_hour'])
      ],
      metrics_specs=[
          tfma.MetricsSpec(
              thresholds={
                  'binary_accuracy':
                      tfma.config.MetricThreshold(
                          value_threshold=tfma.GenericValueThreshold(
                              lower_bound={'value': 0.5}),
                          change_threshold=tfma.GenericChangeThreshold(
                              direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                              absolute={'value': -1e-10}))
              })
      ])

def _create_pipeline(pipeline_name: Text, data_root: Text,
                     module_file: Text,
                     beam_pipeline_args: List[Text]) -> pipeline.Pipeline:

  pipeline_def = ftfx.PipelineDef(name=pipeline_name) \
    .with_beam_pipeline_args(beam_pipeline_args) \
    .cache(False) \
    .with_sqlite_ml_metadata() \
    .from_csv(data_root) \
    .generate_statistics() \
    .infer_schema(infer_feature_shape=False) \
    .validate_input_data() \
    .preprocess(module_file=module_file) \
    .tune(
      module_file=module_file,
      train_args=trainer_pb2.TrainArgs(num_steps=20),
      eval_args=trainer_pb2.EvalArgs(num_steps=5)) \
    .train(
      module_file=module_file,
      train_args=trainer_pb2.TrainArgs(num_steps=1000),
      eval_args=trainer_pb2.EvalArgs(num_steps=150)) \
    .evaluate_model(_eval_config) \
    .push_to(relative_push_uri='models_for_serving')
  # you can also do .infra_validate() with docker installed
  # and .with_imported_schema(uri) instead of importernode 
  return pipeline_def.build()


# .tune(
#   module_file=module_file,
#   train_args=trainer_pb2.TrainArgs(num_steps=20),
#   eval_args=trainer_pb2.EvalArgs(num_steps=5),
#   example_input=ftfx.ExampleInputs.PREPROCESSED_EXAMPLES ) \

# To run this pipeline from the python CLI:
#   $python pipeline_fluent_tfx.py
if __name__ == '__main__':
  absl.logging.set_verbosity(absl.logging.INFO)

  BeamDagRunner().run(
      _create_pipeline(
          pipeline_name=_pipeline_name,
          data_root=_data_root,
          module_file=_module_file,
          beam_pipeline_args=_beam_pipeline_args))
