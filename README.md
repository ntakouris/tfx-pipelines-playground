# TFX Pipelines Playground

This is basically the TFX Example of chicago taxi trips pipeline with some extra stuff going on like Tuning and some notebooks for visualization.

## Repository Contents

The pipeline that is implemented is the: [TFX Chicago Taxi Pipeline Example](https://github.com/tensorflow/tfx/tree/master/tfx/examples/chicago_taxi_pipeline), slightly modified.

| File                                       | Contents                                                                                       |
| :----------------------------------------- | :--------------------------------------------------------------------------------------------- |
| `data`                                     | The CSV raw data for the taxi pipeline example                                                 |
| `pipeline.py`                              | The full, vanilla TFX pipeline declaration and component configuration                         |
| `model.py`                                 | Preprocessing, Model Creation, Training, Hyperparameter tuning                                 |
| `example_data_visualization.ipynb`         | ExampleValidator and StatisticsGen: Raw data and schema visualization notebook through TFDV    |
| `model_analysis_visualization.ipynb`       | Evaluator TFMA result visualization notebook                                                   |
| `tensorboard_training_visualization.ipynb` | Trainer training logs visualization notebook                                                   |
| `tf_serving_communication.ipynb`           | Parse a feature dict to tf.Example and perform a POST request to obtain predictions (notebook) |
| `mlmd_artifacts_from_pipeline_info.ipynb`  | Query the MLMD store by pipeline info to get all artifacts by type                             |
| `launch_model_local.sh`                    | Utility for quick launch of a tf serving docker container                                      |
