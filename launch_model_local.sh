# example usage
# for a model thats under /mymodel/{version_number}/<binaries>
# bash launch_model_local.sh $(pwd)/models_for_serving mymodel
#                               ^ path                  ^ name

echo 'Serving model path ' $1 ' with name ' $2
echo '------------' $'\n\n'

# 8500 -> gRPC
# 8501 -> HTTP

# http endpoints (https://www.tensorflow.org/tfx/serving/api_rest)

# curl -X GET http://localhost:8501/v1/models/mymodel/<optional>/metadata
# optional = /labels/<label> or /version/<version>

docker run -p 8501:8501 \
--mount type=bind,source=$1,target=/models/$2 \
-e MODEL_NAME=$2 -it --rm tensorflow/serving
