### Build Stage
docker build -t unsloth_train . # build the docker image
### Validate
echo $HF_HOME # check if the HF_HOME is set
### Run Stage
docker run --gpus all --rm -it --env-file .env -v "$(pwd):/usr/src/app" -v "$HF_HOME:/huggingface" unsloth_train # run the docker image