FROM tensorflow/serving:2.13.0

ENV MODEL_NAME=adult_model
ENV MODEL_BASE_PATH=/models

ENV PORT=8080

COPY yrbror-pipeline/serving_model_dir /models/${MODEL_NAME}

EXPOSE 8080
EXPOSE 8500

RUN printf '#!/bin/sh\n\n\
echo "Starting TensorFlow Serving..."\n\
echo "MODEL_NAME=${MODEL_NAME}"\n\
echo "MODEL_BASE_PATH=${MODEL_BASE_PATH}"\n\
echo "PORT=${PORT}"\n\n\
tensorflow_model_server \\\n\
  --rest_api_port=${PORT} \\\n\
  --model_name=${MODEL_NAME} \\\n\
  --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}\n' \
  > /usr/bin/tf_serving_entrypoint.sh \
  && chmod +x /usr/bin/tf_serving_entrypoint.sh

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]
