# syntax=docker/dockerfile:1
FROM python:3.12-slim
ARG LITELLM_VERSION=1.104.2
RUN pip install --no-cache-dir "litellm[proxy]==${LITELLM_VERSION}"
EXPOSE 4000
ENTRYPOINT ["litellm"]
CMD ["--host", "0.0.0.0", "--port", "4000", "--num_workers", "1"]
