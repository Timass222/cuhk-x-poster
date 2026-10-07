# Environment files

A Docker image is preferred: place the `Dockerfile` here, and provide the built image as a `docker save` archive or a pinned registry tag (record it in `manifest.yaml` → `environment.docker_image`). Otherwise provide one reproducible environment specification, such as `requirements.txt`, `environment.yml`, or a package-manager lock file. Pin exact versions wherever possible, and document the build or setup command in the root README.

Also document non-Python dependencies, CUDA or accelerator requirements, compilers, system packages, and licences. Do not include credentials or a prebuilt virtual environment.

