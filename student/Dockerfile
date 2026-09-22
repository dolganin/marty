FROM arena-base

LABEL org.opencontainers.image.version="0.16.0"

USER root
WORKDIR /submission
COPY . /submission/

RUN command -v g++ >/dev/null 2>&1 && \
    test -f "$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')/Python.h" || \
    (apt-get update && apt-get install -y --no-install-recommends g++ python3-dev && \
     rm -rf /var/lib/apt/lists/*)
RUN python -c "import pybind11" >/dev/null 2>&1 || \
    python -m pip install --no-cache-dir "pybind11==3.1.0"
RUN python -m pip install --no-cache-dir "stable-baselines3>=2.3,<3"

RUN python -m pip install --no-build-isolation --no-deps --force-reinstall /submission
RUN python -m unittest discover -s /submission/tests -v
RUN mkdir -p /tmp/submission-smoke-output && \
    timeout --signal=TERM --kill-after=10s 180s env \
      MARS_ROVER_SMOKE_TEST=1 \
      MARS_ROVER_OUTPUT_PATH=/tmp/submission-smoke-output/policy.zip \
      OMP_NUM_THREADS=2 \
      python /submission/train.py && \
    python /submission/check_policy.py /tmp/submission-smoke-output/policy.zip && \
    rm -rf /tmp/submission-smoke-output /tmp/training

USER 10000:10000
CMD ["python", "/submission/train.py"]
