FROM public.ecr.aws/lambda/python:3.12

# Install heavy analytics dependencies (too large for Lambda layers)
RUN pip install --no-cache-dir \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    numpy==1.26.4 \
    statsmodels==0.14.4 \
    prophet==1.1.6 \
    lifelines==0.29.0 \
    scipy==1.14.1 \
    pyarrow==17.0.0 \
    vaderSentiment==3.3.2 \
    ruptures==1.1.9 \
    shapely==2.0.6 \
    networkx==3.4.2 \
    python-louvain==0.16 \
    xarray==2024.10.0 \
    netCDF4==1.7.2 \
    pypdf==5.1.0 \
    RestrictedPython==7.3 \
    girth==0.8.0

# Copy runner handler and profile modules (build context is lambdas/)
COPY runner/handler.py ${LAMBDA_TASK_ROOT}/
COPY profiles/ ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
