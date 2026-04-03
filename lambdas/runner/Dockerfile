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
    pyarrow==17.0.0

# Copy runner handler and profile modules (build context is lambdas/)
COPY runner/handler.py ${LAMBDA_TASK_ROOT}/
COPY profiles/ ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
