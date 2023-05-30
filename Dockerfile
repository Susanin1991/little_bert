# Use an official PyTorch runtime as the base image
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /opt/ml/code

# Copy the code into the container
COPY . /opt/ml/code

# Install the required dependencies
RUN pip install -r requirements.txt

# Set the entry point to the training script
ENV SAGEMAKER_PROGRAM distil_bert.py

# Set the command to run the training script
CMD ["python", "-u", "distil_bert.py"]