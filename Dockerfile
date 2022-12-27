FROM python:3.9

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
ENV TORCH_HOME=/app

RUN bash -c "curl https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt > /app/imagenet_classes.txt"
RUN python -c "import torch; torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)"

