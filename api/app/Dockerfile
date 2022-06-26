FROM python:3.9-bullseye

LABEL maintainer="Pooya Mohammadi <https://github.com/pooya-mohammadi>"

# os and opencv libraries
RUN apt-get update -y \
  && apt install libgl1-mesa-glx -y \
  && apt-get install 'ffmpeg' 'libsm6' 'libxext6'  -y \
  && python -m pip install --no-cache-dir --upgrade pip

# fixed python libraries for pytorch models
RUN pip install --no-cache-dir torch==1.10.0+cpu torchvision==0.11.1+cpu \
  -f https://download.pytorch.org/whl/torch_stable.html \
  && pip install --no-cache-dir numpy==1.21.4 \
  && pip install --no-cache-dir opencv-python==4.5.4.58 \
  && pip install --no-cache-dir scikit-learn==1.0.1 \
  && pip install --no-cache-dir matplotlib==3.4.3 \
  && pip install --no-cache-dir pandas==1.3.4 \
  && pip install --no-cache-dir seaborn==0.11.2 \
  && pip install --no-cache-dir scipy==1.7.2 \
  && pip install --no-cache-dir scikit-image==0.18.3 \
  && pip install --no-cache-dir opencv-python==4.6.0.66 \
  && pip install --no-cache-dir pillow==8.4.0 \
  && pip install --no-cache-dir albumentations==1.1.0 \
  && rm -rf /root/.cache/pip

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt
