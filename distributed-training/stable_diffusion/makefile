SHELL = /bin/bash # Use bash syntax
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source ~/miniconda3/etc/profile.d/conda.sh ; conda activate ; conda activate
#CONDA_CREATE=source $(conda info --base)/etc/profile.d/conda.sh ; conda create -n; conda create -n
MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATA_DIR="./dicoo/"

install-miniconda:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh
	
install-dependencies:
	conda create -n diffuser_icom python==3.8.10 && \
	$(CONDA_ACTIVATE) diffuser_icom && \
	pip3 install diffusers==0.24.0 && \
	pip3 install torch==2.0.0+cpu torchvision==0.15.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
	pip3 install transformers==4.36.0 accelerate==0.21.0 && \
	wget https://intel-extension-for-pytorch.s3.amazonaws.com/torch_ccl/cpu/oneccl_bind_pt-2.0.0%2Bcpu-cp38-cp38-linux_x86_64.whl && \
	pip3 install oneccl_bind_pt-2.0.0+cpu-cp38-cp38-linux_x86_64.whl && \
	pip3 install intel_extension_for_pytorch==2.0.0 && \
	pip3 install numpy==1.24.4

download-training-images:
	mkdir dicoo && \
	cd dicoo && \
	wget https://huggingface.co/sd-concepts-library/dicoo/resolve/main/concept_images/0.jpeg && \
	wget https://huggingface.co/sd-concepts-library/dicoo/resolve/main/concept_images/1.jpeg && \
	wget https://huggingface.co/sd-concepts-library/dicoo/resolve/main/concept_images/2.jpeg && \
	wget https://huggingface.co/sd-concepts-library/dicoo/resolve/main/concept_images/3.jpeg && \
	wget https://huggingface.co/sd-concepts-library/dicoo/resolve/main/concept_images/4.jpeg

single-node-test:
	$(CONDA_ACTIVATE) diffuser_icom && \
	python ./textual_inversion_icom.py                        \
	--pretrained_model_name_or_path=$(MODEL_NAME) --train_data_dir=$(DATA_DIR)                  \
	--learnable_property="object"   --placeholder_token="<dicoo>" --initializer_token="toy" \
	--resolution=512  --train_batch_size=1  --seed=7  --gradient_accumulation_steps=1       \
	--max_train_steps=5 --learning_rate=1e-6 --scale_lr --lr_scheduler="constant"     \
	--lr_warmup_steps=3 --output_dir=./textual_inversion_output     \
	--save_as_full_pipeline

setup-stable-diffusion-icom-master: install-dependencies download-training-images
