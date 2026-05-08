IMAGE_NAME  ?= side-train
IMAGE_TAG   ?= latest
CONTAINER   ?= side-train-dev
DATA_DIR    ?= $(PWD)/datasets
OUTPUT_DIR  ?= $(PWD)/outputs
CONFIG      ?= configs/dinov2_mask2former_crack.yaml

.PHONY: help build run shell train-seg train-ssl eval clean

help:
	@echo "Usage:"
	@echo "  make build          Build the Docker image"
	@echo "  make run            Start an interactive container"
	@echo "  make shell          Open a shell in a running container"
	@echo "  make download-data  Download public crack/sewer datasets"
	@echo "  make train-seg      Run Mask2Former segmentation fine-tuning"
	@echo "  make train-ssl      Run DINO SSL pretraining on Sewer-ML"
	@echo "  make eval           Run evaluation"
	@echo "  make clean          Remove stopped containers and dangling images"

build:
	docker build --platform linux/arm64 -t $(IMAGE_NAME):$(IMAGE_TAG) .

run:
	docker run --rm -it \
		--gpus all \
		--shm-size=32g \
		-v $(DATA_DIR):/workspace/datasets \
		-v $(OUTPUT_DIR):/workspace/outputs \
		-v $(PWD):/workspace \
		--name $(CONTAINER) \
		$(IMAGE_NAME):$(IMAGE_TAG)

shell:
	docker exec -it $(CONTAINER) bash

download-data:
	bash data/download_datasets.sh $(DATA_DIR)

train-seg:
	docker run --rm \
		--gpus all \
		--shm-size=32g \
		-v $(DATA_DIR):/workspace/datasets \
		-v $(OUTPUT_DIR):/workspace/outputs \
		-v $(PWD):/workspace \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		bash scripts/run_finetuning.sh $(CONFIG)

train-ssl:
	docker run --rm \
		--gpus all \
		--shm-size=32g \
		-v $(DATA_DIR):/workspace/datasets \
		-v $(OUTPUT_DIR):/workspace/outputs \
		-v $(PWD):/workspace \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		bash scripts/run_pretraining.sh

eval:
	docker run --rm \
		--gpus all \
		-v $(DATA_DIR):/workspace/datasets \
		-v $(OUTPUT_DIR):/workspace/outputs \
		-v $(PWD):/workspace \
		$(IMAGE_NAME):$(IMAGE_TAG) \
		bash scripts/run_evaluation.sh $(CONFIG)

clean:
	docker container prune -f
	docker image prune -f
