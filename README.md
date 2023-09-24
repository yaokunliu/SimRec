## SimRec
This repo is the official implementation for the paper "Attribute Simulation for Item Embedding Enhancement in Multi-Interest Recommendation"

### Environment Setup

- This repo is developed based on [pytorch](https://hub.docker.com/layers/pytorch/pytorch/2.0.1-cuda11.7-cudnn8-runtime/images/sha256-82e0d379a5dedd6303c89eda57bcc434c40be11f249ddfadfd5673b84351e806?context=explore) docker images, which is recommended for simplicity and reproducibility. If you do this, you can run the following command in the docker container to install the other required packages:

```bash
sudo /opt/conda/bin/conda install -c conda-forge -y faiss-gpu==1.7.4
pip install -r requirements.txt
```

- You can also create a new conda environment and install the required packages by running the following command:

```bash
conda create -n SimRec python=3.10.9
conda activate SimRec
# modify the cuda version according to your environment
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge -y faiss-gpu==1.7.4
pip install -r requirements.txt
```

### Dataset Preparation

- All datasets are prepared and stored in the `data` folder.

### Training and Evaluation

- To train the model in the paper, run the `scripts/train.sh` script.

- To evaluate the model in the paper, run the `scripts/eval.sh` script.

