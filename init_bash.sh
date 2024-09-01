#!/bin/bash

MODEL_DIR="model"
REPO_URL="https://huggingface.co/openai/whisper-large-v3"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Создание папки $MODEL_DIR..."
    mkdir "$MODEL_DIR"
fi

cd "$MODEL_DIR" || { echo "Не удалось перейти в директорию $MODEL_DIR"; exit 1; }

if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS не установлен. Устанавливаем..."

    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

    sudo apt-get install -y git-lfs
else
    echo "Git LFS уже установлен."
fi

git lfs install

echo "Клонирование репозитория $REPO_URL..."
git clone "$REPO_URL"