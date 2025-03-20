#!/bin/bash

# Default values
DATA_DIR="files/wikipedia_personas"
LANGUAGE="sw"
BATCH_SIZE=10
MODEL="gpt-4o"
PROVIDER="azure"
DEPLOYMENT_NAME="gpt-4o"
DEPLOYMENT_DATE="2024-02-15"

# Parse named arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --language=*)
            LANGUAGE="${1#*=}"
            ;;
        --batch_size=*)
            BATCH_SIZE="${1#*=}"
            ;;
        --model=*)
            MODEL="${1#*=}"
            ;;
        --deployment_name=*)
            DEPLOYMENT_NAME="${1#*=}"
            ;;
        --model_provider=*)
            PROVIDER="${1#*=}"
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# Print run parameters
echo "=== Please verify the following parameters ==="
echo "Data Directory: ${DATA_DIR}"
echo "Language: ${LANGUAGE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Model: ${MODEL}"
echo "Model Provider: ${PROVIDER}"
echo "Deployment Name: ${DEPLOYMENT_NAME}"
echo "Deployment Date: ${DEPLOYMENT_DATE}"
echo "==========================================="

# Ask for confirmation
read -p "Do you want to proceed with these parameters? (y/n): " confirm

if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    echo "Starting script..."
    python3 -m post_training.data.create_personas_wikipedia \
        --data_directory "${DATA_DIR}" \
        --language "${LANGUAGE}" \
        --batch_size "${BATCH_SIZE}" \
        --model "${MODEL}" \
        --model_provider "${PROVIDER}" \
        --azure_deployment_name "${DEPLOYMENT_NAME}" \
        --azure_deployment_date "${DEPLOYMENT_DATE}"
else
    echo "Script execution cancelled."
    exit 1
fi