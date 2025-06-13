#!/bin/bash

# Base command
base_command="python RIWWER_AP4_AP5_Joint/main.py --args_file=RIWWER_AP4_AP5_Joint/args_files/errorgen_exp/error_args_file.txt"

# Base path to the models
base_path="all_trained_models/global"

# Path to the venv
venv_path="env/bin/activate"

# Define the specific model_dir for each model_type
# These are the best models foreach model_type according to val_MSE
declare -A model_dirs
model_dirs["tft"]="2024-06-27_22_41_46_torch_model_run_562529"
model_dirs["lstm"]="2024-06-25_12_32_15_torch_model_run_1785921"
model_dirs["transformer"]="2024-06-26_21_45_38_torch_model_run_538536"
model_dirs["nhits"]="2024-06-25_14_54_09_torch_model_run_232398"
model_dirs["deepar"]="2024-06-25_16_56_30_torch_model_run_307768"
model_dirs["tcn"]="2024-06-25_22_39_22_torch_model_run_440920"
model_dirs["custom_float_lstm"]="2024-08-02_03_31_52_torch_model_run_2204498"

# Create a screen session (named "errorgen_experiments")
session_name="errorgen_experiments"
screen -dmS $session_name

# Create a new screen window for each model_type
for model_type in "$base_path"/*; do
  # Check if it is a directory
  if [ -d "$model_type" ]; then
    # Get the base name of the model type directory
    model_type_name=$(basename "$model_type")

    # Get the specific model directory for this model_type (if specified)
    model_dir="${model_dirs[$model_type_name]}"

    if [ -z "$model_dir" ]; then
      echo "No specific model directory defined for $model_type_name. Skipping..."
      continue
    fi

    # Construct the full path to the model file
    model_file="$model_type/$model_dir/$model_type_name.pt"

    # Check if the model file exists
    if [ ! -f "$model_file" ]; then
      echo "Model file $model_file does not exist. Skipping $model_type_name..."
      continue
    fi

    # Create a new window in the same screen session for this model type, and name the window
    screen -S $session_name -X screen -t "$model_type_name" bash -c "
    echo 'Activating virtual environment for $model_type_name'
    source $venv_path && echo 'Virtual environment activated for $model_type_name'
    $base_command --inference_model_path=$model_file --model_type=$model_type_name
    exec bash
    "
    echo "Started window for model type: $model_type in session: $session_name"
  fi
done

# Optional: Reattach to the screen session to view the windows
screen -r $session_name
