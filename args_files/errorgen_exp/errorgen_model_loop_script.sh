#!/bin/bash

# Name of the screen session
session_name="errorgen_experiments"

# Base path to the model types
base_path="all_trained_models/global"

# Base Python command
base_command="python RIWWER_AP4_AP5_Joint/main.py --args_file=RIWWER_AP4_AP5_Joint/args_files/errorgen_exp/error_args_file.txt"

# Path to the virtual environment
venv_path="env/bin/activate"

# Create the screen session
screen -dmS $session_name

# set number of windows per model below

# Make subdir if it does not exist
mkdir -p "RIWWER_AP4_AP5_Joint/args_files/errorgen_exp/temp_iterate_scripts"
mkdir -p "errorgen_process_log"

# Iterate over each model type and create multiple windows
for model_type in "$base_path"/*; do
  if [ -d "$model_type" ]; then
    model_type_name=$(basename "$model_type")

    # Give tft model additional windows
    if [ "$model_type_name" == "tft" ]; then
      num_windows=4
    else
      num_windows=2
    fi

    # Get the list of model directories
    model_dirs=("$model_type"/*)
    total_dirs=${#model_dirs[@]}
    
    # Calculate the number of model directories per window
    dirs_per_window=$((total_dirs / num_windows))
    remainder=$((total_dirs % num_windows))  # For handling any leftover directories
    
    # Split the workload across multiple windows
    start_index=0
    for window_id in $(seq 1 $num_windows); do
      # Calculate the end index for this window's workload
      end_index=$((start_index + dirs_per_window))
      if [ $window_id -le $remainder ]; then
        end_index=$((end_index + 1))  # Distribute the remainder evenly
      fi

      # Create the iteration script for this window
      script_name="RIWWER_AP4_AP5_Joint/args_files/errorgen_exp/temp_iterate_scripts/iterate_${model_type_name}_window_${window_id}.sh"
      log_file="errorgen_process_log/log_${model_type_name}_window_${window_id}.log"
      
      echo "#!/bin/bash" > $script_name
      echo "source $venv_path" >> $script_name
      echo "model_type_name=\"$model_type_name\"" >> $script_name
      echo "log_file=\"$log_file\"" >> $script_name
      echo "count=0" >> $script_name
      
      # Write the model_dirs assigned to this window directly into the script
      echo "for model_dir in ${model_dirs[@]:$start_index:$((end_index - start_index))}; do" >> $script_name
      echo "  model_file=\"\$model_dir/\$model_type_name.pt\"" >> $script_name
      echo "  if [ -f \"\$model_file\" ]; then" >> $script_name
      echo "    count=\$((count+1))" >> $script_name
      echo "    timestamp=\$(date '+%Y-%m-%d %H:%M:%S')" >> $script_name
      echo "    echo \"[\$timestamp] Processing: \$model_file (Model \$count)\" | tee -a \$log_file" >> $script_name
      echo "    $base_command --inference_model_path=\$model_file --model_type=\$model_type_name" >> $script_name
      echo "    timestamp=\$(date '+%Y-%m-%d %H:%M:%S')" >> $script_name
      echo "    echo \"[\$timestamp] Finished processing: \$model_file\" | tee -a \$log_file" >> $script_name
      echo "  else" >> $script_name
      echo "    echo \"Model file \$model_file does not exist.\" | tee -a \$log_file" >> $script_name
      echo "  fi" >> $script_name
      echo "done" >> $script_name

      # Make the script executable
      chmod +x $script_name
      echo "Created iteration script: $script_name"
      
      # Create a window for this iteration script
      screen -S $session_name -X screen -t "${model_type_name}_win_${window_id}" bash -c "
        echo 'Window for $model_type_name (window $window_id) created.'
        ./$script_name
        exec bash
      "

      # Update the start index for the next window
      start_index=$end_index
    done
  fi
done

echo "Screen session, windows, and scripts created. Scripts are running in their respective windows."
