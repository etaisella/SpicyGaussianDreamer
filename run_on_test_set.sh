#!/bin/bash
# Rendering function template:
train_default() {
	# Train:
	python launch.py \
	--config="/home/etaisella/repos/SpicyGaussianDreamer/configs/gaussiandreamer-sd-airplane-deabs-test.yaml" \
	--train \
	--random \
	--gpu=0 \
    --guidance="$2" \
	system.prompt_processor.prompt="$1"
}

# Specify the folder to search in
folder="/home/etaisella/repos/SpicyGaussianDreamer/load/abstraction_plane/test_final"
finished_list="/home/etaisella/repos/SpicyGaussianDreamer/outputs/airplane_deabs_test/finished.txt"

# Iterate over all subfolders in the given folder
for subfolder in "$folder"/*; do
    # Check if the subfolder contains the required files
    if [[ -f "$subfolder/annotation.json" && -f "$subfolder/latent_original.pt" ]]; then
        # print subfolder without the parent folder
        echo "Subfolder: ${subfolder##*/}"

        # if a subfolder is in the finished list, skip it
        if grep -q "${subfolder##*/}" "$finished_list"; then
            echo "Skipping ${subfolder##*/}"
            continue
        fi

        # Extract the 'name' field from the annotation.json file
        name=$(jq -r '.name' "$subfolder/annotation.json")
        
        # Print the full path to the latent_original.pt file and the 'name' field
        echo "Path to latent_original.pt: $subfolder/latent_original.pt"
        echo "Name: $name"

        # run the train_default function with the 'name' field and the full path to the latent_original.pt file
        train_default "$name" "$subfolder/latent_original.pt"

        # when finished, add the subfolder to the finished list
        echo "${subfolder##*/}" >> "$finished_list"
    fi
done