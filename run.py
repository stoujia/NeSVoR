import os
import glob
import subprocess
import torch
from argparse import ArgumentParser
from types import SimpleNamespace
import time
import yaml


# --- NeSVoR "Old Version" Imports ---
# try:
from nesvor.image.image import load_slices
from nesvor.inr.train import train
from nesvor.inr.models import NeSVoR
# from nesvor.inr.sample import sample_volume
from nesvor.cli.commands import _sample_inr
from nesvor.inr.data import PointDataset
# except ImportError as e:
#     print("Error importing NeSVoR. Check installation.", e)
#     exit(1)

# --- PATH CONFIGURATION (ADAPTED TO YOUR COMMAND) ---
# 1. Where your raw data lives on the HOST (The folder containing rawdata_bids and derivatives)
HOST_INPUT_ROOT = "/envau/work/meca"
DOCKER_INPUT_MOUNT = "/incoming"

# 2. The Docker Output Mount Point (Fixed internal path)
DOCKER_OUTPUT_MOUNT = "/outgoing"

def path_to_docker_input(host_path):
    """
    Converts a Host path to the Docker /incoming path.
    Example: /home/INT/.../rawdata_bids/... -> /incoming/rawdata_bids/...
    """
    if host_path.startswith(HOST_INPUT_ROOT):
        return host_path.replace(HOST_INPUT_ROOT, DOCKER_INPUT_MOUNT, 1)
    # Fallback/Warning if path is outside the known root
    print(f"WARNING: Path {host_path} is not inside {HOST_INPUT_ROOT}")
    return host_path 

def preprocess_with_docker(sub_id, stacks, masks, output_dir_host):
    """
    Runs 'nesvor register' via Docker using the specific mounts.
    """
    
    # Define where slices will be saved on the HOST
    slices_output_host = os.path.join(output_dir_host, "preproc_slices")
    os.makedirs(slices_output_host, exist_ok=True)
    
    # Check if we already did this
    if len(glob.glob(os.path.join(slices_output_host, "*.pt"))) > 0:
        print(f"  [Info] Slices found for {sub_id}, skipping Docker step.")
        return slices_output_host

    # 1. Convert Input Paths (Stacks/Masks) to /incoming/...
    docker_stacks = [path_to_docker_input(os.path.abspath(p)) for p in stacks]
    docker_masks = [path_to_docker_input(os.path.abspath(p)) for p in masks]
    
    # 2. Handle Output Path
    # For output, we tell Docker to write to /outgoing/
    # And we mount the HOST folder 'slices_output_host' to /outgoing
    docker_output_path = DOCKER_OUTPUT_MOUNT # Just "/" usually results in files in root, safer to use folder

    # 3. Build Docker Command
    cmd = [
        "docker", "run", "--rm", "--gpus", "all", "--ipc=host",
        # MOUNT 1: Input Data (Read Only)
        "-v", f"{HOST_INPUT_ROOT}:{DOCKER_INPUT_MOUNT}:ro",
        # MOUNT 2: Output Folder for THIS subject (Read Write)
        # We mount the specific slice folder to /outgoing inside docker
        "-v", f"{os.path.abspath(slices_output_host)}:{DOCKER_OUTPUT_MOUNT}:rw",
        "junshenxu/nesvor:v0.5.0",
        "nesvor", "register",
        "--registration", "svort",
        "--device", "0",
        "--output-slices", DOCKER_OUTPUT_MOUNT, # Write to /outgoing
        "--input-stacks"
    ] + docker_stacks + [
        "--stack-masks"
    ] + docker_masks

    print(f"  [Docker] Running SVoRT registration for {sub_id}...")
    print("Command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"  [Docker] Complete.")
        return slices_output_host
    except subprocess.CalledProcessError as e:
        print(f"  [Error] Docker failed for {sub_id}: {e}")
        raise e

def process_subject(subject_id, session_id, stacks, masks, output_root, args):
    print(f"--- Processing Subject: {subject_id} ---")
    
    # Prepare Subject Output Directory
    subject_out_dir = os.path.join(output_root, subject_id)
    os.makedirs(subject_out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. SVoRT Preprocessing (Docker)
    # ---------------------------------------------------------
    # This will mount the raw data and the specific output folder
    slices_dir = preprocess_with_docker(subject_id, stacks, masks, subject_out_dir)

    # ---------------------------------------------------------
    # 2. Load Slices (Local)
    # ---------------------------------------------------------
    print("Step 2: Loading slices locally...")
    slices = load_slices(slices_dir)

    # ---------------------------------------------------------
    # 3. Train NeSVoR Model (Local)
    # ---------------------------------------------------------
    print("Step 3: Training NeSVoR model...")

    start_time = time.time()  # <--- Start Timer

    # # To get the boundin box
    # dataset = PointDataset(slices)
    # bb = dataset.bounding_box

    model_inr, output_slices, mask = train(slices, args) #model.inr as an output

    end_time = time.time()    # <--- End Timer
    training_duration = end_time - start_time
    print(f"Training Duration: {training_duration:.2f} seconds")

    # ---------------------------------------------------------
    # 4. Sample Volume (Local)
    # ---------------------------------------------------------
    print("Step 4: Sampling volumes...")
    # volume = sample_volume(model_inr, mask, psf_resolution=args.output_resolution)
    output_volume, _ = _sample_inr(
            args,
            model_inr,
            mask,
            None,
            True,
            False,
        )

    # ---------------------------------------------------------
    # 5. Save Outputs
    # ---------------------------------------------------------
    print("Step 5: Saving NIfTI files...")
    output_volume.save(os.path.join(subject_out_dir, "reconstruction_masked.nii.gz"), masked=True) # with the mask
    output_volume.save_mask(os.path.join(subject_out_dir, "mask.nii.gz"))
    output_volume.save(os.path.join(subject_out_dir, "reconstruction_with_bg.nii.gz"), masked=False) # without the mask
    
    # 6. Save Processing Metadata
    print("Step 6: Saving Metadata...")
    metadata = {
        "subject_id": subject_id,
        "session_id": session_id,
        "training_time_seconds": round(training_duration, 2),
        "training_time_human": time.strftime("%H:%M:%S", time.gmtime(training_duration)),
        "parameters": {
            "resolution": args.output_resolution,
            "iterations": args.n_iter,
            "batch_size": args.batch_size
        },
        "inputs": {
            "stacks": stacks,
            "masks": masks
        }
    }

    # Save to JSON in the output folder
    json_path = os.path.join(subject_out_dir, "processing_info.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Done with {subject_id}. Info saved to {json_path}")





def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file", default="subjects.yaml")
    
    # Tuning
    parser.add_argument("--resolution", type=float, default=0.8)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=1024)

    conf = parser.parse_args()

    if not os.path.exists(conf.config):
        print(f"Error: Config file {conf.config} not found.")
        exit(1)

    with open(conf.config, 'r') as f:
        config_data = yaml.safe_load(f)

    root_output_dir = config_data.get('output_dir')
    if not root_output_dir:
        print("Error: 'output_dir' not defined in YAML.")
        exit(1)


# Consolidating defaults from:
# - build_parser_training()
# - build_parser_outputs_sampling()
# - build_parser_svr()
# - build_parser_common()

    args = SimpleNamespace(
        # --- Model Architecture (Training) ---
        n_features_per_level=2,
        log2_hashmap_size=19,
        level_scale=1.3819,
        coarsest_resolution=16.0,
        finest_resolution=0.5,
        n_levels_bias=0,
        
        # --- Implicit Network (Training) ---
        depth=1,
        width=64,
        n_features_z=15,
        n_features_slice=16,
        no_transformation_optimization=False,  # action="store_true"
        no_slice_scale=False,                  # action="store_true"
        no_pixel_variance=False,               # action="store_true"
        no_slice_variance=False,               # action="store_true"
        
        # --- Deformable Part (Training) ---
        deformable=False,                      # action="store_true"
        n_features_deform=8,
        n_features_per_level_deform=4,
        level_scale_deform=1.3819,
        coarsest_resolution_deform=32.0,
        finest_resolution_deform=8.0,

        # --- Loss and Regularization (Training) ---
        weight_transformation=0.1,
        weight_bias=100.0,
        image_regularization="edge",
        weight_image=1.0,
        delta=0.2,                             # Note: Defined in both training and svr
        img_reg_autodiff=False,                # action="store_true"
        weight_deform=0.1,

        # --- Training Parameters ---
        learning_rate=5e-3,
        gamma=0.33,
        milestones=[0.5, 0.75, 0.9],
        n_epochs=None,
        batch_size=4096,                       # 1024 * 4
        n_samples=256,                         # 128 * 2
        single_precision=False,                # action="store_true"
        
        # *** COLLISION NOTE ***
        # 'n_iter' is defined as 6000 in build_parser_training (for training steps)
        # but as 3 in build_parser_svr (for outer loop iterations).
        # Defaulting to 6000 for standard training compilation.
        n_iter=6000, 

        # --- Outputs Sampling ---
        output_resolution=0.8,
        output_intensity_mean=700.0,
        inference_batch_size=None,             # No default set in parser definition
        n_inference_samples=None,              # No default set in parser definition
        output_psf_factor=1.0,
        sample_orientation=None,
        sample_mask=None,

        # --- SVR (Slice-to-Volume Registration) ---
        no_slice_robust_statistics=False,      # action="store_true"
        no_pixel_robust_statistics=False,      # action="store_true"
        no_local_exclusion=False,              # action="store_true"
        no_global_exclusion=False,             # action="store_true"
        global_ncc_threshold=0.5,
        local_ssim_threshold=0.4,
        with_background=False,                 # action="store_true"
        n_iter_rec=[7],
        psf="gaussian",
        
        # --- Common / Miscellaneous ---
        device=0,
        verbose=1,
        output_log=None,
        seed=None,
        debug=False                            # action="store_true"
    )

    subjects_dict = config_data.get('subjects', {})
    
    for sub_id, sessions in subjects_dict.items():
        for ses_id, file_paths in sessions.items():
            
            session_out_dir = os.path.join(root_output_dir, sub_id, ses_id)
            os.makedirs(session_out_dir, exist_ok=True)
            
            stacks = file_paths.get('stacks', [])
            masks = file_paths.get('masks', [])

            if not stacks:
                print(f"Skipping {sub_id}/{ses_id}: No stacks found in YAML.")
                continue
            
            if len(stacks) != len(masks):
                print(f"Mismatch: {sub_id}/{ses_id} has {len(stacks)} stacks and {len(masks)} masks.")
                continue

            try:
                # Updated call to include ses_id
                process_subject(sub_id, ses_id, stacks, masks, session_out_dir, args)
            except Exception as e:
                print(f"FAILED on {sub_id}/{ses_id}: {e}")
                import traceback
                traceback.print_exc()
if __name__ == "__main__":
    main()