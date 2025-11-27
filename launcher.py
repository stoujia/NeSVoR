import yaml
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to subjects.yaml")
    # You can pass global tuning params here to forward them to the worker
    parser.add_argument("--resolution", type=str, default="0.8")
    parser.add_argument("--iterations", type=str, default="3000")
    args = parser.parse_args()

    # 1. Read the YAML to get the list of jobs
    with open(args.config, 'r') as f:
        data = yaml.safe_load(f)

    subjects = data.get('subjects', {})
    
    print(f"Found {len(subjects)} subjects to process.")

    # 2. Loop through every Subject/Session
    for sub_id, sessions in subjects.items():
        for ses_id in sessions.keys():
            
            print(f"\n==================================================")
            print(f" LAUNCHING JOB: {sub_id} - {ses_id}")
            print(f"==================================================\n")

            # 3. Construct the command
            # This launches a completely new Python instance
            cmd = [
                sys.executable, "run.py",
                "--config", args.config,
                "--subject", sub_id,
                "--session", ses_id,
                "--resolution", args.resolution,
                "--iterations", args.iterations
            ]

            # 4. Run and wait
            try:
                # check=True will raise an error if the script fails
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"\n[!!!] Job failed for {sub_id}/{ses_id}. Continuing to next...\n")
            except KeyboardInterrupt:
                print("\nLauncher stopped by user.")
                sys.exit(0)

            # GPU memory is automatically cleaned up here because the subprocess ended

if __name__ == "__main__":
    main()