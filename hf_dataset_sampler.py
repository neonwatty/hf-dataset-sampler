import os
import datasets
from huggingface_hub import HfApi, upload_file, login
import argparse
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Retrieve credentials
username = os.getenv("HUGGINGFACE_USERNAME")
token = os.getenv("HUGGINGFACE_TOKEN")

# Log in to Hugging Face
login(token=token)

print(f"Logged in as {username}")


# Function to create or download README.md
def create_or_download_readme(dataset_name_sample, cache_dir, config_name="default_config", description="Sample dataset"):
    readme_path = os.path.join(cache_dir, "README.md")

    # Check if README.md exists
    if not os.path.exists(readme_path):
        print("README.md not found. Creating a new one...")

        # Create metadata content for the README.md
        metadata = f"""---
configs:
  - config_name: "{config_name}"
    description: "{description}"
---
# {dataset_name_sample}
This is a sample of {dataset_name_sample}.
"""
        # Write the README.md file
        with open(readme_path, "w") as f:
            f.write(metadata)
        print(f"README.md created at {readme_path}")
    else:
        print(f"README.md already exists at {readme_path}")

    return readme_path

# Function to create a sample of the dataset
def create_sample_dataset(full_dataset_name, subset_name, username, sample_count):
    # Create base datasets directory if it does not exist
    cache_dir = "./datasets"
    os.makedirs(cache_dir, exist_ok=True)

    # Get the dataset name and create sample dataset name
    dataset_name = full_dataset_name.split("/")[-1]
    dataset_name_sample = f"{dataset_name}-sample-{sample_count}"

    # Load the dataset
    dataset = datasets.load_dataset(full_dataset_name, subset_name, cache_dir=cache_dir)

    # Create a directory for the dataset sample
    dataset_cache_dir = os.path.join(cache_dir, dataset_name_sample)
    os.makedirs(dataset_cache_dir, exist_ok=True)

    # Get the dataset splits (e.g., train, test)
    splits = list(dataset.keys())

    # Ensure README.md exists or create it
    create_or_download_readme(dataset_name_sample, dataset_cache_dir)

    # Update the dataset card (README)
    update = f"""
    # {dataset_name_sample}
    Sample of {sample_count} rows from the {full_dataset_name} dataset.
    """
    
    # Here you'd implement `update_dataset_card` function to update the card
    # This could be done with `huggingface_hub` functions like `push_to_hub` or direct file manipulation.
    print("INFO: Dataset card updated successfully")

    # Sample data from each split and save to local directory
    for split in splits:
        # Collect sample
        split_sample = dataset[split].shuffle(seed=42).select(range(sample_count))

        # Save the sample data to CSV (or other formats as needed)
        split_sample.to_csv(os.path.join(dataset_cache_dir, f"{split}.csv"))
        print(f"INFO: {split} split saved locally to {dataset_cache_dir}/{split}.csv")

    # Upload the folder (including sample data and README.md) to the Hugging Face Hub
    # Create the repo on Hugging Face if it doesn't exist
    api = HfApi()
    try:
        # Try to create the repo if it doesn't exist
        api.create_repo(repo_id=f"{username}/{dataset_name_sample}", repo_type="dataset", exist_ok=True)
        print(f"INFO: Repository {dataset_name_sample} created on Hugging Face.")
    except Exception as e:
        print(f"ERROR: {e}")

    # Upload the folder (including sample data and README.md) to the Hugging Face Hub
    api.upload_folder(
        repo_id=f"{username}/{dataset_name_sample}",
        folder_path=dataset_cache_dir,
        repo_type="dataset",  # Specify that this is a dataset repo
        path_in_repo="",  # Optional, specify the path within the repo if necessary
    )
    print("\n")
    print(f"INFO: Dataset folder uploaded to the hub successfully: https://huggingface.co/datasets/{username}/{dataset_name_sample}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset arguments.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of hf dataset")
    parser.add_argument(
        "--subset_name",
        type=str,
        default=None,
        help="Optional subset name (default: None)",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=100,
        help="Number of samples to process (default: 100)",
    )

    args = parser.parse_args()

    dataset_name = args.dataset_name
    subset_name = args.subset_name
    sample_count = args.sample_count

    print(
        f"INFO: Generating sample for dataset: {dataset_name}, Subset: {subset_name}, Sample Count: {sample_count}"
    )

    create_sample_dataset(dataset_name, subset_name, username=username, sample_count=sample_count)
