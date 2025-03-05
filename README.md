# hf-dataset-sampler

Create a small sample of any huggingface dataset for testing and push it to your profile.

- [Usage instructions](#usage-instructions)
    - [Keep your credentials safe](#keep-your-credentials-safe-in-a-env-file)
    - [Install the requirements](#install-requirements)
    - [Run the script](#how-to-run-the-script)

# Usage instructions

## Keep your credentials safe in a `.env` file

In a `.env` file set your HF username and access token as

```bash
HUGGINGFACE_USERNAME=<your_hf_user_name>
HUGGINGFACE_TOKEN=<your_hf_access_token>
```

## Install requirements

Use `pip` or `uv` to install tool requirements in `requirements.txt`.  For example with `uv` first create and activate a venv as

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Then install the requirements as

```bash
uv pip install -r requirements.txt
```


## How to run the script

First choose a huggingface dataset you wish to sample, like [the standard imdb sentiment analysis dataset](https://huggingface.co/datasets/stanfordnlp/imdb) consisting of around 25,000 rows.  Copy down its name - in this instance `stanfordnlp/imdb`.

Choose a `sample_count` (defaults to `100`) and if need a `subset_name` (defaults to `None`) and run the script as

```bash
python hf_dataset_sampler.py --dataset_name <hf_dataset_name> --sample_count <desired_number_of_samples>
```

This will pull the entire original dataset, sample `sample_count` rows from it, and push the smaller sample to your `Datasets` on huggingface.

For example, to create a sample of `250` rows from `stanfordnlp/imdb` run the script as 

```bash
python hf_dataset_sampler.py --dataset_name stanfordnlp/imdb --sample_count 250
```

This creates [the dataset shown here](https://huggingface.co/datasets/neonwatty/imdb-sample-250).