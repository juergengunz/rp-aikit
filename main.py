import runpod
import os
import yaml
import dataset_downloader
from toolkit.job import get_job
from google.cloud import storage
from google.oauth2 import service_account
from huggingface_hub import HfApi
import time
import random
import concurrent.futures

GCS_SCHEMA = {
    'projectId': {'type': str, 'required': True},
    'bucketName': {'type': str, 'required': True},
    'credentialsJson': {'type': str, 'required': True}
}

HF_SCHEMA = {
    'token': {'type': str, 'required': True},
    'repo_id': {'type': str, 'required': True},
}

# Define the same schema for hfConfig2
HF_SCHEMA2 = {
    'token': {'type': str, 'required': True},
    'repo_id': {'type': str, 'required': True},
}

def edit_yaml(job_input, dataset_resolution=None):
    file_path = "config/main.yaml"
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    data['config']['name'] = job_input['lora_file_name']
    data['config']['process'][0]['sample']['samples'] = []  # Empty array for samples
    data['config']['process'][0]['trigger_word'] = job_input['trigger_word']
    data['config']['process'][0]['datasets'][0]['folder_path'] = job_input['dataset']
    
    # Set train.steps, train.batch_size, and train.lr
    data['config']['process'][0]['train']['steps'] = job_input.get('train_steps', 1000)
    data['config']['process'][0]['train']['batch_size'] = job_input.get('train_batch_size', 1)
    data['config']['process'][0]['train']['gradient_accumulation'] = job_input.get('gradient_accumulation', 1)

    data['config']['process'][0]['train']['lr'] = job_input.get('train_lr', 4e-4)

    # Set train.warmup_steps (sourced from top level of job_input)
    if job_input.get('warmup_steps') is not None:
        # Ensure the train dictionary exists
        if 'train' not in data['config']['process'][0]:
            data['config']['process'][0]['train'] = {}
        data['config']['process'][0]['train']['warmup_steps'] = job_input['warmup_steps']

    data['config']['process'][0]['model']['quantize'] = job_input['quantize']

    # Set model parameters from job_input if they exist
    if 'name_or_path' in job_input:
        data['config']['process'][0]['model']['name_or_path'] = job_input['name_or_path']
    if 'name_or_path_original' in job_input:
        data['config']['process'][0]['model']['name_or_path_original'] = job_input['name_or_path_original']
    if 'transformer_repo_id' in job_input:
        data['config']['process'][0]['model']['transformer_repo_id'] = job_input['transformer_repo_id']
    if 'transformer_filename' in job_input:
        data['config']['process'][0]['model']['transformer_filename'] = job_input['transformer_filename']

    
    # Set dataset.resolution using the passed parameter, ensuring it's a list
    if dataset_resolution is not None:
        current_resolution_val = dataset_resolution
        if not isinstance(current_resolution_val, list):
            current_resolution_val = [current_resolution_val] # Wrap in a list if it's not one
        data['config']['process'][0]['datasets'][0]['resolution'] = current_resolution_val
    
    # Set train.optimizer
    if 'optimizer' in job_input.get('train', {}):
        data['config']['process'][0]['train']['optimizer'] = job_input['train']['optimizer']
    
    # Set network.linear and network.linear_alpha
    if 'linear' in job_input.get('network', {}):
        data['config']['process'][0]['network']['linear'] = job_input['network']['linear']
    if 'linear_alpha' in job_input.get('network', {}):
        data['config']['process'][0]['network']['linear_alpha'] = job_input['network']['linear_alpha']
    
    # Set network.type
    if 'type' in job_input.get('network', {}):
        data['config']['process'][0]['network']['type'] = job_input['network']['type']
    
    # Set network.lokr_factor
    if 'lokr_factor' in job_input.get('network', {}):
        data['config']['process'][0]['network']['lokr_factor'] = job_input['network']['lokr_factor']
    
    # Set network.lokr_full_rank
    if 'lokr_full_rank' in job_input.get('network', {}):
        data['config']['process'][0]['network']['lokr_full_rank'] = job_input['network']['lokr_full_rank']
    
    # Set network_kwargs.ignore_if_contains
    if 'ignore_if_contains' in job_input and job_input['ignore_if_contains']:
        if 'network_kwargs' not in data['config']['process'][0]['network']:
            data['config']['process'][0]['network']['network_kwargs'] = {}
        data['config']['process'][0]['network']['network_kwargs']['ignore_if_contains'] = job_input['ignore_if_contains']
    
    # Set network_kwargs.only_if_contains
    if 'only_if_contains' in job_input and job_input['only_if_contains']:
        if 'network_kwargs' not in data['config']['process'][0]['network']:
            data['config']['process'][0]['network']['network_kwargs'] = {}
        data['config']['process'][0]['network']['network_kwargs']['only_if_contains'] = job_input['only_if_contains']
    
    # Set train.lr_scheduler
    if 'lr_scheduler' in job_input.get('train', {}):
        data['config']['process'][0]['train']['lr_scheduler'] = job_input['train']['lr_scheduler']
    
    # Set train.optimizer_params
    if 'optimizer_params' in job_input.get('train', {}):
        data['config']['process'][0]['train']['optimizer_params'] = job_input['train']['optimizer_params']

    # Set train.linear_timesteps
    if 'linear_timesteps' in job_input.get('train', {}):
        data['config']['process'][0]['train']['linear_timesteps'] = job_input['train']['linear_timesteps']
    
    # Set train.timestep_type
    if 'timestep_type' in job_input.get('train', {}):
        data['config']['process'][0]['train']['timestep_type'] = job_input['train']['timestep_type']
    
    # Set train.content_or_style
    if 'content_or_style' in job_input.get('train', {}):
        data['config']['process'][0]['train']['content_or_style'] = job_input['train']['content_or_style']
    
    # Set train.do_differential_guidance
    if 'do_differential_guidance' in job_input.get('train', {}):
        data['config']['process'][0]['train']['do_differential_guidance'] = job_input['train']['do_differential_guidance']
    
    # Set train.differential_guidance_scale
    if 'differential_guidance_scale' in job_input.get('train', {}):
        data['config']['process'][0]['train']['differential_guidance_scale'] = job_input['train']['differential_guidance_scale']
    
    # Set train.lr_scheduler_params.min_lr
    if job_input.get('min_lr') is not None:
        if 'lr_scheduler_params' not in data['config']['process'][0]['train']:
            data['config']['process'][0]['train']['lr_scheduler_params'] = {}
        data['config']['process'][0]['train']['lr_scheduler_params']['min_lr'] = job_input['min_lr']

    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)
    return True

def download(job_input):
    downloaded_input = dataset_downloader.file(job_input['data_url'])
    dataset = os.path.join(downloaded_input['extracted_path'])
    job_input['dataset'] = dataset
    return job_input

def _do_hf_upload(file_location, file_name, hf_config):
    """Internal function to perform the actual HuggingFace upload."""
    api = HfApi(token=hf_config['token'])
    api.upload_file(
        path_or_fileobj=file_location,
        path_in_repo=file_name,
        repo_id=hf_config['repo_id'],
        repo_type="model",
    )
    return f"https://huggingface.co/{hf_config['repo_id']}/resolve/main/{file_name}"

def upload_file_to_huggingface(file_name, file_location, hf_config, gcs_config=None, max_retries=3, initial_delay=1, timeout_seconds=300):
    """Upload file to HuggingFace with retry logic and timeout.
    
    Args:
        timeout_seconds: Maximum time in seconds to wait for upload (default 5 minutes)
    """
    for attempt in range(max_retries):
        try:
            # Use ThreadPoolExecutor to enforce timeout on the upload
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_hf_upload, file_location, file_name, hf_config)
                try:
                    result = future.result(timeout=timeout_seconds)
                    return result
                except concurrent.futures.TimeoutError:
                    print(f"Upload timed out after {timeout_seconds} seconds (Attempt {attempt + 1}/{max_retries})")
                    raise TimeoutError(f"HuggingFace upload timed out after {timeout_seconds} seconds")
        except Exception as e:
            if attempt < max_retries - 1:
                delay = 10 * (attempt + 1)  # This will give us 10, 20, 30 seconds
                print(f"Upload failed. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"Upload failed after {max_retries} attempts. Error: {str(e)}")
                raise

def upload_file_to_gcs(file_name, file_location, gcs_config, hf_config=None):
    credentials = service_account.Credentials.from_service_account_info(gcs_config['credentialsJson'])
    client = storage.Client(project=gcs_config['projectId'], credentials=credentials)
    bucket = client.get_bucket(gcs_config['bucketName'])
    
    
    blob = bucket.blob(f"models/{file_name}")
    
    blob.upload_from_filename(file_location, timeout=1800)
    return f"https://storage.googleapis.com/{gcs_config['bucketName']}/models/{file_name}"


def train_lora(job):
    job_input = job["input"]
    
    hf_token_to_set = None
    if 'hfConfig' in job_input and 'token' in job_input['hfConfig']:
        hf_token_to_set = job_input['hfConfig']['token']
        print("Found token in hfConfig, preparing to set HF_TOKEN environment variable.")
    elif 'hfConfig2' in job_input and 'token' in job_input['hfConfig2']:
        hf_token_to_set = job_input['hfConfig2']['token']
        print("Found token in hfConfig2, preparing to set HF_TOKEN environment variable.")
    else:
        print("No Hugging Face token found in hfConfig or hfConfig2 in job input.")

    if hf_token_to_set:
        os.environ['HF_TOKEN'] = hf_token_to_set
        print("HF_TOKEN environment variable set.")
    # Check if any upload configuration exists
    if not any(config in job_input for config in ['hfConfig', 'hfConfig2', 'gcsConfig']):
        return {"error": 'No upload configuration found. At least one of hfConfig, hfConfig2, or gcsConfig must be set!'}

    # Extract dataset resolution BEFORE job_input['dataset'] is overwritten by download()
    original_dataset_config = job_input.get('dataset', {})
    dataset_resolution_to_pass = None
    if isinstance(original_dataset_config, dict):
        dataset_resolution_to_pass = original_dataset_config.get('resolution')

    job_input = download(job_input)
    if edit_yaml(job_input, dataset_resolution=dataset_resolution_to_pass):
        job = get_job('config/main.yaml', None)
        job.run()
        job.cleanup()
        lora_path = os.path.join('output', job_input['lora_file_name'])
        job_output = {'lora_url': [], 'refresh_worker': True}
        
        for file in os.listdir(lora_path):
            if file.endswith('.safetensors'):
                file_location = os.path.join(lora_path, file)
                upload_success = False
                
                # Try primary hfConfig first
                if 'hfConfig' in job_input and not upload_success:
                    try:
                        print("Attempting upload with primary hfConfig...")
                        lora_url = upload_file_to_huggingface(
                            file_name=file,
                            file_location=file_location,
                            hf_config=job_input['hfConfig'],
                            max_retries=3,
                            initial_delay=1
                        )
                        job_output['lora_url'].append(lora_url)
                        upload_success = True
                        print("Upload with primary hfConfig successful!")
                    except Exception as e:
                        print(f"Primary hfConfig upload failed: {str(e)}")
                
                # Try backup hfConfig2 if primary failed
                if 'hfConfig2' in job_input and not upload_success:
                    try:
                        print("Attempting upload with backup hfConfig2...")
                        # Set HF_TOKEN to hfConfig2's token for this upload
                        if 'token' in job_input['hfConfig2']:
                            os.environ['HF_TOKEN'] = job_input['hfConfig2']['token']
                            print("Updated HF_TOKEN to use hfConfig2 token.")
                        lora_url = upload_file_to_huggingface(
                            file_name=file,
                            file_location=file_location,
                            hf_config=job_input['hfConfig2'],
                            max_retries=3,
                            initial_delay=1
                        )
                        job_output['lora_url'].append(lora_url)
                        upload_success = True
                        print("Upload with backup hfConfig2 successful!")
                    except Exception as e:
                        print(f"Backup hfConfig2 upload failed: {str(e)}")
                
                # Fall back to gcsConfig if both HF uploads failed
                if 'gcsConfig' in job_input and not upload_success:
                    try:
                        print("Attempting upload with gcsConfig...")
                        lora_url = upload_file_to_gcs(
                            file_name=file,
                            file_location=file_location,
                            gcs_config=job_input['gcsConfig']
                        )
                        job_output['lora_url'].append(lora_url)
                        upload_success = True
                        print("Upload with gcsConfig successful!")
                    except Exception as e:
                        print(f"GCS upload failed: {str(e)}")
                
                # If all upload methods failed
                if not upload_success:
                    return {'error': 'All configured upload methods failed. Check logs for details.'}
                
        return job_output
    else:
        return {'error': 'Training YAML error'}

runpod.serverless.start({"handler": train_lora})