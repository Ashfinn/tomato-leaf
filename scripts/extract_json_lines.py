import os
import json

def extract_specific_metrics(file_path):
    """
    Extracts specific metrics from a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - dict: Dictionary containing the extracted metrics.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            print(f"Warning: The file {file_path} does not contain a JSON object. Skipping.")
            return {}
        
        # Define the list of metrics to extract
        metrics_to_extract = [
            "model_name",
            "image_size",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "cohen_kappa",
            "roc_auc",
            "model_size_mb",
            "parameter_count",
            "estimated_flops_gflops",
            "inference_time_ms",
            "throughput_fps",
            "efficiency_score"
        ]
        
        # Extract the specified metrics
        extracted_metrics = {}
        for key in metrics_to_extract:
            if key in data:
                extracted_metrics[key] = data[key]
            else:
                extracted_metrics[key] = None  # or you can choose to skip or handle missing keys as needed
        
        return extracted_metrics
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"An error occurred while processing file {file_path}: {e}")
        return {}

def process_json_files(directory, output_file):
    """
    Processes all JSON files in the specified directory and extracts specific metrics.

    Parameters:
    - directory (str): Path to the directory containing JSON files.
    - output_file (str): Path to the output text file.
    """
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return
    
    metrics_data = []
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith('.json'):
            file_path = os.path.join(directory, filename)
            extracted_metrics = extract_specific_metrics(file_path)
            
            if extracted_metrics:
                metrics_data.append({
                    'filename': filename,
                    'metrics': extracted_metrics
                })
    
    # Write the extracted metrics to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for entry in metrics_data:
                out_f.write(f"Metrics for {entry['filename']}:\n")
                for key, value in entry['metrics'].items():
                    out_f.write(f"{key}: {value}\n")
                out_f.write("\n")
        print(f"Extracted metrics saved to {output_file}")
    except Exception as e:
        print(f"Failed to write to file {output_file}: {e}")

def main():
    directory = '.'  # Current directory
    output_file = 'extracted_specific_metrics.txt'
    
    process_json_files(directory, output_file)

if __name__ == "__main__":
    main()
