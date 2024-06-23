import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--input_data", type=str, help="CSV train input file")
parser.add_argument("--model_name", type=str, help="Model name", default="sample")
parser.add_argument("--epochs", type=int, help="Number of training epochs", default=100)
parser.add_argument("--batch_size", type=int, help="Batch size", default=100)

args = parser.parse_args()

input_data = args.input_data
total_iterations = args.epochs
model_name = args.model_name
batch_size = args.batch_size

# Path to your script to run
script_path = './src/model/gann/train_gann.py'

# Additional arguments for your script
additional_args = [
    '--input_data',
    input_data,
    '--model_name',
    model_name
]

# Loop through batches
for start_epoch in range(0, total_iterations, batch_size):
    end_epoch = min(start_epoch + batch_size, total_iterations)
    print(f"Running batch {start_epoch // batch_size + 1} from epoch {start_epoch} to {end_epoch - 1}...")

    # Construct the command to run your script with appropriate arguments
    command = ['python', script_path]
    command.extend(additional_args)
    command.extend(['--epochs', str(end_epoch - start_epoch)])
    
    # Start the subprocess asynchronously
    process = subprocess.Popen(command)
    
    # Wait for the subprocess to finish
    exit_code = process.wait()
    
    if exit_code:
        print("Failed to run all batches")
        exit(exit_code)

print("All batches completed.")
