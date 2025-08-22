import subprocess
import os

models = [
    # add this before because it asks for permissions
    #"microsoft/Phi-3-small-8k-instruct", # > 7B

    # <= 4B
    #"microsoft/Phi-3.5-mini-instruct", # needs transformer==4.48 ?!
    #"meta-llama/Llama-3.2-3B-Instruct",
    #"ibm-granite/granite-3.2-2b-instruct",
    #"microsoft/Phi-4-mini-instruct",

    # > 7B and < 10B
    "allenai/Llama-3.1-Tulu-3.1-8B",
    "CohereForAI/c4ai-command-r7b-12-2024",
    "google/gemma-2-9b-it", 
    #("ibm-granite/granite-3.2-8b-instruct", "results/2025-05-04_16:11"),
    #"nvidia/AceInstruct-7B",
    #"Qwen/Qwen2.5-7B-Instruct",
    #"tiiuae/Falcon3-7B-Instruct"
]

old_run_folders = {
    "microsoft/Phi-3-small-8k-instruct": {
        1: "results/2025-04-29_12:18",
        5: "results/2025-04-29_12:19",
        10: "results/2025-04-29_12:22",
    },
    "microsoft/Phi-3.5-mini-instruct": {
        1: "results/2025-04-29_19:09",
        5: "results/2025-04-29_20:32",
        10: "results/2025-05-05_12:30",
    },
    "allenai/Llama-3.1-Tulu-3.1-8B": {
        1: "results/2025-05-12_10:33",
        5: "results/2025-05-12_17:04",
        10: "results/2025-05-13_01:00",
    },
    "CohereForAI/c4ai-command-r7b-12-2024": {
        1: "results/2025-04-30_02:59",
        5: "results/2025-04-30_05:06",
        10: "results/2025-05-05_17:29",
    },
    "google/gemma-2-9b-it": {
        1: "results/2025-04-30_10:16",
        5: "results/2025-05-07_13:44",
        10: "results/2025-05-07_13:09",
    },
}

# pos_neg_examples_cd = [0, 1, 5, 10, 15]

examples_per_class_tc = [1, 5, 10]

#examples_per_class_tc = [10]

def run_benchmark(model_name, pos_neg_examples, examples_per_class, cuda_device, old_run_folder=None):

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Run the benchmark command
    command = f"python app_cv.py --model_name_cd {model_name} --model_name_tc {model_name} --pos_neg_examples_cd {pos_neg_examples} --examples_per_class_tc {examples_per_class}"

    if old_run_folder:
        command = f"python app_cv.py --model_name_cd {model_name} --model_name_tc {model_name} --pos_neg_examples_cd {pos_neg_examples} --examples_per_class_tc {examples_per_class} --old_run_folder {old_run_folder}"

    subprocess.run(command, env=env, shell=True)

if __name__ == "__main__":
    # ask user for GPU device
    cuda_device = input("Enter the GPU device number (e.g., 0): ")
    if not cuda_device.isdigit():
        print("Invalid input. Please enter a valid GPU device number.")
        exit(1)
    cuda_device = int(cuda_device)

    for model_name in models:
        print(f"Running benchmark for model: {model_name}")
        for examples_per_class in examples_per_class_tc:
            try:
                run_benchmark(model_name, examples_per_class, examples_per_class, cuda_device, old_run_folders[model_name][examples_per_class])
            except Exception as e:
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory error encountered. Skipping this configuration.")
                    continue
                else:
                    raise e
            print(f"Completed benchmark for {model_name} with {examples_per_class} examples.")
        print(f"Finished all benchmarks for model: {model_name}")
    print("All benchmarks completed.")

