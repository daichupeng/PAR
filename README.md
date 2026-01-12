# PAR (Predictive Acquisition & Retrieval)

## Overview

PAR is an Active Feature Acquisition (AFA) framework designed to efficiently acquire features for classification tasks using a Student-Teacher distillation approach combined with Retrieval-Augmented Generation (RAG).

The system operates in two main phases:
1.  **Training**:
    *   A **Teacher Model** (Gradient Boosted Trees like CatBoost) is trained on the complete dataset.
    *   A **Student Model** (`StudentEmbedder`) is trained to mimic the Teacher and generate robust embeddings, even with missing features. The Student learns to predict both the target class and the retrieval embeddings.
    *   A **Vector Database** (FAISS) is built using the Student's embeddings of the training data.

2.  **Inference (Active Feature Acquisition)**:
    *   For a test instance, features are acquired sequentially.
    *   At each step, the current observed features are used to query the Vector Database for similar "neighbor" cases.
    *   An **Agent** (which can be an LLM, a heuristic, or a random policy) decides which feature to acquire next, based on the current state and the insights from retrieved neighbors.
    *   The process continues until a budget is met or a prediction is made.

## Requirements

The project requires Python 3.8+ and the following libraries:

```bash
pip install torch numpy pandas scikit-learn tqdm faiss-cpu install catboost openai
```

## Usage

The main entry point is `main.py`.

### Basic Command

```bash
python main.py <dataset_name> [options]
```

### Arguments

*   `dataset_name`: (Positional) Name of the dataset to use. Supported options: `german_credit`, `student`, `wine`, `fraud`, `ctgs`, `income`. Put the dataset csv file in datasets/
*   `--api_key`: API Key for the LLM test method. Defaults to the `openai_api_key` environment variable.
*   `--device`: Device to run on (e.g., `cuda`, `cpu`, `mps`). Default: `cuda`.
*   `--test_method`: The method/agent used for feature acquisition. Default: `gpt-5-mini`. Common options:
    *   `gpt-4o`, `gpt-5-mini`: LLM-based agents.
    *   `random`: Random feature acquisition.
    *   `neighbor`: Heuristic-based on neighbor votes.
*   `--precedents`: Number of precedents to retrieve from the vector DB. Default: `5`.
*   `--anonymous_feature`: Flag to use anonymized feature names (e.g., "Feature 0" instead of "Age").
*   `--print_llm_results`: Flag to print LLM prompts and responses to the console.
*   `--num_threads`: Number of threads for parallel processing of test instances. Default: `32`.

### Examples

**Run with GPT-4o and 5 precedents:**
```bash
export openai_api_key="sk-..."
python main.py fraud --test_method gpt-4o --precedents 5
```


## Project Structure

*   `main.py`: Entry point. Sets up the dataset, trains models, builds the index, and runs the prediction pipeline.
*   `tabular_dataset.py`: Data loaders for various tabular datasets.
*   `student.py`: The state model. Trained via contrastive learning and distillation from the tree-based teacher.
*   `tree_training.py`: Handles training of Tree-based Teacher models (CatBoost).
*   `embedding.py`: The vector database. Implements `AFAVectorDatabase`, wrapping FAISS for context retrieval.
*   `prediction.py`: The prediction pipeline. Contains the core loop for `run_prediction_pipeline`, managing the step-by-step feature acquisition and Agent interactions.
*   `prompts_config.py`: Configuration for LLM prompts.
*   `utils.py`: Mapping of categorical features to their values.
*   `datasets/`: Directory containing dataset CSV files.
