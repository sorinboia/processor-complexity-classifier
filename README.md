# F5 AI Gateway - Prompt Complexity Classifier Processor

## Overview

This processor integrates the [Nvidia Prompt Task and Complexity Classifier](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) model into the F5 AI Gateway ecosystem. It analyzes incoming prompts (specifically, non-system messages) to classify them based on task type and various complexity dimensions.

The results of the classification are added as tags to the request, which can then be used within the AI Gateway configuration for routing, policy enforcement, or observability.

## Functionality

1.  **Input Processing:** Receives the request object containing the prompt messages.
2.  **Text Extraction:** Filters out messages with the `SYSTEM` role and concatenates the content of the remaining messages.
3.  **Tokenization:** Uses the tokenizer associated with the `nvidia/prompt-task-and-complexity-classifier` model to prepare the text for the model.
4.  **Model Inference:** Feeds the tokenized input into a custom model (`CustomModel`) which includes:
    *   A `microsoft/DeBERTa-v3-base` backbone.
    *   Multiple classification heads tailored for the specific tasks and complexity dimensions defined by the Nvidia model.
5.  **Result Processing:** Calculates the final scores and labels from the model's output logits. This includes:
    *   Identifying the primary task type (e.g., "Text Generation", "Summarization").
    *   Calculating scores for complexity dimensions (Creativity, Reasoning, etc.).
    *   Computing an overall complexity score based on a weighted sum of the dimensions.
6.  **Tagging:** If the processor parameter `annotate` is `true` (default), it adds the following tags to the request:
    *   `Task`: The predicted primary task type.
    *   `Complexity`: The overall calculated complexity score.
    *   `Creativity`: The creativity score.
    *   `Reasoning`: The reasoning score.
    *   `Contextual Knowledge`: The contextual knowledge score.
    *   `Domain Knowledge`: The domain knowledge score.
    *   `Constraints`: The constraints score.
    *   `# of Few Shots`: The score related to the number of few-shot examples detected (often 0 if none are present).

## Model Used

*   **Hugging Face Model:** [`nvidia/prompt-task-and-complexity-classifier`](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier)
*   **Backbone:** `microsoft/DeBERTa-v3-base`

## Dependencies

*   `f5-ai-gateway-sdk`
*   `transformers`
*   `torch`
*   `numpy`
*   `huggingface_hub`
*   `starlette` (for running the processor service)
*   An ASGI server (e.g., `uvicorn`)

## Running the Processor

The processor is built as a standard ASGI application using Starlette. You can run it locally using an ASGI server like Uvicorn:

```bash
# Ensure you are in the directory containing complexity-classifier.py
pip install uvicorn transformers torch numpy huggingface_hub f5-ai-gateway-sdk starlette

# Run the server (adjust host/port as needed)
python -m uvicorn complexity-classifier:app --host 127.0.0.1 --port 9999
```

## AI Gateway Configuration

To use this processor within F5 AI Gateway, configure it in your `aigw.yml` file under the `processors` section:

```yaml
processors:
  - name: complexity-classifier # Or any name you prefer
    type: external
    config:
      endpoint: "localhost:9999" # Adjust if running elsewhere
      namespace: f5             # Must match the namespace in the processor code
      version: 1                # Must match the version in the processor code
      # Optional: Add params if you need to override defaults (e.g., disable annotation)
      # params:
      #   annotate: false
```

You can then reference this processor by its name (`complexity-classifier` in this example) in the `steps` section of your policies.
