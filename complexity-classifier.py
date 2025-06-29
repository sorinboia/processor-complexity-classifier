from f5_ai_gateway_sdk.parameters import Parameters
from f5_ai_gateway_sdk.processor import Processor, Result
from f5_ai_gateway_sdk.request_input import RequestInput, Message, MessageRole
from f5_ai_gateway_sdk.response_output import ResponseOutput
from f5_ai_gateway_sdk.signature import BOTH_SIGNATURE
from f5_ai_gateway_sdk.tags import Tags
from f5_ai_gateway_sdk.type_hints import Metadata
from f5_ai_gateway_sdk.processor_routes import ProcessorRoutes
from starlette.applications import Starlette
from starlette.requests import Request

from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import DebertaV2Model
from huggingface_hub import PyTorchModelHubMixin
import torch
import torch.nn as nn
import numpy as np
import logging
import os

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        ).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super(CustomModel, self).__init__()
        logger.info("Initializing CustomModel")
        deberta_model_name = "microsoft/DeBERTa-v3-base"
        logger.info(f"Loading DeBERTa backbone config from: {deberta_model_name}")
        self.backbone_config = AutoConfig.from_pretrained(deberta_model_name)
        logger.info(f"Loading DeBERTa backbone model from: {deberta_model_name}")
        self.backbone = DebertaV2Model.from_pretrained(deberta_model_name, config=self.backbone_config)
        
        logger.info(f"Successfully loaded backbone: {type(self.backbone)}")
        self.target_sizes_values = list(target_sizes.values()) # Store only the sizes as a list
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map

        hidden_size = self.backbone_config.hidden_size
        logger.info(f"Backbone hidden size: {hidden_size}")
        logger.info(f"Initializing heads with target sizes (order matters): {self.target_sizes_values}")
        # Create heads as a standard list first
        self.heads = [
            MulticlassHead(hidden_size, sz)
            for sz in self.target_sizes_values
        ]
        # Explicitly register each head with add_module for correct naming
        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)
        logger.info(f"Initialized and registered {len(self.heads)} heads with 'head_X' naming.")
        self.pool = MeanPooling()

    def compute_results(self, preds, target, decimal=4):
        if target == "task_type":
            task_type = {}

            top2_indices = torch.topk(preds, k=2, dim=1).indices
            softmax_probs = torch.softmax(preds, dim=1)
            top2_probs = softmax_probs.gather(1, top2_indices)
            top2 = top2_indices.detach().cpu().tolist()
            top2_prob = top2_probs.detach().cpu().tolist()

            top2_strings = [
                [self.task_type_map[str(idx)] for idx in sample] for sample in top2
            ]
            top2_prob_rounded = [
                [round(value, 3) for value in sublist] for sublist in top2_prob
            ]

            counter = 0
            for sublist in top2_prob_rounded:
                if sublist[1] < 0.1:
                    top2_strings[counter][1] = "NA"
                counter += 1

            task_type_1 = [sublist[0] for sublist in top2_strings]
            task_type_2 = [sublist[1] for sublist in top2_strings]
            task_type_prob = [sublist[0] for sublist in top2_prob_rounded]

            return (task_type_1, task_type_2, task_type_prob)

        else:
            preds = torch.softmax(preds, dim=1)

            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
            scores = weighted_sum / self.divisor_map[target]

            scores = [round(value, decimal) for value in scores]
            if target == "number_of_few_shots":
                scores = [x if x >= 0.05 else 0 for x in scores]
            return scores
            
    def process_logits(self, logits):
        result = {}

        # Round 1: "task_type"
        task_type_logits = logits[0] # Assumes logits[0] is task_type
        task_type_results = self.compute_results(task_type_logits, target="task_type")
        result["task_type_1"] = task_type_results[0]
        result["task_type_2"] = task_type_results[1]
        result["task_type_prob"] = task_type_results[2]

        # Round 2: "creativity_scope"
        creativity_scope_logits = logits[1] # Assumes logits[1] is creativity_scope
        target = "creativity_scope"
        result[target] = self.compute_results(creativity_scope_logits, target=target)

        # Round 3: "reasoning"
        reasoning_logits = logits[2] # Assumes logits[2] is reasoning
        target = "reasoning"
        result[target] = self.compute_results(reasoning_logits, target=target)

        # Round 4: "contextual_knowledge"
        contextual_knowledge_logits = logits[3] # Assumes logits[3] is contextual_knowledge
        target = "contextual_knowledge"
        result[target] = self.compute_results(
            contextual_knowledge_logits, target=target
        )

        # Round 5: "number_of_few_shots"
        number_of_few_shots_logits = logits[4] # Assumes logits[4] is number_of_few_shots
        target = "number_of_few_shots"
        result[target] = self.compute_results(number_of_few_shots_logits, target=target)

        # Round 6: "domain_knowledge"
        domain_knowledge_logits = logits[5] # Assumes logits[5] is domain_knowledge
        target = "domain_knowledge"
        result[target] = self.compute_results(domain_knowledge_logits, target=target)

        # Round 7: "no_label_reason"
        no_label_reason_logits = logits[6] # Assumes logits[6] is no_label_reason
        target = "no_label_reason"
        result[target] = self.compute_results(no_label_reason_logits, target=target)

        # Round 8: "constraint_ct"
        constraint_ct_logits = logits[7] # Assumes logits[7] is constraint_ct
        target = "constraint_ct"
        result[target] = self.compute_results(constraint_ct_logits, target=target)

        # Round 9: "prompt_complexity_score"
        # Note: This calculation relies on the keys set in rounds 2-8 being correct.
        result["prompt_complexity_score"] = [
            round(
                0.35 * creativity
                + 0.25 * reasoning
                + 0.15 * constraint
                + 0.15 * domain_knowledge
                + 0.05 * contextual_knowledge
                + 0.05 * few_shots,
                5,
            )
            for creativity, reasoning, constraint, domain_knowledge, contextual_knowledge, few_shots in zip(
                result["creativity_scope"],
                result["reasoning"],
                result["constraint_ct"],
                result["domain_knowledge"],
                result["contextual_knowledge"],
                result["number_of_few_shots"],
            )
        ]

        return result

    def forward(self, batch):
        outputs = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        pooled = self.pool(outputs.last_hidden_state, batch["attention_mask"])
        logits = [head(pooled) for head in self.heads]
        return self.process_logits(logits)

class ComplexityClassifierParameters(Parameters):
    """
    This parameters class currently does not add new parameters.
    It inherits common parameters (annotate, modify, reject) from Parameters.
    """
    pass


class ComplexityClassifierProcessor(Processor[RequestInput, ResponseOutput, ComplexityClassifierParameters]):
    """
    Processor that uses the Nvidia Prompt Task and Complexity Classifier.

    It aggregates non-system messages from the prompt, passes them through
    the Nvidia model, and adds tags for Complexity, Creativity, Reasoning,
    Contextual Knowledge, Domain Knowledge, and Constraints based on the model output.
    """
    def __init__(self):
        super().__init__(
            name="complexity-classifier",
            version="v1",
            namespace="f5",
            parameters_class=ComplexityClassifierParameters,
            signature=BOTH_SIGNATURE,
        )
        logger.info("Initializing ComplexityClassifierProcessor")

        # Get history length from env or default to 2
        self.history_len = int(os.getenv("COMPLEXITY_HISTORY_LEN", "2"))
        logger.info(f"Using last {self.history_len} non-system messages for classification")

        try:
            model_name = "nvidia/prompt-task-and-complexity-classifier"
            logger.info("Loading custom model configuration")
            config = AutoConfig.from_pretrained(model_name)
            logger.info(f"Loaded config. Task type map: {config.task_type_map}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            logger.info("Initializing custom model")
            self.model = CustomModel(
                target_sizes=config.target_sizes,
                task_type_map=config.task_type_map,
                weights_map=config.weights_map,
                divisor_map=config.divisor_map,
            ).from_pretrained(model_name)
            self.model.eval()
            logger.info("Successfully loaded custom model")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
        # The following logging lines were removed as self.classifier does not exist.
        # logger.info(f"Pipeline created. Pipeline model type: {self.classifier.model.config.model_type}")
        # logger.info(f"Pipeline tokenizer vocab size: {self.classifier.tokenizer.vocab_size}")

        # --- Removed patching logic ---
        # The following patching logic for forward and preprocess methods,
        # and setting pad_token, was removed as it referenced 'self.classifier',
        # which is not used in the current implementation (self.model and self.tokenizer are used directly).
        # This patching might have been relevant for a Hugging Face pipeline object,
        # but is likely unnecessary/incorrect for the current CustomModel setup.
        # --- End Removed patching logic ---


    def process(
        self,
        prompt: RequestInput | None,
        response: ResponseOutput | None,
        metadata: Metadata,
        parameters: ComplexityClassifierParameters,
        request: Request,
    ) -> Result:
        tags = Tags()
        processor_result = {}

        if not prompt:
            logger.warning("No prompt provided")
            return Result(
                processor_result={"error": "No prompt provided"},
                tags=tags,
                modified=False,
            )

        # Concatenate all non-system messages from the prompt.
        texts = [message.content for message in prompt.messages if message.role != MessageRole.SYSTEM]
        if not texts:
            logger.warning("No non-system messages in the prompt")
            return Result(
                processor_result={"error": "No applicable messages in prompt"},
                tags=tags,
                modified=False,
            )

        # Use only the last N non-system messages, where N = self.history_len
        texts = texts[-self.history_len:]
        used_history_len = len(texts)

        combined_text = "Prompt: " + " ".join(texts)
        logger.info("Running complexity classifier on prompt text of length %d", len(combined_text))

        try:
            logger.debug(combined_text)
            logger.info("Tokenizing and classifying prompt...")
            inputs = self.tokenizer(
                [combined_text],  # single combined prompt with prefix
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
                add_special_tokens=True
            ).to(next(self.model.parameters()).device)  # Ensure inputs are moved to the correct device
            classification_results = self.model(inputs)
            logger.info("Classification successful")
            logger.debug("Results: %s", classification_results)
        except Exception as e:
            logger.error(f"Error during classification: {e}", exc_info=True)
            return Result(
                processor_result={"error": "Classification failed"},
                tags=tags,
                modified=False,
            )

        # Map results to our expected format
        result_obj = {
            "Task": classification_results.get("task_type_1", ["Unknown"])[0],
            "Complexity": classification_results.get("prompt_complexity_score", [0.0])[0],
            "Creativity": classification_results.get("creativity_scope", [0.0])[0],
            "Reasoning": classification_results.get("reasoning", [0.0])[0],
            "Contextual Knowledge": classification_results.get("contextual_knowledge", [0.0])[0],
            "Domain Knowledge": classification_results.get("domain_knowledge", [0.0])[0],
            "Constraints": classification_results.get("constraint_ct", [0.0])[0],
            "# of Few Shots": classification_results.get("number_of_few_shots", [0])[0],
            "History Length": used_history_len
        }

        processor_result["classification"] = result_obj

        keys_to_tag = [
            "Task",
            "Complexity",
            "Creativity",
            "Reasoning",
            "Contextual Knowledge",
            "Domain Knowledge",
            "Constraints",
            "# of Few Shots",
            "History Length"
        ]

        logger.debug(result_obj)

        for key in keys_to_tag:
            value = result_obj.get(key)
            if value is not None:
                tags.add_tag(key, str(value))
                logger.debug("Added tag: %s = %s", key, value)
            else:
                logger.debug("Key '%s' not found in classifier output", key)

        return Result(
            processor_result=processor_result,
            tags=tags if parameters.annotate else None,
            modified=False,
        )


app = Starlette(routes=ProcessorRoutes([ComplexityClassifierProcessor()]))
