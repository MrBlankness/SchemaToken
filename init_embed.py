import math
import numpy as np
import torch
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Literal

def round_to_nearest_multiple(vocabulary_size, multiple):
    rounded_size = math.ceil(vocabulary_size / multiple) * multiple
    return rounded_size

def instantiate_model_by_mean(
    source_model: AutoModelForCausalLM,
    source_tokenizer: AutoTokenizer,
    target_tokenizer: AutoTokenizer,
    tie_word_embeddings: bool = False,
    init_strategy: Literal["mean", "first"] = "mean"
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    # init
    source_embeddings = source_model.get_input_embeddings().weight.detach().numpy()
    target_embeddings = np.zeros(
        (round_to_nearest_multiple(len(target_tokenizer), 8), 
         source_embeddings.shape[1])
    )
    target_embeddings[:len(source_tokenizer)] = source_embeddings[:len(source_tokenizer)]
    if not tie_word_embeddings:
        # print("You are using the output projection init.")
        source_head_embeddings = source_model.get_output_embeddings().weight.detach().numpy()
        target_head_embeddings = np.zeros(
            (round_to_nearest_multiple(len(target_tokenizer), 8), 
             source_head_embeddings.shape[1])
        )
        target_head_embeddings[:len(source_tokenizer)] = source_head_embeddings[:len(source_tokenizer)]
    
    # initialize the rest of the embeddings
    for i in range(len(source_tokenizer), len(target_tokenizer)):
        token = target_tokenizer.convert_ids_to_tokens(i)
        source_ids = source_tokenizer.convert_tokens_to_ids(source_tokenizer.tokenize(token))
        target_embeddings[i] = source_embeddings[source_ids].mean(axis=0)
        if not tie_word_embeddings:
            if init_strategy == "mean":
                target_head_embeddings[i] = source_head_embeddings[source_ids].mean(axis=0)
            elif init_strategy == "first":
                target_head_embeddings[i] = source_head_embeddings[source_ids[0]]
    
    # expand the embeddings
    target_model = source_model
    target_model.resize_token_embeddings(
        len(target_tokenizer), 
        pad_to_multiple_of=8 # See https://github.com/huggingface/transformers/issues/26303
    )
    target_model.get_input_embeddings().weight.data = torch.from_numpy(target_embeddings)
    target_model.config.vocab_size = round_to_nearest_multiple(len(target_tokenizer), 8)
    if not tie_word_embeddings:
        target_model.get_output_embeddings().weight.data = torch.from_numpy(target_head_embeddings)
    else:
        target_model.tie_weights()
    
    return target_model, target_tokenizer

def main(args):
    source_tokenizer = AutoTokenizer.from_pretrained(
        args.source_model_name_or_path,
        local_files_only = True,
        cache_dir=args.cache_dir
    )
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer_name_or_path)
    source_model = AutoModelForCausalLM.from_pretrained(
        args.source_model_name_or_path,
        local_files_only = True,
        cache_dir=args.cache_dir
    )

    tie_word_embeddings = source_model.config.tie_word_embeddings

    target_model, target_tokenizer = instantiate_model_by_mean(
        source_model, source_tokenizer, target_tokenizer, tie_word_embeddings, args.init_strategy
    )
    
    # Save the target model and tokenizer
    target_model.save_pretrained(args.output_dir)
    target_tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = ArgumentParser("Initialize the target model.")
    parser.add_argument(
        "--source_model_name_or_path", 
        type=str, 
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--target_tokenizer_name_or_path", 
        type=str, 
        default="/data/llmbase/ljw/tokenization/models/qwen3-8b",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/data/llmbase/ljw/tokenization/models/qwen3-8b",
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="/data/llmbase/ljw/hf_cache"
    )
    parser.add_argument(
        "--init_strategy", 
        type=str, 
        choices=["mean", "first"],
        default="mean",
        help="The strategy to initialize the target model's embeddings."
    )
    args = parser.parse_args()
    main(args)
