# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import os
import json
import pandas as pd
from llama import Llama
from typing import List
from torch.distributed.elastic.multiprocessing.errors import record
import torch

def data_process_boolq(path):
    yn_question_list = []
    answer_list = []
    with open(os.path.join(path,"boolq_train.jsonl")) as dev:
        for line in dev.readlines():
            line = json.loads(line)
            yn_question_list.append(line['question'])
            answer_list.append(line['answer'])
    
    answer_dict = {'true':'yes','false':'no'}
        
    #answer_list = [answer_dict[item.lower()] for item in answer_list]

    return yn_question_list,answer_list


def generate_prompt(question: str, number:str) -> str:
    
##Yes-no question: "Is there a duke of oxford?"
##new question: "Who is the duke of oxford?"

#{number} examples are given as references.
### Examples:

#Yes-no question: "Can betta fish survive without oxygen?"
#new question: "How do betta fish survive without oxygen?"

#Yes-no question: "Is there a duke of oxford?"
#new question: "Who is the duke of oxford?"

#Yes-no question: "Does finn die in star wars?"
#new question: "when does finn die in star wars?"

#Yes-no question: "Was titanic built in southampton?"
#new question: "Where was the titanic built southampton?"

#Yes-no question: "Is canola oil made from corn?"
#new question: "What is canola oil made from corn?"

    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    
Transform yes-no questions into new questions starting with any of the following words: "What", "Which", "Who", "When", "Where", "Why" and "How".

#Yes-no question: "Is a drive belt the same as a cambelt?" => #new question: "Why is a drive belt the same as a cambelt?"

#Yes-no question: "Is the isle of man part of great britain?" => #new question: "When is the isle of man part of great britain?"

#Yes-no question: "Is new york movie based on a true story?" => #new question: "which true story is new york movie based on?"

#Yes-no question: "Can you buy liquor at walmart in kansas?" => #new question: "Where can you buy liquor at walmart in kansas?"

#Yes-no question: "Is there a duke of oxford?" => #new question: "Who is the duke of oxford?"

#Yes-no question: "Is there a town called inverness in australia?" => #new question: "What is the town called inverness in australia?"

#Yes-no question: "Is hanger steak and skirt steak the same?" => #new question: "How are hanger steak and skirt steak the same?"

#Yes-no question: "{question.capitalize()}?" => #new question:"""


@record
def main(
    ckpt_dir: str="llama-2-7b/",
    tokenizer_path: str="tokenizer.model",
    dataset_path: str = "dataset/",
    store_path: str = "transformation_llama_results/",
    few_shot_number: int = 7,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_gen_len: int = 16,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 

    yn_questions, labels = data_process_boolq(dataset_path)

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )


    # prompts: List[str] = [
    #     # For these prompts, the expected answer is the natural continuation of the prompt
    #     "I believe the meaning of life is",
    #     "Simply put, the theory of relativity states that ",
    #     """A brief message congratulating the team on the launch:

    #     Hi everyone,
        
    #     I just """,
    #     # Few shot prompt (providing a few examples before asking model to complete more);
    #     """Translate English to French:
        
    #     sea otter => loutre de mer
    #     peppermint => menthe poivrée
    #     plush girafe => girafe peluche
    #     cheese =>""",
    # ]
    task = f'llama_prompting_transform_ynquestion_to_whquestion_boolq_indomain_{few_shot_number}_shot'
    prompts_list = []
    results = []
    for batch in range(int(len(yn_questions)/max_batch_size)):
        prompts = []                                                        

        for yn_question, label in zip(yn_questions[batch*max_batch_size:(batch+1)*max_batch_size],labels[batch*max_batch_size:(batch+1)*max_batch_size]):
            prompt = generate_prompt(yn_question, few_shot_number)
            prompts.append(prompt)
            prompts_list.append(prompt)                                              

       
        result = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        results.extend(result)
        #print(f"number of prompts: {len(prompts)}")
    for prompt, result,yn_question, label in zip(prompts_list, results,yn_questions,labels):
        print(f"----------PROMPT-----------\n{prompt}\n----------END-------------\n")
        print(label)
        print(f"> {result['generation']}")
        print("\n==================================\n")
        with open (f'{store_path}/generated_question_{task}.txt','a') as wf:
            wf.write('\n')
            wf.write(f'input yn question: {yn_question}\n')
            wf.write(f'answer to yn question: {label}\n')
            wf.write(f"generated wh question: {result['generation']}\n")
     
            wf.close()

if __name__ == "__main__":
    #torch.set_num_threads=1
    fire.Fire(main)

