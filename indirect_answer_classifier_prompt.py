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


def data_process_qa2(path):

    qa2_train = pd.read_csv(os.path.join(path,"QAQA_adaptation_set_Dec2022.csv"))
    train_wh_question = qa2_train['question'].tolist()
    train_yn_question = qa2_train['yesno_verification_question_including_valid_qs'].tolist()

    qa2_test = pd.read_csv(os.path.join(path,"QAQA_evaluation_set_Dec2022.csv"))
    test_wh_question = qa2_test['question'].tolist()
    test_yn_question = qa2_test['yesno_verification_question_including_valid_qs'].tolist()
    test_label = qa2_test['all_assumptions_valid'].tolist() 
    # qa2_test = pd.read_csv(os.path.join(path,"qa2_test.csv"))
    # test_texts = qa2_test['question'].tolist()
    # test_labels = qa2_test['yesno_verification_question_including_valid_qs'].tolist()
    
 
    # train_labels =  [qa2_label_dict[item] for item in train_labels]
    # val_labels =  [qa2_label_dict[item] for item in val_labels]
    # test_labels =  [qa2_label_dict[item] for item in test_labels]

    return train_wh_question, train_yn_question, test_wh_question, test_yn_question, test_label



def data_process_crepe(path):
    train_texts = []
    train_labels = []
    val_texts = []
    val_labels = []
    test_texts = []
    test_labels = []
    with open(os.path.join(path,"crepe_train.jsonl")) as t:
        for line in t.readlines():
            line =  json.loads(line)

            train_texts.append(line['question'])
            
            if(line['labels'] == ['normal']):
                train_labels.append(1)
            elif(line['labels'] == ['false presupposition']):
                train_labels.append(0)
            else:
                train_labels.append(1)

    with open(os.path.join(path,"crepe_dev.jsonl")) as t:
        for line in t.readlines():
            line =  json.loads(line)

            val_texts.append(line['question'])
            if(line['labels'] == ['normal']):
                val_labels.append(1)
            elif(line['labels'] == ['false presupposition']):
                val_labels.append(0)
            else:
                val_labels.append(1)


    with open(os.path.join(path,"crepe_test.jsonl")) as t:
        for line in t.readlines():
            line =  json.loads(line)

            test_texts.append(line['question'])
            if(line['labels'] == ['normal']):
                test_labels.append(1)
            elif(line['labels'] == ['false presupposition']):
                test_labels.append(0)
            else:
                test_labels.append(1)



    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels




def data_process_falseqa(path):
    falseqa_train = pd.read_csv(os.path.join(path,"falseqa_train.csv"))
    train_texts = falseqa_train['question'].tolist()
    train_labels = falseqa_train['label'].tolist()
    
    falseqa_val = pd.read_csv(os.path.join(path,"falseqa_valid.csv"))
    val_texts = falseqa_val['question'].tolist()
    val_labels = falseqa_val['label'].tolist()
    
    falseqa_test = pd.read_csv(os.path.join(path,"falseqa_test.csv"))
    test_texts = falseqa_test['question'].tolist()
    test_labels = falseqa_test['label'].tolist()
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


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


Transform a given question into a statement, like the following examples.

#question: "Why is a drive belt the same as a cambelt?" => #statement: "A drive belt is the same as a cambelt."

#question: "When is the isle of man part of great britain?" => #statement: "The isle of man is part of great britain."

#question: "which true story is new york movie based on?" => #statement: "New york movie is based on true story."

#question: "Where can you buy liquor at walmart in kansas?" => #statement: "You can buy liquor at walmart in kansas."

#question: "Who is the duke of oxford?" => #statement: "There is a duke of oxford."

#question: "What is the town called inverness in australia?" => #statement: "There is town called inverness in australia."

#question: "How are hanger steak and skirt steak the same?" => #statement: "Hanger steak and skirt steak are the same."

#question: "{question.capitalize()}" => #statement:"""


@record
def main(
    ckpt_dir: str="llama-2-7b/",
    tokenizer_path: str="tokenizer.model",
    dataset_path: str = "dataset/",
    test_dataset: str = "falseqa",
    split: str = "test",
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

    #yn_questions, labels = data_process_boolq(dataset_path)

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
    

    if test_dataset =='qa2':
        train_wh_question, train_yn_question, test_question, test_yn_question,test_labels = data_process_qa2(dataset_path)
    elif test_dataset == 'crepe':
        train_question,train_labels,val_question,val_labels,test_question,test_labels = data_process_crepe(dataset_path)
    elif test_dataset == 'falseqa':
        train_wh_question,train_labels,val_question,val_labels,test_question,test_labels = data_process_falseqa(dataset_path)
    
    elif test_dataset == 'boolq':                                             

        test_wh_question, labels = data_process_boolq(dataset_path)
    
    task = f'llama7b_prompting_transform_whquestion_to_statement_{test_dataset}_{split}_dataset_{few_shot_number}_shot_32_gen_tokens'
    if split == "test":
        test_wh_question = test_question
        labels = test_labels
    


    prompts_list = []
    results = []
   
    
    for batch in range(int(len(test_wh_question)/max_batch_size)):
        prompts = []                                                        

        for wh_question, label in zip(test_wh_question[batch*max_batch_size:(batch+1)*max_batch_size],labels[batch*max_batch_size:(batch+1)*max_batch_size]):
            prompt = generate_prompt(wh_question, few_shot_number)
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
    for prompt, result,wh_question, label in zip(prompts_list, results,test_wh_question,labels):
        print(f"----------PROMPT-----------\n{prompt}\n----------END-------------\n")
        print(label)
        print(f"> {result['generation']}")
        print("\n==================================\n")
        with open (f'{store_path}/generated_question_{task}.txt','a') as wf:
            wf.write('\n')
            wf.write(f'input yn question: {wh_question}\n')
            wf.write(f'answer to yn question: {label}\n')
            wf.write(f"generated wh question: {result['generation']}\n")
     
            wf.close()

if __name__ == "__main__":
    #torch.set_num_threads=1
    fire.Fire(main)

