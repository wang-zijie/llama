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


def data_process_statement(path):
    statement_list = []
    #answer_list = []
    with open(os.path.join(path,"llama combined test prompt.txt")) as f:
        for line in f.readlines():
            line = line.split('\n')[0]
            line = line.split('. ')[1]
            statement_list.append(line)

    # with open(os.path.join(path,"boolq_train.jsonl")) as dev:
    #     for line in dev.readlines():
    #         line = json.loads(line)
    #         yn_question_list.append(line['question'])
    #         answer_list.append(line['answer'])
    
    # answer_dict = {'true':'yes','false':'no'}
        
    #answer_list = [answer_dict[item.lower()] for item in answer_list]

    return statement_list


def generate_prompt(statement: str) -> str:
    
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
## McCartney, in Interview, Compares Global Warming Skeptics to Holocaust Deniers Sir Paul McCartney just can't let it be. => ## CerPos

## Results like these indicate that investing in soil-building practices would help feed a warming world. => ## CerPos

## The odds for such an extreme drought are "highly unlikely" without factoring in what scientists like to call "anthropogenic" effects that include long-term warming trends in the Eastern Mediterranean and declines in soil moisture, according to this paper. => ## CerPos

## Global temperatures are on the rise, sir, Mair answered, sticking to the party line. => ## CerPos

## It's true that the Russians are earning money from oil and gas, but to compound that problem by accelerating oil and gas in America would go against the climate goals, and climate is like war: If we don't handle it, people are going to die and they're going to be suffering. => ## CerPos

## According to meteorologist Joe D'Aleo, who co-authored the study with statistician James Wallace and Cato Institute climate scientist Craig Idso, this has the effect of exaggerating the warming trend: "Nearly all of the warming they are now showing are in the adjustments." => ## CerPos

## These words came just two days after a group of climate scientists released findings that one of the most cited examples of accelerated global warmingthe Antarctic Peninsulahad nothing to do with human behavior whatsoever, but was "entirely consistent with natural climate variability." => ## CerPos
## Rubio focused on the "mitigation" action he has pushed for to address issues like rising sea levels while refusing to address the root causes of climate change. => ## CerNeg

## The Post claims drought, heat waves, and wildfires in the western United States are the result of climate change. => ## CerNeg

## The Earth system is just too complex to be represented in current climate models. => ## CerNeg

## Koonin, who believes that human activity is influencing climate change, but is critical of the way climate data are presented to the public, writes in a chapter called "Hyping the Heat" that "there are high levels of uncertainty in detecting trends in extreme weather. => #CerNeg

## It also targets the U.S. for not reducing emissions enough "because misinformation about climate change and the politicization of climate science has caused widespread public confusion about the true risks of global warming," NPR said the report charged. => ## CerNeg

## Greenpeace co-founder: No scientific proof humans are dominant cause of warming climate A co-founder of Greenpeace told lawmakers there is no evidence man is contributing to climate change, and said he left the group when it became more interested in politics than the environment. => ## CerNeg

## The Syrian Civil War Was Not Caused By Climate Change. => ## CerNeg

## Any theory needing to rely so consistently on fudging the evidence, I concluded, must be looked on not as science at all, but as simply a rather alarming case study in the aberrations of group psychology. => #CerNeg

    return f"""Let's start a new categorization exercise. This one will focus on the vested interests variable certainty. The goal will be to label sentences according to whether they promote or undermine beliefs in the certainty of climate change. I will follow this with two prompts, one defining a CerPos label for statements that promote belief in the certainty of climate change, and second defining a CerNeg label for statements that undermine a belief in the certainty of climate change. After that I will give you several sets of sentences to label CerPos, CerNeg, or n/a (not applicable) based on these definitions.

Some statements promote the belief that we are certain climate change is real and having negative effects. Such statements should be labeled CerPos. Some statements undermine the belief that we are certain climate change is real and having negative effects. Such statements should be labeled CerNeg. 

## Russell Crowe has avowed that "the tragedy unfolding in Australia is climate change-based" and Cate Blanchett stressed that "when one country faces a climate disaster, we all face a climate disaster, so we're in it together." https://t.co/IGhD0f0PmE  Breitbart News (@BreitbartNews) September 10, 2021 " => ## CerPos

## G20 members are responsible for over 80 per cent of global emissions. => ## CerPos 

## He's been saying for over a quarter of a century that the whole global warming thing is a scam, but hardly anyone has been listening for reasons we'll come to in a moment. => ## CerNeg

## But at least we climate skeptics have been proved right yet again, that's the main thing. => ## CerNeg
 
## {statement} => ## """


@record
def main(
    ckpt_dir: str="llama-2-7b/",
    tokenizer_path: str="tokenizer.model",
    dataset_path: str = "dataset/",
    store_path: str = "transformation_llama_results/",
    few_shot_number: int = 7,
    temperature: float = 0.6, #0.6
    top_p: float = 0.9, #0.9
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
    statements = data_process_statement(dataset_path)

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
    ckpt_dir = str(ckpt_dir).split("/")[0]
    task = f"{ckpt_dir}_prompting_onr_labeling_statement_19shot"
    prompts_list = []
    results = []
    for batch in range(int(len(statements)/max_batch_size)):
        prompts = []                                                        

        for statement in statements[batch*max_batch_size:(batch+1)*max_batch_size]:
            prompt = generate_prompt(statement)
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
    for prompt, statement, result in zip(prompts_list, statements, results):
        print(f"----------PROMPT-----------\n{prompt}\n----------END-------------\n")
        # print(label)
        print(f"> {result['generation']}")
        print("\n==================================\n")
        with open (f'{store_path}/generated_question_{task}.txt','a') as wf:
            wf.write('\n')
            wf.write(f'input statement: {statement}\n')
            wf.write(f"predicted label: {result['generation']}\n")
     
            wf.close()

if __name__ == "__main__":
    #torch.set_num_threads=1
    fire.Fire(main)

