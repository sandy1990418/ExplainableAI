from extra.logging import get_logger
from typing import Optional, Any, List, Tuple, Dict, Union
from dataclasses import dataclass, field
import torch
logger = get_logger(__name__)


def get_start_and_end_position(input_token,
                               input_ids)->Tuple[torch.LongTensor,torch.LongTensor]:
    """_summary_

    Args:
        input_token (_type_): _description_
        input_ids (_type_): _description_

    Returns:
        Tuple[torch.LongTensor,torch.LongTensor]: _description_
    
    
    Ref : https://huggingface.co/docs/transformers/tasks/question_answering
    """
    
    '''
    Demo 
    # mask = [i != 1 for i in inputs.sequence_ids()]
    # mask[0] = False
    # mask = torch.tensor(mask)[None]
    # inputs['mask'] = mask

    # inputs['answers']=input_ids['answers']

    # inputs['start_positions']=torch.tensor(inputs['answers']['answer_start'])
    # inputs['end_positions'] = torch.tensor([inputs['answers']['answer_start'][0]+len(inputs['answers']['text'][0])])
    '''
    ## Get Start and End position 
    start_positions = []
    end_positions = []
    for i, offset in enumerate(input_token["offset_mapping"]):
        sample_idx = input_token["overflow_to_sample_mapping"][i]
        start_char = input_ids['answers']["answer_start"][0]
        end_char = input_ids['answers']["answer_start"][0] + len(input_ids['answers']["text"][0])
        sequence_ids = input_token.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)


    ## get Mask 
    mask = [i != 1 for i in input_token.sequence_ids()]
    mask[0] = False
    mask = torch.tensor(mask)[None]
    return (torch.LongTensor(start_positions), torch.LongTensor(end_positions),mask)



'''

Text preprocess in Question-Answering task

'''
def qa_task_preprocess(tokenizer,
                    model_token,
                    input_ids,
                    padding:Optional[Any]="max_length",
                    doc_stride:Optional[int]=None, 
                    max_seq_len:Optional[int]=None,
                    return_token_type_ids:Optional[bool]=True,
                    return_overflowing_tokens:Optional[bool]=True,
                    return_offsets_mapping:Optional[bool]=True,
                    return_special_tokens_mask:Optional[bool]=False,
                    return_decode_text:Optional[bool]=False,
                    **preprocess_kwargs)-> Union[Dict[str, Any], tuple]:
    # The maximum length of a feature (question and context)
    if max_seq_len is None:
        max_seq_len = min(tokenizer.model_max_length, 384)
    # The authorized overlap between two part of the context when splitting it is needed.
    if doc_stride is None:
        doc_stride = min(max_seq_len // 2, 128)
    question_first = tokenizer.padding_side == "right"

    logger.info('Tokenizer data')
    if 'gpt' in model_token:
        logger.info('Model is GPT model and tokenize input')
        tokenizer.pad_token = tokenizer.eos_token

    elif 'bert' in model_token  :   # or 'Bert' in model.config.tokenizer_class
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    inputs=tokenizer(
            text = input_ids['question'] if question_first else input_ids['context'],
            text_pair = input_ids['context'] if  input_ids['context'] else  input_ids['question'],
            padding=padding,
            return_tensors='pt',
            truncation="only_second" if question_first else "only_first",
            max_length=max_seq_len,
            stride=doc_stride,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_offsets_mapping=return_offsets_mapping,
            return_special_tokens_mask=return_special_tokens_mask,
            **preprocess_kwargs
        )


    logger.info('Get Start and End position in QA task')

    inputs['start_positions'],inputs['end_positions'] ,inputs['mask']=get_start_and_end_position(inputs,input_ids)
    
    decode_text=[]
    for i in range(inputs['input_ids'][0].shape[0]):
        decode_text.extend([tokenizer.decode(inputs['input_ids'][0][i])])

    return inputs,decode_text if return_decode_text else inputs


'''

Text preprocess in Translation or Summarization task

'''
def translation_summarization_task_preprocess(tokenizer,
                                            model_token,
                                            input_ids,
                                            targets,
                                            return_decode_text:Optional[bool]=False,
                                            **preprocess_kwargs)-> Union[Dict[str, Any], tuple]:
        
    if 't5' in model_token:        
        inputs=tokenizer(
                        text = input_ids ,
                        return_tensors='pt',
                        **preprocess_kwargs, 
                        )

        ## Ref : https://github.com/huggingface/transformers/issues/18455
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(text = targets, 
                                return_tensors='pt',
                                **preprocess_kwargs, 
                                )
            
        inputs["labels"] = targets["input_ids"]
        logger.info('Get Translation or Summarization task')
        decode_text=[]
        for i in range(inputs['input_ids'][0].shape[0]):
            decode_text.extend([tokenizer.decode(inputs['input_ids'][0][i])])

        return inputs,decode_text if return_decode_text else inputs
