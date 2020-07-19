import torch
import config

class EntityDataset:
    def __init__(self,texts,pos,tags):
        self.texts = texts  # tokenized based on space, list of lists [["this" ,"is", "sent", "1"],["this" ,"is", "sent", "2"]]
        self.pos = pos
        self.tags = tags
    
    def __len__(self):
        # number of samples/datapoints in dataset
        return len(self.texts)
    
    def __getitem__(self,item):
        text = self.texts[item]
        pos = self.pos[item]
        tags = self.tags[item]

        ids = []
        target_pos = []
        target_tag = []
        for idx,sent_ in enumerate(text):
            inputs = config.TOKENIZER.encode(sent_,add_special_toekns = False)
            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend([pos[idx] * input_len])
            target_tag.extend([tags[idx] * input_len])

        #padding
        ids = ids[:config.MAX_LEN -2 ]
        target_pos = target_pos[:config.MAX_LEN -2 ]
        target_tag = target_tag[:config.MAX_LEN -2 ]
        ids = ids[:config.MAX_LEN -2 ]

        ids = [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]
        mask = [1] *len(ids)

        taken_type_ids = [0]*len(ids) 

        padding_len = config.MAX_LEN - len(ids)


        ids = ids + ([0]*padding_len)
        mask = mask + ([0]*padding_len)

        token_type_ids = token_type_ids + ([0]*padding_len)
        target_pos = target_pos + ([0]*padding_len)
        target_tag = target_tag + ([0]*padding_len)

        return {
            "ids": torch.tensor(ids,dtype = torch.long),
            "mask": torch.tensor(mask,dtype = torch.long),
            "token_type_ids": torch.tensor(token_type_ids,dtype = torch.long),
            "target_pos": torch.tensor(target_pos , dtype = torch.long),
            "target_tag": torch.tensor(target_tag , dtype = torch.long),
        }

