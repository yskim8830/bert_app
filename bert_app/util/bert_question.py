import torch
import gluonnlp as nlp
import numpy as np
from datetime import datetime
import os
import logging
from .bert_learning import standard_bert, prochat_path
logger = logging.getLogger('my')

## Setting parameters
max_len = 64

#질의의 유사도를 계산하여 인텐트를 직접 리턴함.
class bertQuestion:
    def __init__(self, site_no):
        #from ..apps import get_model
        global standard_bert
        self.site_no = site_no
        self.data_set = standard_bert.bert_data[site_no]
        global prochat_path
        
    def question(self, question):
        starttime = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        self.modelload = self.data_set['modelload']
        self.mapping = self.data_set['mapping']
        self.tok = self.data_set['tok']
        self.devices = self.data_set['device']
        
        # self.modelload.load_state_dict(torch.load(os.path.join(prochat_path,'learningModel_'+self.site_no+'.pt'), self.devices))
        def getIntent(seq):
            cate = [self.mapping[i] for i in range(0,len(self.mapping))]
            tmp = [seq]
            transform = nlp.data.BERTSentenceTransform(self.tok, max_len, pad=True, pair=False)
            tokenized = transform(tmp)
            # self.modelload.eval()
            #result = self.modelload(torch.tensor([tokenized[0]]).to(self.devices), [tokenized[1]], torch.tensor(tokenized[2]).to(self.devices))
            result = self.modelload(torch.tensor(np.array(tokenized[0], ndmin = 2)).to(self.devices), [tokenized[1]], torch.tensor(np.array(tokenized[2], ndmin = 2)).to(self.devices))
            
            idx = result.argmax().cpu().item()
            score = softmax(result,idx)
            endtime = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
            runtime = (datetime.strptime(endtime, '%Y%m%d%H%M%S%f')-datetime.strptime(starttime, '%Y%m%d%H%M%S%f')).total_seconds()
            results = []
            results.append({"dialogNo" : idx, "dialogNm" : cate[idx], "score" : score
                      , "reliability" : "{:.2f}%".format(score)})
            result = {"results" : results, "question" : seq, "runtime" : runtime}
            #logger.debug("질의의 카테고리는:", answer["intent"])
            #logger.debug("신뢰도는:", answer["reliability"])
            
            return result
        
        return getIntent(question)

def softmax(vals, idx):
    valscpu = vals.cpu().detach().squeeze(0)
    a = 0
    for i in valscpu:
        a += np.exp(i)
    return ((np.exp(valscpu[idx]))/a).item() * 100

# 질의의 벡터를 리턴하는 클래스
class convertQuestion:
    def __init__(self):
        global standard_bert
        self.model = standard_bert.esmodel
    def convert_vector(self, question):
        corpus_embeddings = self.model.encode(question, convert_to_tensor=True)
        return [vector.tolist() for vector in corpus_embeddings]