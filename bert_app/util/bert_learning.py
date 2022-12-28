from django.apps import AppConfig
import configparser
import torch
import time
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import logging
import pandas as pd
import random
import math
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder

from sentence_transformers import SentenceTransformer, models
from ko_sentence_transformers.models import KoBertTransformer
import concurrent.futures as tpe
import ThreadPoolExecutorPlus

import json
logger = logging.getLogger('my')

## Setting parameters
max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

#서버 시작시 모델을 load 해서 global 변수에 담음.(--noreload 필수)
global standard_bert
global prochat_path

class BertAppConfig(AppConfig):
    name = 'bert_app'
    verbose_name = 'bert app'
    def ready(self):
        print('load model')
        if not os.environ.get('APP'):
            os.environ['APP'] = 'True'
            properties = configparser.ConfigParser()
            properties.read('prochat.ini')
            config = properties["CONFIG"] ## 섹션 선택
            global prochat_path
            prochat_path = properties["CONFIG"]["prochat_path"]
            berts = init_model(prochat_path)
            global standard_bert
            standard_bert = berts

#1. 모델 파일로 저장하는 학습
class bert_file:
    def __init__(self, site_no, path, question_file):
        self.site_no = site_no
        self.intent_path = path
        self.question_file = question_file
        
    def learning(self):
        ret = True
        try :
            global standard_bert
            self.device = standard_bert.device
            self.bertmodel = standard_bert.bertmodel
            self.vocab = standard_bert.vocab
            
            dataset_train = []
            dataset_test = []
            
            # 학습용 데이터셋 불러오기 
            dataset_train1 = pd.read_csv(os.path.join(self.intent_path,self.question_file))
            logger.info(dataset_train1.head())
            """
            #임의의 카테고리를 테스트용으로 셋 (수정 필요)
            data1 = dataset_train1[dataset_train1['dialogNm'] == '인터넷뱅킹_IM뱅크 신청']
            data2 = dataset_train1[dataset_train1['dialogNm'] == '폰뱅킹 신청_해지']
            data3 = dataset_train1[dataset_train1['dialogNm'] == '인터넷뱅킹 본인인증 문자']
            data4 = dataset_train1[dataset_train1['dialogNm'] == '일상대화_자기소개']
            data5 = dataset_train1[dataset_train1['dialogNm'] == '전화번호 조회_기획 담당자 및 연락처']
            new_data = data1.append([data2, data3, data4, data5], sort=False)
            new_data = pd.DataFrame(new_data)
            logger.info(new_data.head())
            """
            #임의의 카테고리를 테스트용으로 셋
            ran_list = []
            ran_size = 5
            if dataset_train1['intent'].size < 5:
                ran_size = dataset_train1['intent'].size
            ran_num = random.randint(0,dataset_train1['intent'].size)
            for i in range(ran_size):
                while ran_num in ran_list:
                    ran_num = random.randint(0,dataset_train1['intent'].size)
                ran_list.append(ran_num)
            ran_list.sort()

            data_list = []
            for i in range(ran_size):
                data_list.append(dataset_train1[dataset_train1['intent'] == dataset_train1['intent'][ran_list[i]]])
            new_data = pd.concat(data_list)
            logger.info(new_data.head())
            
            # 라벨링
            encoder = LabelEncoder()
            encoder.fit(dataset_train1['intent'])
            dataset_train1['intent'] = encoder.transform(dataset_train1['intent'])
            # dataset_train1.head()

            encoder_test = LabelEncoder()
            encoder_test.fit(new_data['intent'])
            new_data['intent'] = encoder_test.transform(new_data['intent'])
            # new_data.head()
            
            # 라벨링된 카테고리 매핑
            mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))
            mapping_len = len(mapping)
            logger.info('Mapping intent Length is ' + str(mapping_len))
            #BERT 데이터셋으로 만들기위해 리스트 형으로 형변환
            dataset_train = dataset_train1.values.tolist()
            dataset_test = new_data.values.tolist()
            
            tokenizer = get_tokenizer()
            tok = nlp.data.BERTSPTokenizer(tokenizer, self.vocab, lower=False)
            
            data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
            data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
            
            train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5, shuffle=True)
            
            model = BERTClassifier(self.bertmodel,  dr_rate=0.5, num_classes=mapping_len).to(self.device)
            
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            
            # 옵티마이저 선언
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
            loss_fn = nn.CrossEntropyLoss()
            
            t_total = len(train_dataloader) * num_epochs
            warmup_step = int(t_total * warmup_ratio)
            
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
            
            # 모델 학습 시작
            for e in range(num_epochs):
                train_acc = 0.0
                test_acc = 0.0
                
                model.train()
                for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
                    optimizer.zero_grad()
                    token_ids = token_ids.long().to(self.device)
                    segment_ids = segment_ids.long().to(self.device)
                    valid_length= valid_length
                    label = label.long().to(self.device)
                    out = model(token_ids, valid_length, segment_ids)
                    loss = loss_fn(out, label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # gradient clipping
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    train_acc += calc_accuracy(out, label)
                    if batch_id % log_interval == 0:
                        print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
                logger.info("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
                
                model.eval() # 평가 모드로 변경
                
                for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
                    token_ids = token_ids.long().to(self.device)
                    segment_ids = segment_ids.long().to(self.device)
                    valid_length= valid_length
                    label = label.long().to(self.device)
                    out = model(token_ids, valid_length, segment_ids)
                    test_acc += calc_accuracy(out, label)
                logger.info("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
            ##모델 학습 끝
            
            #모델 저장 (elasticsearch에저장 가능?)
            torch.save(model.state_dict(), os.path.join(self.intent_path,'learningModel_'+self.site_no+'.pt'))

        except Exception as e:
            ret = False
            print(e)
            if(e.msg is None):
                logger.error("[trainToDev] BERT error Msg : "+ e)
            else:
                logger.error("[trainToDev] BERT error Msg : "+ e.msg)
        finally:
            return ret
        
#2. 모델 Elasticsearch로 저장하는 학습
class bert_es:
    def __init__(self, site_no, path, question_file):
        self.site_no = site_no
        self.intent_path = path
        self.question_file = question_file
        
    def wordEmbedding(self):
        global standard_bert
        self.model = standard_bert.esmodel
        self.device = standard_bert.device
        # 학습용 데이터셋 불러오기 
        dataset_train1 = pd.read_csv(os.path.join(self.intent_path,self.question_file))
        vectors = {}
        logger.info('os.cpu_count : ' + str(os.cpu_count()))
        print('os.cpu_count : ',str(os.cpu_count()))
        total_size = dataset_train1.shape[0]
        max_job = 100 # 1개의 스레드당 할당되는 embedding 대상 데이터 최소 갯수
        ##data Frame question vector 변환
        #worker = math.ceil(total_size / max_job)
        
        start = time.time()
        print(start)
        
        def embedding(ids):
            sub = {}
            num = 1
            for question , intent  in dataset_train1.values.tolist()[ids*max_job:(ids+1)*max_job]:
                if num % 100 ==0 :
                    print('[thread no. ' + str(ids) + ' count ' +str(num)+' complated]', end='')
                    logger.info('thread no. ' + str(ids) + ' count ' +str(num)+' complated')
                corpus_embeddings = self.model.encode(question, convert_to_tensor=True, device=self.device)
                #sub_vector.append(corpus_embeddings.tolist())
                sub[question] = corpus_embeddings.tolist()
                num = num +1
            return sub
        
        #너무 많은 양의 스레드 생성시 컨텍스트 전환이 너무 많이 발생하여 오히려 더 느릴수 있음.
        #ThreadPoolExecutor
        
          
        with ThreadPoolExecutorPlus.ThreadPoolExecutor() as executor:
            worker = executor._max_workers 
            worker = 10
            if total_size > (max_job * worker):
                max_job = math.ceil(total_size / worker)
            logger.info('Set 1 Core embedding Max Job : ' + str(max_job))
            print('Set 1 Core embedding Max Job : ', str(max_job))
            future_to_map = {executor.submit(embedding, w): w for w in range(worker)}
            for future in tpe.as_completed(future_to_map):
                try:
                    #vectors += future.result()
                    vectors.update(future.result())
                except Exception as exc:
                    print('%r generated an exception: %s' % (exc))

            #vectors = [f.result() for f in future_to_map]
        """  
        #ProcessPoolExecutor
        with tpe.ProcessPoolExecutor() as executor:
            worker = executor._max_workers
            for result in executor.map(embedding, range(worker)):
                vectors += result 
        """
        #results = {**dataset_train1.to_dict('records'), **vectors}
        end = time.time()
        print("병렬처리 수행 시각", end-start, 's')
        
        results = []
        records = dataset_train1.to_dict('records')
        for value in records:
            result = {}
            result['question'] = value['question']
            result['intent'] = value['intent']
            result['vector'] = vectors[value['question']]
            results.append(result)
        #dataset_train1['vector'] = vectors
        #dataset_train1.head()
        
        return results

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
    
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes = 0, # softmax 사용 <- binary일 경우는 2
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.num_classes = num_classes
                 
        self.classifier = nn.Linear(hidden_size , self.num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
# 학습 평가 지표인 accuracy 계산 -> 얼마나 타겟값을 많이 맞추었는가
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def set_init():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info('device set GPU')
    else:
        device = torch.device('cpu')
        logger.info('device set CPU')
    bertmodel, vocab = get_pytorch_kobert_model()
    
    return device, bertmodel, vocab

class init_model:
    #device를 설정하고 KOBERT 를 로드한다.
    def __init__(self, path):
        if torch.cuda.is_available():
            self.__device = torch.device("cuda:0")
            logger.info('device set GPU')
        else:
            self.__device = torch.device('cpu')
            logger.info('device set CPU')
        logger.info('INIT MODEL')
        #model 파일용 bert 모델 로드
        self.__bertmodel, self.__vocab = get_pytorch_kobert_model()
        self.__bert_data =  get_model(path, self.__device,self.__bertmodel, self.__vocab)
        
        #es 학습용 bert 모델 로드
        word_embedding_model = KoBertTransformer('monologg/kobert', max_seq_length=75)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.__device)
        self.__es_model =  model
    
    @property
    def device(self):
        return self.__device
    @device.setter
    def device(self, str):
        self.__device = str
    
    @property
    def esmodel(self):
        return self.__es_model    
    @property
    def bertmodel(self):
        return self.__bertmodel
    @property
    def vocab(self):
        return self.__vocab
    @property
    def bert_data(self):
        return self.__bert_data

def get_model(prochat_path, device, bertmodel, vocab):
    #read learning file
    file_name = '@prochat_dialog_question'
    file_ext = r".csv"
    file_list = [_ for _ in os.listdir(prochat_path) if _.endswith(file_ext) & _.startswith(file_name)]
    #print('pid : ', os.getpid(), ' : ', file_list)
    berts_data = {}
    for question_file in file_list:
        site_no = question_file.replace(file_name+'_', '').replace(file_ext, '')
        if os.path.isfile(os.path.join(prochat_path,'learningModel_'+site_no+'.pt')) :
            dataset_cate = pd.read_csv(os.path.join(prochat_path,question_file))
            
            # 라벨링
            encoder = LabelEncoder()
            encoder.fit(dataset_cate['intent'])
            dataset_cate['intent'] = encoder.transform(dataset_cate['intent'])
            
            # 라벨링된 카테고리 매핑
            mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))
            mapping_len = len(mapping)
            
            tokenizer = get_tokenizer()
            tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
            
            modelload = BERTClassifier(bertmodel,  dr_rate=0.5, num_classes=mapping_len).to(device)
            modelload.load_state_dict(torch.load(os.path.join(prochat_path,'learningModel_'+site_no+'.pt'), device))
            modelload.eval()
            data_set = {'modelload' : modelload, 'mapping' : mapping, 'tok' : tok, 'device' : device}
            logger.info('bert data set, site_no is ' + str(site_no) + ' : load success.')
            
            berts_data[site_no] = data_set
    return berts_data

