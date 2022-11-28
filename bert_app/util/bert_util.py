from django.apps import AppConfig
import configparser
import torch
import time
import os
import logging
import math
import pandas as pd

from sentence_transformers import SentenceTransformer, models
from ko_sentence_transformers.models import KoBertTransformer

import concurrent.futures as tpe
import ThreadPoolExecutorPlus
from .const import const
logger = logging.getLogger('my')
index = const()
#서버 시작시 모델을 load 해서 global 변수에 담음.(--noreload 필수)
global standard_bert

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
            prochat_path = properties["CONFIG"]["prochat_path"]
            berts = init_model(prochat_path)
            global standard_bert
            standard_bert = berts

class bert:
    def __init__(self, site_no, path, question_file):
        self.site_no = site_no
        self.intent_path = path
        self.question_file = question_file
        
    def wordEmbedding(self):
        global standard_bert
        self.model = standard_bert.bertmodel
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
    
class bertQuestion:
    def __init__(self, site_no, search_ip, version):
        from ..util.es_util import elastic_util
        es_urls = search_ip.split(':')
        self.es = elastic_util(es_urls[0], es_urls[1])
        
        global standard_bert
        self.site_no = site_no
        self.version = version
        self.model = standard_bert.bertmodel

    def question_vector(self, question):
        corpus_embeddings = self.model.encode(question, convert_to_tensor=True)
        #return [vector.tolist() for vector in corpus_embeddings]
        #개발 일 경우 version >-1 이상이고 version -1 일 경우 운영
        q_index = index.als_idx+index.question
        if int(self.version) > -1:
            q_index = index.dev_idx+index.question
        result = self.es.search(q_index+str(self.site_no), self.es.question_vector_query(self.version, [vector.tolist() for vector in corpus_embeddings]))
        self.es.close()
        return result

class convertQuestion:
    def __init__(self):
        global standard_bert
        self.model = standard_bert.bertmodel
    def convert_vector(self, question):
        corpus_embeddings = self.model.encode(question, convert_to_tensor=True)
        return [vector.tolist() for vector in corpus_embeddings]

class init_model:
    def __init__(self, path):
        if torch.cuda.is_available():
            self.__device = "cuda:0"
            logger.info('device set GPU')
        else:
            self.__device = 'cpu'
            logger.info('device set CPU')
        logger.info('INIT MODEL')
        word_embedding_model = KoBertTransformer('monologg/kobert', max_seq_length=75)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=self.__device)
        
        self.__bert_model =  model
    @property
    def bertmodel(self):
        return self.__bert_model
    @property
    def device(self):
        return self.__device
