import logging
from datetime import datetime
from .const import const
from .es_util import elastic_util
logger = logging.getLogger('my')
index = const()
from .bert_learning import standard_bert

class bertQuestion2:
    def __init__(self, site_no, search_ip, version):
        es_urls = search_ip.split(':')
        self.es = elastic_util(es_urls[0], es_urls[1])
        
        global standard_bert
        self.site_no = site_no
        self.version = version
        self.model = standard_bert.esmodel
        self.device = standard_bert.device

    def question_vector(self, question):
        starttime = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        # corpus_embeddings = self.model.encode(question, convert_to_tensor=True, device=self.device)
        corpus_embeddings = self.model.encode(sentences=question, device=self.device)
        #개발 일 경우 version >-1 이상이고 version -1 일 경우 운영
        q_index = index.als_idx+index.question
        if int(self.version) > -1:
            q_index = index.dev_idx+index.question
        
        # total_results = self.es.search(q_index+str(self.site_no), self.es.question_vector_query(self.version, [vector.tolist() for vector in corpus_embeddings]))
        total_results = self.es.search(q_index+str(self.site_no), self.es.question_vector_query(self.version, corpus_embeddings))
        endtime = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        runtime = (datetime.strptime(endtime, '%Y%m%d%H%M%S%f')-datetime.strptime(starttime, '%Y%m%d%H%M%S%f')).total_seconds()
        
        results = []
        for idx, rst in enumerate(total_results):
            # results.append(rst)
            results.append({ "dialogNo" : rst['_source']['dialogNo'], "dialogNm" : rst['_source']['dialogNm']
                    , "score" : rst['_score'], "reliability" : "{:.2f}%".format(rst['_score'])})
        result = {"results" : results, "question" : question, "runtime" : runtime}      
        self.es.close()
        return result
        return {"result" : "blank"}
