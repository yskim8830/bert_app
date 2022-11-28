import json
import time
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .util import run_util
from .util.bert_util import bertQuestion
from .util.bert_util import convertQuestion
from .util.bert_util2 import bert2Question
from .learn import learning
from .learn import distribute

from django.http import Http404


class train(APIView):
    def post(self , request):
        data = json.loads(request.body) #파라미터 로드
        mode = str(data['mode'])
        siteNo = str(data['siteNo'])
        result_dic = {} #결과 set
        
        if mode == 'run' or mode == 'train':
            #질의를 embedding 하여 저장
            run = learning.learn('siteNo_'+siteNo)
            result_dic = run.learningBERT(data)
        
        elif mode == 'send' or mode == 'dist':
            # 생성된 사전 모델을 엘라스틱으로 insert
            send = distribute.dist('siteNo_'+siteNo)
            result_dic = send.distributeBERT(data)
            
        elif mode == 'status':
            result_dic = run_util.status(data)
            
        elif mode == 'runstop':
            run = learning.learn('siteNo_'+siteNo)
            print('stop main')
            result = run.raise_exception()
            if result == True :
                run.join()
                result_dic = {'result' :  'success'}
            else :
                result_dic = {'result' : 'fail, learning is not working'}
                
        elif mode == 'clear':
            result_dic = run_util.recoverySite(data)
            
        elif mode == 'dic':
            result_dic = run_util.save_dict(data)
            
        return Response(result_dic)
    
class question(APIView):
    def get(self , request):
        site_no = request.query_params.get('siteNo')
        question = request.query_params.get('query')
        searchip = request.query_params.get('searchIp')
        version = request.query_params.get('version')
        bert_query = bertQuestion(site_no, searchip, version)
        #result_answer = bert_query.question(question)
        result_answer = bert_query.question_vector(question)
        return Response(result_answer)
    
    def post(self , request):
        data = json.loads(request.body) #파라미터 로드
        site_no = str(data['siteNo'])
        question = str(data['query'])
        searchip = str(data['searchIp'])
        version = str(data['version'])
        bert_query = bertQuestion(site_no, searchip, version)
        #result_answer = bert_query.question(question)
        result_answer = bert_query.question_vector(question)
        return Response(result_answer)

class question2(APIView):
    def get(self , request):
        site_no = request.query_params.get('siteNo')
        question = request.query_params.get('query')
        bert_query = bert2Question(site_no)
        result_answer = bert_query.question(question)
        return Response(result_answer)
    
    def post(self , request):
        data = json.loads(request.body) #파라미터 로드
        site_no = str(data['siteNo'])
        question = str(data['query'])
        bert_query = bert2Question(site_no)
        result_answer = bert_query.question(question)
        return Response(result_answer)
 
class convert(APIView):
    def get(self , request):
        question = request.query_params.get('query')
        bert_query = convertQuestion()
        result_answer = bert_query.convert_vector(question)
        return Response(result_answer)
    
    def post(self , request):
        data = json.loads(request.body) #파라미터 로드
        question = str(data['query'])
        bert_query = convertQuestion()
        result_answer = bert_query.convert_vector(question)
        return Response(result_answer)