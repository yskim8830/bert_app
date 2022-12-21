import os
import logging
import platform
import threading
import ctypes
from datetime import datetime
from konlpy.tag import Mecab
from ..util import run_util
from ..util import file_util
from ..util import string_util
from ..util.const import const
from ..util.file_util import dic
#from ..util.bert_util import bert
from ..util.bert_util2 import bert
from ..util.es_util import elastic_util

logger = logging.getLogger('my')
index = const()
class learn(threading.Thread):
    def __init__(self,name):
        threading.Thread.__init__(self)
        self.name = name #Thread Name
        
    def learningBERT(self, data):
        #debug = data['debug']
        
        es_urls = str(data['esUrl']).split(':')
        #검색엔진에 연결한다.
        es = elastic_util(es_urls[0], es_urls[1])
        
        site_no = data['siteNo']
        dic_path = data['dicPath']
        mecab_dic_path = '/usr/local/lib/mecab/dic'
        if data.get('mecabDicPath') != None :
            mecab_dic_path = data['mecabDicPath']
        userId = data['userId']
        
        error_msg = ""
        es.createindex(index.train_state,'') #$train_state 존재여부 확인 후 생성
        
        #현재 사이트가 학습 중 인지 확인한다.
        version = run_util.isRunning(es,site_no)
        modify_date = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        
        intentCount = 0
        questionCount = 0
        dialogCount = 0
        
        if version > -1:
            logger.debug("[trainToDev] model not running : "+str(site_no))
            # Index 생성여부를 확인하고 해당 데이터를 만든다.
            run_util.createQuestionIndex(es,site_no)
            # $train_state 상태를 업데이트 한다.
            mapData = {}
            mapData['id'] = site_no
            mapData['version'] = version ##학습중인 상태를 나타냄. -1
            mapData['siteNo'] = site_no
            mapData['state'] = 'y'
            mapData['modify_date'] = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
            es.insertData(index.train_state, site_no, mapData)
            
            #ES 사전 정보 파일로 저장
            dicList = es.search_srcoll('@prochat_dic','')
            result = string_util.save_dictionary(dic_path,dicList)
            if not result:
                return {'result' : 'fail'}
            #사전파일 가져오기
            dics = dic(dic_path)
            stopwords = dics.stopword()
            synonyms = dics.synonym()
            userdefine = dics.compound()
            
            try:
                #start_date = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
                new_version = version + 1 #업데이트 되는 버전 정보
                query_string = {
                    "query": {
                        "query_string": {
                            "query": "siteNo:" + str(site_no) + " AND useYn:y"
                        }
                    }
                }
                if userdefine.get(str(site_no)):
                    #사용자 사전 export
                    file_util.export_user_dic(mecab_dic_path,userdefine[str(site_no)])
                    #사용자 사전 적용 실행
                    if platform.system() == 'Windows':
                        file_util.run_power_shell(mecab_dic_path)
                    else:
                        file_util.run_lnx_shell(mecab_dic_path)

                #패턴
                patternMapList = es.search_srcoll('@prochat_dialog_pattern',query_string)
                logger.info("[trainToDev] pattern to dev start [ site : "+str(site_no) +" /  pattern count : "+str(len(patternMapList))+" ] ")
                patternData = dict()
                for pattern in patternMapList:
                    pattern = pattern['_source']
                    key = str(pattern['dialogNo'])
                    value = str(pattern['pattern']).replace(',','')
                    if key in patternData:
                        new_value = str(patternData[key]).lower()
                        new_value += ' ' + value
                        patternData[key] = new_value
                    else:
                        patternData[key] = value
                logger.info("[trainToDev] pattern to dev end [ site : "+str(site_no) +" ] ")
                        
                #인텐트
                m = Mecab(dicpath=os.path.join(mecab_dic_path, 'mecab-ko-dic')) # 사전 저장 경로에 자신이 mecab-ko-dic를 저장한 위치를 적는다. (default: "/usr/local/lib/mecab/dic/mecab-ko-dic") https://lsjsj92.tistory.com/612
                intentNameMap = dict()
                categoryMap = dict()
                intentMapList = es.search_srcoll('@prochat_dialog',query_string)
                intentCount = len(intentMapList)
                logger.info("[trainToDev] intent to dev start [ site : "+str(site_no) +" /  intent count : "+str(intentCount)+" ] ")
                devIntent = []
                questionList = []
                for intent in intentMapList:
                    intent = intent['_source']
                    id = str(intent['dialogNo'])
                    dialogNm = str(intent['dialogNm'])
                    
                    intentNameMap[id] = dialogNm
                    categoryMap[id] = str(intent['categoryNo'])
                    body = {}
                    _source = {}
                    
                    body['_index'] = index.dev_idx + index.intent + str(site_no)
                    body['_action'] = 'index'
                    body['_id'] = str(site_no)+'_'+str(new_version)+id
                    _source['dialogNo'] = id
                    _source['dialogNm'] = dialogNm
                    _source['siteNo'] = str(site_no)
                    _source['categoryNo'] = intent['categoryNo']
                    _source['desc'] = intent['desc']
                    if id in patternData:
                        _source['pattern'] = patternData[id]
                    else:
                        _source['pattern'] = ''
                    _source['version'] = new_version
                    _source['keywords'] = str(intent['keywords']).replace(',',' ')
                    
                    sentence = string_util.filterSentence(dialogNm.lower())
                    morphList = getWordStringList(m, sentence, 'EC,JX,ETN')
                    
                    #불용어 제거 stopwords
                    sList = []
                    for morph in morphList:
                        if stopwords.get(str(site_no)):
                            if morph not in stopwords[str(site_no)]:
                                sList.append(morph)
                        else:
                            sList.append(morph)
                    orgTerm = ' '.join(sList)
                    orgTermCnt = len(sList)
                    
                    #동의어 처리
                    reList = []
                    for morph in morphList:
                        if synonyms.get(str(site_no)):
                            if morph in synonyms[str(site_no)]: 
                                reList.append(synonyms[str(site_no)][morph])
                            else:
                                reList.append(morph)
                        else:
                            reList.append(morph)
                    synonymTerm = ' '.join(reList)
                    
                    #questionList.append(synonymTerm)
                    questionList.append(reList)
                    if(orgTerm != synonymTerm):
                        #questionList.append(orgTerm)
                        questionList.append(sList)
                        
                    _source['term'] = orgTerm
                    _source['term_syn'] = synonymTerm
                    _source['termNo'] = orgTermCnt
                    
                    body['_source'] = _source
                    devIntent.append(body)
                    
                try:
                    es.bulk(devIntent)
                except Exception as e:
                    logger.error("[trainToDev] template index not install : "+ e)
                logger.info("[trainToDev] intent to dev end [ site : "+str(site_no) +" ]")
                
                #질의 question
                questionMapList = es.search_srcoll('@prochat_dialog_question',query_string)
                #questionMapList = questionMapList[0:100] #갯수 제한(테스트용)
                logger.info("[trainToDev] question to dev start [ site : "+str(site_no) +" /  question count : "+str(len(questionMapList))+" ] ")
                devQuestion = []
                for question in questionMapList:
                    question = question['_source']
                    body = {}
                    _source = {}
                    
                    id = str(question['dialogNo'])
                    sentence = string_util.filterSentence(str(question['question']).lower())
                    morphList = getWordStringList(m, sentence, 'EC,JX,ETN')
                    #불용어 제거 stopwords
                    sList = []
                    for morph in morphList:
                        if stopwords.get(str(site_no)):
                            if morph not in stopwords[str(site_no)]:
                                sList.append(morph)
                        else:
                            sList.append(morph)    
                    orgTerm = ' '.join(sList)
                    orgTermCnt = len(sList)
                    
                    #동의어 처리
                    reList = []
                    for morph in morphList:
                        if synonyms.get(str(site_no)):
                            if morph in synonyms[str(site_no)]: 
                                reList.append(synonyms[str(site_no)][morph])
                            else:
                                reList.append(morph)
                        else:
                            reList.append(morph)
                    synonymTerm = ' '.join(reList)
                    
                    dialogNm = ''
                    if id in intentNameMap:
                        dialogNm = intentNameMap[id]
                    
                    categoryNo = '0'
                    if id in categoryMap:
                        categoryNo = categoryMap[id]
                    
                    body['_index'] = index.dev_idx + index.question + str(site_no)
                    body['_action'] = 'index'
                    body['_id'] = str(site_no)+'_'+str(new_version)+'_'+str(question['questionNo'])
                    
                    _source['questionNo']  = question['questionNo']
                    _source['question']  = question['question']
                    _source['version']  = new_version
                    _source['siteNo']  = site_no
                    _source['dialogNo']  = question['dialogNo']
                    _source['dialogNm']  = dialogNm
                    _source['categoryNo']  = int(categoryNo)
                    _source['termNo']  = orgTermCnt
                    _source['term']  = orgTerm
                    _source['term_syn']  = synonymTerm
                    _source['keywords']  = string_util.specialReplace(sentence).replace(' ','')
                    
                    #questionList.append(synonymTerm)
                    questionList.append(reList)
                    if(orgTerm != synonymTerm):
                        #questionList.append(orgTerm)
                        questionList.append(sList)
                        _source['terms']  = orgTerm.replace(' ','') + ' ' + synonymTerm.replace(' ','')
                    else:
                        _source['terms']  = orgTerm.replace(' ','')
                    
                    body['_source'] = _source
                    devQuestion.append(body)

                #학습데이터를 만든다. BERT
                logger.info("[trainToDev] BERT Train start [ siteNo :"+str(site_no)+" / size : "+str(len(questionList))+"]")
                
                #질의 파일 저장
                file_util.save_question_file(dic_path,'@prochat_dialog_question_'+site_no,questionMapList) #dialogNm, question 만 저장
                #file_util.save_question_file2(dic_path,'@prochat_dialog_question_'+site_no,questionMapList) #dialogNo, questionNo 까지 저장
                
                #저장한 파일 다시 읽어옴
                bertResult = bert(site_no, dic_path, '@prochat_dialog_question_'+site_no+'.csv')
                
                #분석방식 1,2
                bertResult.learning() #1. 모델 파일을 저장하는 로직
                #qVector = bertResult.wordEmbedding() #2. vector로 변환하여 elasticsearch 에 저장는 로직
                
                #학습결과를 ES 인덱스에 넣는다.
                """
                logger.info("[trainToDev] question to dev search vector list start [ site : "+str(site_no) +" ]")
                totalDevQuestion = []
                if len(devQuestion) > 0:
                    
                    #vector로 변환하여 elasticsearch 에 저장 할시
                    for question_row, vector in zip(devQuestion, qVector):
                            source = question_row['_source']
                            source['question_vec'] = vector['vector']
                            source['question_vec_2'] = vector['question']
                            question_row['_source'] = source
                            totalDevQuestion.append(question_row)
                    logger.info("complete set vector")
                    print('complete set vector')
                    
                    #vector 저장하지 않을시
                    #totalDevQuestion = devQuestion
                    
                    try:
                        es.bulk(totalDevQuestion)
                    except Exception as e:
                        logger.error(e)
                logger.info("[trainToDev] question to dev search vector list end [ site : "+str(site_no) +" /  count : "+str(len(devQuestion))+" ]")
                """        
                #모델 
                #모델 개수를 count 한다. (학습상관없음)
                dialogCount = es.countBySearch('@prochat_dialog_model', query_string)
                
                #학습이 완료 되었으므로 버전을 올린다.
                version = new_version
                
            except Exception as e:
                error_msg = str(e)
                logger.error(e)
            finally:
                end_date = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
                #$train_state 상태를 변경한다.
                mapData['version'] = version
                mapData['state'] = 'n'
                mapData['modify_date'] = end_date
                es.updateData('$train_state', site_no, mapData)
                
                
                #learning log에 학습결과를 적재한다.
                log_data = {}
                id = str(site_no) + '_' + str(version)
                log_data['learningLogNo'] = id
                log_data['version'] = version
                log_data['siteNo'] = site_no
                log_data['service'] = 'n'
                log_data['createUser'] = userId
                log_data['modifyUser'] = userId
                runtime = (datetime.strptime(end_date, '%Y%m%d%H%M%S%f')-datetime.strptime(modify_date, '%Y%m%d%H%M%S%f')).total_seconds()
                log_data['runtime'] = runtime
                log_data['runStartDate'] = modify_date
                log_data['runEndDate'] = end_date
                log_data['createDate'] = modify_date
                log_data['modifyDate'] = modify_date
                log_data['intentCnt'] = intentCount
                log_data['dialogCnt'] = dialogCount
                log_data['questionCnt'] = questionCount
                log_data['order'] = 0
                if error_msg != '':
                    log_data['state'] = 'error'
                    log_data['message'] = error_msg
                else:
                    log_data['state'] = 'success'
                es.insertData('@prochat_learning_log', id, log_data)
                es.close()
        elif version == -1:
            status = "[trainToDev] model running : "+str(site_no) +" [check $train_state index check]"
            logger.info(status)
            return {'result' : 'fail', 'error_msg' : status}
        return {'result' : 'success', 'version' : version}
    
    def get_id(self): 
        # returns id of the respective thread 
        if hasattr(self, '_thread_id'): 
            return self._thread_id 
        for id, thread in threading._active.items(): 
            if thread is self: 
                return id
   
    def raise_exception(self): 
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 
            ctypes.py_object(SystemExit)) 
        if res > 1: 
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0) 
            print('Exception raise failure')
            
def getWordStringList(mec, sentence, stopTag):
    result = []
    morphs = mec.pos(sentence)
    for word in morphs:
        if str(stopTag).find(word[1]) == -1:
            result.append(word[0])
    return result
                
    





    