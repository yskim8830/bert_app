import logging
from datetime import datetime
from . import string_util
from .const import const
from .es_util import elastic_util

logger = logging.getLogger('my')
index = const()
def isRunning(es, site_no):
    body = {
        "query": {
            "query_string": {
                "query": "siteNo:" + str(site_no) + " "
            }
        }
    }
    isLearnig = es.search(index.train_state,body)
    version = -1
    if len(isLearnig) > 0:
        siteInfo = isLearnig[0]['_source']
        if siteInfo['state'] == 'n':
            version = int(siteInfo['version'])
    else:
        version = 0
    return version

def createQuestionIndex(es, site_no):
    """
    if not es.existIndex("$dev_" + "model_" + str(site_no)):
        es.createindex("$dev_" + "model_" + str(site_no), '')
        es.createindex("$svc_" + "model_" + str(site_no) + "_0", '')
        es.createindex("$svc_" + "model_" + str(site_no) + "_1", '')
        es.createAlias("$als_" + "model_" + str(site_no) + "_1","$svc_" + "model_" + str(site_no) + "_1")
    """
    if not es.existIndex(index.dev_idx + index.intent + str(site_no)):
        es.createindex(index.dev_idx + index.intent + str(site_no), '')
        es.createindex(index.svc_idx + index.intent + str(site_no) + "_0", '')
        es.createindex(index.svc_idx + index.intent + str(site_no) + "_1", '')
        es.createAlias(index.als_idx + index.intent + str(site_no) ,index.svc_idx + index.intent + str(site_no) + "_1")
    
    try:
        if not es.existIndex(index.dev_idx + index.question + str(site_no)):
            es.createindex(index.dev_idx + index.question + str(site_no), es.question_index_template())
            es.createindex(index.svc_idx + index.question + str(site_no) + "_0", es.question_index_template())
            es.createindex(index.svc_idx + index.question+ str(site_no) + "_1", es.question_index_template())
            es.createAlias(index.als_idx + index.question + str(site_no) ,index.svc_idx + index.question + str(site_no) + "_1")
    except Exception as e:
        logger.error("elastiknn plugin not installed.", e)
        
def status(data):
    es_urls = str(data['esUrl']).split(':')
    site_no = data['siteNo']
    #검색엔진에 연결한다.
    es = elastic_util(es_urls[0], es_urls[1])
    version = isRunning(es, site_no)
    es.close()
    if version > -1 :
        return {'code' : 0, 'result' : 'site '+ str(site_no) +' is no running'}
    else :
        return {'code' : 1, 'result' : 'site '+ str(site_no) + ' is running'}

def save_dict(data):
    es_urls = str(data['esUrl']).split(':')
    #검색엔진에 연결한다.
    es = elastic_util(es_urls[0], es_urls[1])
    #ES 사전 정보 파일로 저장
    dicList = es.search_srcoll('@prochat_dic','')
    dic_path = data['dicPath']
    result = string_util.save_dictionary(dic_path,dicList)
    es.close()
    if not result:
        return {'result' : 'fail'}
    return {'result' : 'success'}

def recoverySite(data):
    #프로세서가 비정상 종료일시 상태값을 복구한다.
    
    #프로세서를 완전히 종료한다.
    
    
    #프로세서상태를 복구한다.
    try:
        es_urls = str(data['esUrl']).split(':')
        #검색엔진에 연결한다.
        es = elastic_util(es_urls[0], es_urls[1])
        
        mapData = {}
        mapData['id'] = data['siteNo']
        mapData['siteNo'] = data['siteNo']
        mapData['state'] = 'n'
        mapData['modify_date'] = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        es.updateData(index.train_state, data['siteNo'], mapData)
    except Exception as e:
        error_msg = str(e)
        logger.error(e)
        return {'result' : 'fail', 'error_msg' : error_msg}
    finally :
        es.close()
    return {'result' : 'success'}