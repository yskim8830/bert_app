from elasticsearch import Elasticsearch, helpers

class elastic_util:
    def __init__(self, host='127.0.0.1', port='6251'):
        self.host = host
        self.port = port
        server_list = [ {'host':host, 'port':port}]
        self.es = Elasticsearch( server_list )
    
    #get es object
    def getEs(self):
        return self.es
    
    #getHealth
    def getHealth(self):
        return self.es.indices()
    
    #getInfo
    def getInfo(self):
        return self.es.info()
    
    #existIndex
    def existIndex(self, idx):
        return self.es.indices.exists(index=idx)
    
    #createindex
    def createindex(self, idx, mapping):
        if self.es.indices.exists(index=idx):
            pass
        else:
            return self.es.indices.create(index=idx, body=mapping)
    
    #deleteindex
    def deleteindex(self, idx):
        return self.es.indices.delete(index=idx, ignore=[400, 404])
    
    #existAlias
    def getAlias(self, aidx):
        return self.es.indices.get_alias(name=aidx)
    
    #createAlias
    def createAlias(self, aidx, idx):
        if self.es.indices.exists_alias(name=aidx):
            pass
        return self.es.indices.put_alias(name=aidx, index=idx)
    
    #changeAlias : alias, index, removeIndex
    def changeAlias(self, aidx, idx, ridx):
        body = {
            "actions": [
                {
                "remove": {
                    "index": ridx,
                    "alias": aidx
                }
                },
                {
                "add": {
                    "index": idx,
                    "alias": aidx
                }
                }
            ]
        }
        
        return self.es.indices.update_aliases(body=body)
        
    #reIndex
    def reIndex(self, sidx, tidx, mapping):
        body = {
            "source": {
                "index": sidx,
                "query": mapping['query']
            },
            "dest": {
                "index": tidx
            }
        }
        self.es.reindex(body=body, request_timeout=1000)
    
    #searchAll
    def searchAll(self, idx, size=10):
        response = self.es.search(index=idx, size=size, body={"query": {"match_all": {}}})
        fetched = len(response['hits']['hits'])
        result = []
        for i in range(fetched):
            result.append(response['hits']['hits'][i])
        return result
    
    #searchById
    def searchById(self, idx, id):
        response = self.es.search(index=idx, body={"query": {"match": { "_id" : id}}})
        fetched = len(response['hits']['hits'])
        result = []
        for i in range(fetched):
            result.append(response['hits']['hits'][i])
        return result
    
    #search
    def search(self, idx, body):
        response = self.es.search(index=idx, body=body)
        fetched = len(response['hits']['hits'])
        result = []
        for i in range(fetched):
            result.append(response['hits']['hits'][i])
        return result
    
    #countBySearch
    def countBySearch(self, idx, body):
        response = self.es.count(index=idx, body=body)
        return response['count']
    
    #scroll search
    def search_srcoll(self, idx, body):
        _KEEP_ALIVE_LIMIT='30s'
        response = self.es.search(index=idx, body=body, scroll=_KEEP_ALIVE_LIMIT, size = 100,)
        
        sid = response['_scroll_id']
        fetched = len(response['hits']['hits'])
        result = []
        for i in range(fetched):
            result.append(response['hits']['hits'][i])
        while(fetched>0): 
            response = self.es.scroll(scroll_id=sid, scroll=_KEEP_ALIVE_LIMIT)
            fetched = len(response['hits']['hits'])
            for i in range(fetched):
                result.append(response['hits']['hits'][i])
        return result
    
    def close(self):
        self.es.close()
    
    def refresh(self,idx):
        return self.es.indices.refresh(index=idx)
    
    #insertData
    def insertData(self, idx, id, doc):
        return self.es.index(index=idx, id=id, body=doc)
    
    #updateData
    def updateData(self, idx, id, doc):
        return self.es.update(index=idx, id=id, body={"doc" : doc })
    
    #updateAllData
    def updateAllData(self, idx, doc):
        return self.es.update_by_query(index=idx, body=doc)
    
    #deleteData
    def deleteData(self, idx, id):
        return self.es.delete(index=idx, id=id)
        
    #deleteAllData
    def deleteAllData(self, idx):
        return self.es.delete_by_query(index=idx, body={"query":{"match_all":{}}})
    
    #bulk
    def bulk(self, body):
        """예시
            body.append({
            '_index': [인덱스_이름],
            '_source': {
                "category": "test"
                "c_key": "test"
                "status": "test"
                "price": 1111
                "@timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                }
            })
        """
        return helpers.bulk(self.es, body)
    
    def question_vector_query(self, version, vector: list):
        query = {
            "from": 0,
            "size": 5,
            "query": {
                "bool": {
                    "should": [
                        {
                            "elastiknn_nearest_neighbors": {
                                "field": "question_vec",
                                "similarity": "angular",
                                "model": "lsh",
                                "candidates": 50,
                                "vec": vector
                            }
                        }
                    ]
                }
            }
        }
        if int(version) > -1 :
            filter = [
                {
                    "query_string": {
                        "query": "version:" + version
                    }
                }
            ]
            query['query']['bool']['filter'] = filter
        
        return query

    def question_index_template(self):
        return {
            "settings": {
                "similarity": {
                "pro_tfidf": {
                    "type": "scripted",
                    "script": {
                        "source": "double norm = (doc.freq); return query.boost  *norm;"
                    }
                }
                },
                "index": {
                "number_of_shards": "5",
                "elastiknn": True,
                "auto_expand_replicas": "0-1",
                "analysis": {
                    "analyzer": {
                    "whitespace_analyzer": {
                        "filter": [
                        "lowercase",
                        "trim"
                        ],
                        "tokenizer": "my_whitespace"
                    }
                    },
                    "tokenizer": {
                        "my_whitespace": {
                            "type": "whitespace",
                            "max_token_length": "60"
                            }
                        }
                    }
                },
                "index.mapping.total_fields.limit": 99999999
            },
            "mappings": {
                "properties": {
                    "id": {
                        "analyzer": "whitespace_analyzer",
                        "type": "text"
                    },
                    "version": {
                        "type": "long"
                    },
                    "siteNo": {
                        "type": "long"
                    },
                    "intentNo": {
                        "type": "long"
                    },
                    "categoryNo": {
                        "type": "long"
                    },
                    "dialogNm": {
                        "type": "text",
                        "analyzer": "pro10_kr",
                        "search_analyzer": "pro10_search"
                    },
                    "question": {
                        "type": "text",
                        "analyzer": "pro10_kr",
                        "search_analyzer": "pro10_search"
                    },
                    "question_vec": {
                        "type": "elastiknn_dense_float_vector",
                        "elastiknn": {
                            "dims": 768,
                            "model": "lsh",
                            "similarity": "angular",
                            "L": 99,
                            "k": 1
                        }
                    },
                    "term": {
                        "analyzer": "whitespace_analyzer",
                        "type": "text",
                        "similarity": "pro_tfidf"
                    },
                    "term_syn": {
                        "analyzer": "whitespace_analyzer",
                        "type": "text",
                        "similarity": "pro_tfidf"
                    },
                    "terms": {
                        "analyzer": "whitespace_analyzer",
                        "type": "text"
                    },
                    "termNo": {
                        "type": "long"
                    },
                    "keywords": {
                        "analyzer": "whitespace_analyzer",
                        "type": "text"
                    },
                    "modifyDate": {
                        "type": "date",
                        "format": "yyyyMMddHHmmssSSS"
                    },
                    "createDate": {
                        "type": "date",
                        "format": "yyyyMMddHHmmssSSS"
                    },
                    "createUser": {
                        "analyzer": "whitespace_analyzer",
                        "type": "text"
                    },
                    "createUserNm": {
                        "analyzer": "whitespace_analyzer",
                        "type": "text"
                    },
                    "modifyUser": {
                        "analyzer": "whitespace_analyzer",
                        "type": "text"
                    },
                    "modifyUserNm": {
                        "analyzer": "whitespace_analyzer",
                        "type": "text"
                    },
                    "desc": {
                        "analyzer": "whitespace_analyzer",
                        "type": "text"
                    },
                    "useYn": {
                        "type": "keyword"
                    }
                }
            }
        }

#es = elastic_util('192.168.0.5', '6251')
#print(es.countBySearch('@prochat_dic', ''))