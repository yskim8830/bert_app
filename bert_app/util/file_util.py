import os
import logging
import subprocess
from pprint import pprint as pp
from jamo import h2j, j2hcj

logger = logging.getLogger('my')
path = 'D:\proten\prochat\elasticsearch\config\prosearch'
#filename = 'test.txt'
class dicFile:
    def __init__(self, path):
        self.path = path
        self.filename = ''
        self.phrase = ''
    
    def stopword(self):
        self.filename = 'chat_stopword.txt'
        self.phrase = "# Stopwords Configuration - pro10-chat for Elasticsearch\n" + "# Lines starting with '#' and empty lines are ignored."
        
    def synonym(self):
        self.filename = 'chat_sysnonym.txt'
        self.phrase = "# Synonym Configuration - pro10-chat for Elasticsearch\n" + "# Lines starting with '#' and empty lines are ignored."
    
    def compound(self):
        self.filename = 'chat_user_define.txt'
        self.phrase = "# Compound-Noun Composition Configuration - pro10-chat for Elasticsearch\n" + "# Lines starting with '#' and empty lines are ignored."
        
    def create_dic_file(self,dic,str=''):
        if dic == 'stopword':
            self.stopword()
        elif dic == 'synonym':
            self.synonym()
        elif dic == 'compound':
            self.compound()
        else:
            return 'not found dic type'
        with open(os.path.join(self.path,self.filename),'w') as f:
            f.write(self.phrase)
            f.write(str)
            
class dic:
    def __init__(self, path):
        self.path = path
    def stopword(self):
        self.filename = 'chat_stopword.txt'
        site = ''
        dic_map = {}
        setMap = set()
        with open(os.path.join(self.path,self.filename),'r') as f:
            for line in f:
                line = line.strip()
                if line[0] != '#':
                    data = line.split('\t')
                    if(len(data) > 1):
                        if(site == ''):
                            site = data[0]
                        elif site != data[0]:
                            dic_map[site] = setMap
                            setMap = set()
                            site = data[0]
                        setMap.add(data[1].strip())
                if len(setMap) > 0:
                    dic_map[site] = setMap
        return dic_map
    def synonym(self):
        self.filename = 'chat_sysnonym.txt'
        site = ''
        dic_map = {}
        setMap = {}
        with open(os.path.join(self.path,self.filename),'r') as f:
            for line in f:
                line = line.strip()
                if line[0] != '#':
                    data = line.split('\t')
                    if(len(data) > 1):
                        if(site == ''):
                            site = data[0]
                        elif site != data[0]:
                            dic_map[site] = setMap
                            setMap = {}
                            site = data[0]
                        for synm in data[2].split(','):
                            setMap[synm] = data[1].strip()
                if len(setMap) > 0:
                    dic_map[site] = setMap
        return dic_map
    def compound(self):
        self.filename = 'chat_user_define.txt'
        site = ''
        dic_map = {}
        setMap = {}
        with open(os.path.join(self.path,self.filename),'r') as f:
            for line in f:
                line = line.strip()
                if line[0] != '#':
                    data = line.split('\t')
                    if(len(data) > 1):
                        if(site == ''):
                            site = data[0]
                        elif site != data[0]:
                            dic_map[site] = setMap
                            setMap = {}
                            site = data[0]
                            
                        chk_key = data[1][0] #1개음절을 setmap 키로 담음
                        dic_data = []
                        dic_data.append(data[1])
                        
                        if len(data) > 2:
                            dic_data.append(data[2])
                        else:
                            dic_data.append(data[1])
                            
                        value_data = []
                        for setMap_key in setMap:
                            if setMap_key == chk_key:
                                value_data = setMap[chk_key]
                        value_data.append(dic_data)
                        setMap[chk_key] = value_data
                        
                if len(setMap) > 0:
                    dic_map[site] = setMap
        return dic_map
    
def export_user_dic(mecab_dic_path, user_dic):
    try:
        with open(os.path.join(mecab_dic_path,'user-dic','nnp.csv'), 'w' , encoding='utf-8') as f:
            for words in user_dic.values():
                if len(words) >0:
                    for wordset in words:
                        if len(wordset) > 1:
                            for word in wordset[1].split(','):
                                jongsung_TF = get_jongsung_TF(word)
                                line = '{},,,,NNP,*,{},{},*,*,*,*,*\n'.format(word, jongsung_TF, word)
                                f.write(line)
    except Exception as e:
        logger.error(e)
        return False
    return True

def get_jongsung_TF(sample_text):
    sample_text_list = list(sample_text)
    last_word = sample_text_list[-1]
    last_word_jamo_list = list(j2hcj(h2j(last_word)))
    last_jamo = last_word_jamo_list[-1]

    jongsung_TF = "T"

    if last_jamo in ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ', 'ㅘ', 'ㅚ', 'ㅙ', 'ㅝ', 'ㅞ', 'ㅢ', 'ㅐ,ㅔ', 'ㅟ', 'ㅖ', 'ㅒ']:
        jongsung_TF = "F"

    return jongsung_TF


#윈도우 파워쉘 실행
def run_power_shell(mecab_dic_path):
    try:   
        args=[r"powershell",os.path.join(mecab_dic_path,'tools','add-userdic-win.ps1')] # windows 일시
        p=subprocess.Popen(args, stdout=subprocess.PIPE)
        dt=p.stdout.read()
        return True
    except Exception as e:
        print(e)
        return False
    
#리눅스 배치쉘 실행
def run_lnx_shell(mecab_dic_path):
    try:
        os.system(os.path.join(mecab_dic_path,'tools','add-userdic.sh'))
        os.system(os.path.join(mecab_dic_path,'make') + ' install')
    except Exception as e:
        print(e)
        return False

#검색결과 파일저장
def save_question_file(path, filename, dataset):
    with open(os.path.join(path,filename+'.csv'),'w' , encoding='utf-8') as f:
        f.write('question,intent' + '\n')
        for data in dataset:
            data = data['_source']
            f.write(str(data['question']).replace(',',' ') + ',' + data['dialogNm'] + '\n')
            
def save_question_file2(path, filename, dataset):
    with open(os.path.join(path,filename+'.csv'),'w' , encoding='utf-8') as f:
        f.write('questionNo,question,dialogNo,dialogNm' + '\n')
        for data in dataset:
            data = data['_source']
            f.write(str(data['questionNo'])+ ',' + str(data['question']).replace(',',' ') + ',' + str(data['dialogNo'])+ ',' +data['dialogNm'] + '\n')
            
#버전정보 파일저장
def save_status_file(path, site_no, version):
    with open(os.path.join(path,'site_no_'+str(site_no)+'.properties'),'w' , encoding='utf-8') as f:
        f.write(str(version))

#버전정보 파일읽기
def load_status_file(path, site_no):
    with open(os.path.join(path,'site_no_'+str(site_no)+'.properties'),'r' , encoding='utf-8') as f:
        return int(f.readline())

#버전정보 파일삭제
def del_status_file(path, site_no):
    if os.path.exists(os.path.join(path,'site_no_'+str(site_no)+'.properties')):
        os.remove(os.path.join(path,'site_no_'+str(site_no)+'.properties'))