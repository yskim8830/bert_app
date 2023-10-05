
class const():
    def __init__(self):
        
        self._train_state = '$ctrain_state'
        self._dev_idx = '$cdev_'
        self._svc_idx = '$csvc_'
        self._als_idx = '$cals_'
        self._intent = 'intent_'
        self._question = 'question_'
        self._model = 'model_'
    
    @property    
    def train_state(self):
        return self._train_state    
    @property    
    def dev_idx(self):
        return self._dev_idx
    @property    
    def svc_idx(self):
        return self._svc_idx
    @property    
    def als_idx(self):
        return self._als_idx
    @property    
    def intent(self):
        return self._intent
    @property    
    def question(self):
        return self._question
    @property    
    def model(self):
        return self._model