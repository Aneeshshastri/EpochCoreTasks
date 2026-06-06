
import re
class RegexTokenizer:
    def __init__(self,punctuation=None,retain_whitespace=False):
        self.punctuation=punctuation
        self.retain_whitespace=retain_whitespace
        self.vocab=None
        self.token_map=dict()
        self.inverse_token_map=dict()
    def split (self,text):
        if(self.punctuation):
          escaped_punc = "".join(re.escape(p) for p in set(self.punctuation))
          pattern = rf"(\s+|[{escaped_punc}]+)" if self.retain_whitespace else rf"\s+|([{escaped_punc}]+)" 
        else:
            pattern = rf"(\s+|[^\w\s]+)" if self.retain_whitespace else rf"\s+|([^\w\s]+)" 
        return [item for item in re.split(pattern, text) if item]
    def train(self,text):
        self.vocab=set(self.split(text))
        token_id=0
        for token in self.vocab:
            self.token_map[token]=token_id
            self.inverse_token_map[token_id]=token
            token_id+=1
    def encode(self,text):
        return [self.token_map[token] for token in self.split(text)]
    def decode(self,token_list):
        if(self.retain_whitespace):
            return ''.join([self.inverse_token_map[token_id] for token_id in token_list])
        else:
            return ' '.join([self.inverse_token_map[token_id] for token_id in token_list])

    

if __name__=="__main__":
    sample_text="Hey, it's me! Goku. I heard you were strong, How about a sparring match?"
    Tokenizer=RegexTokenizer(retain_whitespace=True)
    Tokenizer.train(sample_text)
    print(Tokenizer.vocab)
    print(Tokenizer.token_map)
    print(Tokenizer.encode(sample_text))
    print(Tokenizer.decode(Tokenizer.encode(sample_text)))