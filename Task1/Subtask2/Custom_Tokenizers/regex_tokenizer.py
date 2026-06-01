
"""
 A Custom Regex based Tokenizer.  Simple, not optimized in either memory or time.
 Based off of Section 2.2 and 2.3 in the book, but doesn't exactly copy paste it.
 I took the Regex based splitting logic froms section 2.2 of the book and implemented it in the split() function
 Converting to token_ids and making a class to encode and decode inputs was taken from Section 2.3

 This code serves as a submission for both assignment 1 and 2

 (I thought it's unproductive to just write the code from the Book as-is. 
 So I will write my own interpretations of the underlying logic, to truly test my understanding)

 Apologies if this unintentionally causes an inconvenience
"""

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