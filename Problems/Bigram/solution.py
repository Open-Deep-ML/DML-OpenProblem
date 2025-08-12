s='''<s> I am Jack </s>
   <s> Jack I am </s>
   <s> Jack I like </s>
   <s> Jack I do like </s>
   <s> do I like Jack </s>'''



def bigram_calculation(s,antecedent,anaphor):
    tokens=s.split()
    total_word_count=len(tokens)
    word_count=tokens.count(antecedent)
    bigram_count=0
    for i in range(total_word_count-1):
        if(tokens[i]==antecedent):
            if(tokens[i+1]==anaphor):
                bigram_count=bigram_count+1

    bigram_p=bigram_count/word_count
    return round(bigram_p,4)

def test():

    assert bigram_calculation(s,"Jack","I")==0.6,"Test Case 1 Failed !"

    assert bigram_calculation(s,"I","like")==0.4,"Test Case 2 Failed !"


if __name__ == "__main__":
    test()
    print("All Unigram tests passed.")
