s='''<s> I am Jack </s>
   <s> Jack I am </s>
   <s> Jack I like </s>
   <s> Jack I do like </s>
   <s> do I like Jack </s>'''

def unigram_calculation(s,word):
    tokens=s.split()
    total_word_count=len(tokens)
    word_count=tokens.count(word)
    unigram_p=word_count/total_word_count
    return round(unigram_p,4)

def test():

    #test case 1 for the word Jack
    assert unigram_calculation(s,"Jack")==0.1852,"Test Case 1 Failed !"

    #test case 2 for the word like
    assert unigram_calculation(s,"like")==0.1111,"Test Case 2 Failed !"


if __name__ == "__main__":
    test()
    print("All Unigram tests passed.")

