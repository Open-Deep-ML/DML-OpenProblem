def min_max(x:list[int])->list[float]:

    #trying to modify the input list in-place instead creating a new list

    largest=max(x) #finding the largest value in the input array
    smallest=min(x) #finding the minimum value in the input array

    if largest==smallest: # if input has identical elements
        return[0.0]*len(x)
    for i in range(len(x)):
        #using the formula to normalize the input
        x[i]=round((x[i]-smallest)/(largest-smallest),4)

    return(x)

def test_min_max():
    #first_test_case:

    assert min_max([1,2,3,4,5])==[0.0, 0.25, 0.5, 0.75, 1.0],"Test Case 1 failed"

    assert min_max([30,45,56,70,88])==[0.0, 0.2586, 0.4483, 0.6897, 1.0],"Test Case 2 failed"

    assert min_max([5,5,5,5])==[0.0,0.0,0.0,0.0], "Test Case 3 failed"


if __name__=="__main__":
    test_min_max()
    print("All Min Max Normalization test cases passed.")

