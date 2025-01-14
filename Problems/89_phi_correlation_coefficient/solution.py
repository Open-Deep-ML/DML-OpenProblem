def phi_corr(x:list[int],y:list[int]):
    x1y1=0
    x1y0=0
    x0y1=0
    x0y0=0

    for i in range(len(x)):
        if x[i]==1:
            if y[i]==1:
                x1y1+=1
            else:
                x1y0+=1
        if x[i]==0:
            if y[i]==1:
                x0y1+=1
            else:
                x0y0+=1

    dr=((x0y0+x0y1)*(x1y0+x1y1)*(x0y0+x1y0)*(x0y1+x1y1))**0.5
    nr=(x0y0*x1y1)-(x0y1*x1y0)
    phi=round(nr/dr,4)

    return phi

def test_phi_corr():
    # Test case 1: z = 0
    assert phi_corr([1,1,0,0],[0,0,1,1]) == -1, "Test case 1 failed"
    
    # Test case 2: z = 1
    assert phi_corr([1,1,0,0],[1,0,1,1]) == -0.5774, "Test case 2 failed"

if __name__ == "__main__":
    test_phi_corr()
    print("All phi correlation coefficient tests passed.")
