import numpy as np 


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
   # stick to lowercase
   a = A

   a_t = np.transpose(a)
   a_2 = a_t @ a

   v = np.eye(2)

   for _ in range(1):
       # Compute rotation angle for a 2x2 matrix
       if a_2[0,0] == a_2[1,1]:
           theta = np.pi/4
       else:
           theta = 0.5 * np.arctan2(2 * a_2[0,1], a_2[0,0] - a_2[1,1])
       
       # Create rotation matrix
       r = np.array(
           [
               [np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]
               ]
           )
       
       # apply rotation
       d = np.transpose(r) @ a_2 @ r

       # update a_2
       a_2 = d

       # accumulate v
       v = v @ r

   # sigma is the diagonal elements squared
   s = np.sqrt([d[0,0], d[1,1]])
   s_inv = np.array([[1/s[0], 0], [0, 1/s[1]]])
   
   u = a @ v @ s_inv
   
   return (u, s, v.T)
