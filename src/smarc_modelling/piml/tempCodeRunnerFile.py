   for i in range(6):
        for j in range(i+1):
            v = speed[j]
            D_chol_pred[i,j] = L[i,j] + Q[i,j]*v
