
import matplotlib.pyplot as plt
import numpy as np
from Second_model import *






if __name__=="__main__":
    answer = np.argmax(second_model.predict(np.array(predicted)),axis=-1)
    print(test_prediction)
    print(answer)



