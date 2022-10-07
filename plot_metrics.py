import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('./checkpoint/HEV_no_img_v1/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

ax = plt.subplot(111)

train_loss = np.array(metrics['train_loss'])
val_loss = np.array(metrics['val_loss'])
ax.plot(train_loss, label='train loss')
ax.plot(val_loss, label='val loss')
ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

plt.show()
