import re
import matplotlib.pyplot as plt
import numpy as np


logfile = "logs/dec22.log"

# Batch loss on step 112799: 5.48	Batch accuracy@3: 0.20	ms/batch: 1042.53
# Validation loss: 5.96	Validation accuracy@3 0.15   
# pattern = r"^Batch loss on step \d+: ([\d\.]+).*Batch accuracy@3: ([\d\.]+)	ms/batch: .*$"
pattern = r"^Validation loss: (\d.\d\d).*Validation accuracy\@3 (\d.\d\d).*$"

# x = 1
loss, acc = [], []
loss_avg, acc_avg = [], []
with open(logfile, encoding="utf-8") as file_in:
    for line in file_in:
        # if line.startswith("Batch"):
        if line.startswith("Validation"):
            print(line)
            mo = re.match(pattern, line)
            loss.append(float(mo.group(1)))
            acc.append(float(mo.group(2)))
        # if line.startswith("Step"):
        # if len(loss) == 5000:
            # loss = np.array(loss) 
            # acc = np.array(acc) 
            # loss_avg.append(loss.mean())
            # acc_avg.append(acc.mean())
            # loss, acc = [], []
        

loss = np.array(loss) 
acc = np.array(acc) 
loss_avg.append(loss.mean())
acc_avg.append(acc.mean())

# print(loss_avg)
# print(acc_avg)
print(loss)

# plt.plot(acc_avg)
plt.plot(acc)
plt.title("Validation accuracy")
# plt.show()
plt.savefig("logs/dec22_val_acc_t.png")

