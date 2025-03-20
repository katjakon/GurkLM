import argparse
import re
import os

import matplotlib.pyplot as plt
import numpy as np


# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument("log_file", type=str, help="Path to the log file containing the log output of the model")
parser.add_argument("config_file", type=str, help="Path to the config file used for the training run")
parser.add_argument("out_dir", type=str, help="Directory to store the output files at")
parser.add_argument("--plot-steps", type=int, default=200, help="Number of steps to average over for plotting the training data learning curve")

args = parser.parse_args()

logfile = args.log_file
out_dir = args.out_dir
config_name = os.path.split(args.config_file)[1].split(".")[0]

pattern_training = r"^Batch loss on step \d+: ([\d\.]+).*Batch accuracy@3: ([\d\.]+)	ms/batch: .*$"
pattern_validation = r"^Validation loss: (\d.\d\d).*Validation accuracy\@3 (\d.\d\d).*$"

# Set up lists to collect values
val_loss, val_acc = [], []
train_loss, train_acc = [], []
train_loss_avg, train_acc_avg = [], []
# Collect values from log file
with open(logfile, encoding="utf-8") as file_in:
    for line in file_in:
        # Validation results line
        if line.startswith("Validation"):
            mo = re.match(pattern_validation, line)
            val_loss.append(float(mo.group(1)))
            val_acc.append(float(mo.group(2)))
        # Training batch result
        if line.startswith("Batch"):
            mo = re.match(pattern_training, line)
            train_loss.append(float(mo.group(1)))
            train_acc.append(float(mo.group(2)))
        # Average over steps
        if len(train_loss) == args.plot_steps:
            train_loss = np.array(train_loss) 
            train_acc = np.array(train_acc) 
            train_loss_avg.append(train_loss.mean())
            train_acc_avg.append(train_acc.mean())
            train_loss, train_acc = [], []
        
# Include final steps as well
train_loss = np.array(train_loss) 
train_acc = np.array(train_acc) 
train_loss_avg.append(train_loss.mean())
train_acc_avg.append(train_acc.mean())

# Plot training accuracy
plt.figure()
plt.plot(train_acc_avg)
plt.title(f"Train accuracy each {args.plot_steps} steps")
filename = f"{config_name}_training_accuracy_curve.png"
plt.savefig(os.path.join(out_dir, filename))

# Plot training loss
plt.figure()
plt.plot(train_loss_avg)
plt.title(f"Train loss averaged each {args.plot_steps} steps")
filename = f"{config_name}_training_loss_curve.png"
plt.savefig(os.path.join(out_dir, filename))

# Plot validation accuracy
plt.figure()
plt.plot(val_acc)
plt.title(f"Validation accuracy each {args.plot_steps} steps")
filename = f"{config_name}_validation_accuracy_curve.png"
plt.savefig(os.path.join(out_dir, filename))

# Plot training loss
plt.figure()
plt.plot(val_loss)
plt.title(f"Validation loss averaged each {args.plot_steps} steps")
filename = f"{config_name}_validation_loss_curve.png"
plt.savefig(os.path.join(out_dir, filename))

print(f"Plots saved under {out_dir}")

