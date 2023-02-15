from pytorch_lightning.cli import LightningCLI
import sys
import numpy as np

# Modules
from module.VIT import VIT

# Data_Modules
from dataModule.MNIST import MNIST

# Callbacks
#from callback.LogTestImageCallback import LogTestImageCallback

def cli_main():
  cli = LightningCLI(datamodule_class=MNIST)

def split_list(input_list, value):
  index = input_list.index(value)
  return input_list[:index], input_list[index+1:]

def execute_np_random(string):
  return eval("np.random."+string)

if __name__=="__main__":
  # multiple execution with the given random func.
  if "multiple" in sys.argv:
    args, [mult_arg_name, mult_arg_executable] = split_list(sys.argv, "multiple")
    random_args = execute_np_random(mult_arg_executable)
    for random_arg in random_args:
      sys.argv = args + [mult_arg_name] + [str(random_arg)]
      cli_main()
  
  # just execute once.
  else:
    cli_main()
  