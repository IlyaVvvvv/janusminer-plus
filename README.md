# Janusminer Plus 
# ¯\\_(ツ)\_/¯
Based on janusminer 0.2.7 (by CoinFuMasterShifu),
99% code is the same.  
Tested with different combinations 1-3 Nvidia GPUs paired with 8 and 16 core ryzen CPUs on Linux Ubuntu 22.04.3 LTS pool and solo - works well.
##  What have changed?
Filtering logic and  fixed gpu batch size
## 💻 System Requirements
## 😵‍💫 BUILD INSTRUCTIONS

## ▶️ USAGE
All same as janusminer
## 🛠 DIFFERENCES
#### Filtering:
By default autofiltering is enabled, works accurate enough.

If you know what are hashrate limits of your hardware, you can calculate optimal filtering upper bound by yourself and set fixed value by parameter -f.
Value as power of 2 without minus (example: for sha256t_float hash 0.012 it will be -f 6.380821784)  
You can use this for more details and calculations: https://www.desmos.com/Calculator/lcjylqepyv
#### Gpubatchsize:
Somehow fixed gpu batch size gives more stable load on gpus.  
By default 20000000. Tried this value with different combinations for 1-3 Nvidia GPUs (rtx 2070, rtx 3070ti and rtx 3090),  
achieved stable 100% gpu load in every configuration.
Increasing further didn't give me anything in terms of overall performance.   
If you want you can change it with parameter -g (example: -g 25000000).
## ☝️ Important: pcie bandwidth and gpu performance
As gpus constantly send data to cpu, pcie bandwidth matters a lot, it is much better to use full x16 pcie slots.

Pcie x1 gen2 risers for most gpus are not enough, as every cpu thread is giving some load to pcie lane.   
But i've found that if at least 1 gpu runs in full pcie x16, the second one in pcie x1 will give much higher performance  
(but stil not as much as in full pcie x16).
