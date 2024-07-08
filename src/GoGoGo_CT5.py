import os
import threading

# Set length
# src len, tgt len, 
task_length = { 
    "ACL": [512,100],
    "AAAI": [512,100]
}

def run(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc=5, LR=5e-5):
    os.system(f'bash ./run.sh "{port}" "{gpuid}" "{model}" "{model_tag}" "{task}" "{maxSL}" "{maxTL}" "{dataDir}" "{BSize}" "{GAcc}" "{LR}"')

BSize=20
GAcc=1
LR=5e-5
port =8756
model = '/Salesforce/codet5-base'

# **************** ACL ****************
task='ACL'
maxSL, maxTL = task_length[task]
dataDir = '../dataset/ACL'
model_tag = f'CodeT5'+f'-{str(BSize)}-{str(LR)}-{str(GAcc)}'

gpuid=0
port = port+1
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()

IQR_TAG = "_05IQR"
task = f'ACL_static{IQR_TAG}'
gpuid=1
port = port+1
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()

task = f'ACL_dynamic'
gpuid=2
port = port+1
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()

# **************** AAAI ****************
task='AAAI'
maxSL, maxTL = task_length[task]
dataDir = '../dataset/AAAI'
model_tag = f'CodeT5'+f'-{str(BSize)}-{str(LR)}-{str(GAcc)}'

task='AAAI_test150'
gpuid=2
port = port+1
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()

IQR_TAG = "_05IQR"
task=f'AAAI_static_test150{IQR_TAG}'
gpuid=2
port = port+1
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()

task=f'AAAI_dynamic_test150'
gpuid=2
port = port+1
t = threading.Thread(target=run, args=(port, gpuid, model, model_tag, task, maxSL, maxTL, dataDir, BSize, GAcc, LR))
t.start()
