import pickle as pkl
from collections import OrderedDict



l=[OrderedDict([('train', ['bcv_6','bcv_7' ,'bcv_9', 'bcv_10', 'bcv_21' ,'bcv_23' ,'bcv_24','bcv_26' ,'bcv_27' ,'bcv_31', 'bcv_33' ,'bcv_34' ,'bcv_39', 'bcv_40','bcv_5', 'bcv_28', 'bcv_30', 'bcv_37']), ('val', ['bcv_1', 'bcv_2', 'bcv_3', 'bcv_4', 'bcv_8', 'bcv_22','bcv_25', 'bcv_29', 'bcv_32', 'bcv_35', 'bcv_36', 'bcv_38'])])]
with open("splits_Task017_nnf.pkl", "wb") as f:
	pkl.dump(l,f)
