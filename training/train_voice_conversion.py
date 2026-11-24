"""Binary detection of human-source voice conversion under the ASVspoof LA protocol."""
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.train_audio_official import load_part,check_split
from training.train_public import metrics,save_report

# Wang et al., ASVspoof 2019 database, Table 1. A13-15 use synthesized inputs.
CONVERTED={"A05","A06","A17","A18","A19"}
HYBRID={"A13","A14","A15"}


def family_part(name):
    x,_,rows=load_part(name)
    keep=np.array([r["attack"] not in HYBRID for r in rows])
    return x[keep],np.array([r["attack"] in CONVERTED for r in rows])[keep],rows


if __name__=="__main__":
    with threadpool_limits(limits=3):
        train,y,train_rows=family_part("train")
        classifier=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,class_weight="balanced",random_state=42).fit(train,y)
        dev,vy,dev_rows=family_part("val");check_split(dev_rows,train_rows)
        cal=LogisticRegression().fit(classifier.decision_function(dev).reshape(-1,1),vy)
        test,ty,te_rows=family_part("test");check_split(te_rows,train_rows+dev_rows)
        probs=cal.predict_proba(classifier.decision_function(test).reshape(-1,1))[:,1]
        folder=save_report("voice_conversion",{"dataset":"ASVspoof 2019 LA voice conversion subset","architecture":"Spectral band statistics and gradient boosting",
            "counts":{"train":len(y),"validation":len(vy),"test":len(ty)},"test":metrics(ty,probs),
            "evaluation_scope":"official partitions; hybrid systems excluded from evaluation",
            "source":"https://www.eurecom.eu/publication/6352/download/sec-publi-6352.pdf",
            "positive_attacks":sorted(CONVERTED),"excluded_hybrid_attacks":sorted(HYBRID),
            "limitations":"Experimental human-source voice conversion detection against bona fide or text-to-speech. Does not identify the speaker, the target of impersonation or a particular cloning system."})
        joblib.dump({"classifier":classifier,"calibration":cal},folder/"model.joblib",compress=3)
