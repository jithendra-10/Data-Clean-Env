"""
DataClean-Env — Task Registry  v2.0
Business rule graders added (Scaler bonus + Theme 3.1 enterprise requirement).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd


@dataclass
class Task:
    task_id: str
    description: str
    max_steps: int
    generate: Callable
    grade: Callable
    canonical_column_names: dict = field(default_factory=dict)
    irrelevant_columns: list     = field(default_factory=list)
    business_rules: dict         = field(default_factory=dict)


# ── Grader helpers ────────────────────────────────────────────────────────────

def _null_score(df, cols, t=0.01):
    s = [1.0 if df[c].isna().mean()<=t else max(0,1-df[c].isna().mean()*5)
         for c in cols if c in df.columns]
    return float(np.mean(s)) if s else 0.0

def _dtype_score(df, expected):
    s = []
    for col, kind in expected.items():
        if col not in df.columns: s.append(0.0); continue
        a = df[col].dtype
        if kind=="numeric": s.append(1.0 if pd.api.types.is_numeric_dtype(a) else 0.0)
        elif kind=="string": s.append(1.0 if a==object else 0.5)
        else: s.append(1.0 if str(a)==kind else 0.0)
    return float(np.mean(s)) if s else 1.0

def _outlier_score(df, cols, max_frac=0.05):
    s = []
    for c in cols:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]): continue
        clean = df[c].dropna()
        if len(clean)<4: continue
        q1,q3 = clean.quantile(0.25), clean.quantile(0.75)
        frac = ((clean<q1-1.5*(q3-q1))|(clean>q3+1.5*(q3-q1))).mean()
        s.append(1.0 if frac<=max_frac else max(0,1-frac*10))
    return float(np.mean(s)) if s else 1.0

def _dup_score(df):
    r = df.duplicated().mean()
    return 1.0 if r<0.005 else max(0.0,1-r*10)

def _business_rule_score(df, rules):
    """Enterprise compliance: fraction of rows satisfying domain constraints."""
    if not rules: return 1.0
    scores = []
    for col, rule in rules.items():
        if col not in df.columns: continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series)==0: continue
        parts = dict(p.split(":") for p in rule.split(","))
        lo = float(parts.get("min", "-inf"))
        hi = float(parts.get("max", "inf"))
        scores.append(((series>=lo)&(series<=hi)).mean())
    return float(np.mean(scores)) if scores else 1.0


# ── TASK 1 — Employee  (Easy) ────────────────────────────────────────────────

def _gen1(rng):
    n=300
    ages=rng.integers(22,65,size=n).astype(float)
    salaries=rng.normal(55_000,12_000,n)
    depts=rng.choice(["Engineering","Marketing","Sales","HR"],size=n)
    tenure=rng.integers(0,20,size=n).astype(float)
    ages[rng.choice(n,size=int(n*.22),replace=False)]=np.nan
    salaries[rng.choice(n,size=12,replace=False)]=rng.choice([-99_999,500_000,999_000],size=12)
    tenure_s=[str(int(v)) if not np.isnan(v) else "N/A" for v in tenure]
    base=pd.DataFrame({"age":ages,"salary":salaries,"department":depts,"years_at_company":tenure_s})
    dupes=base.sample(n=30,random_state=int(rng.integers(0,9999)))
    df=pd.concat([base,dupes],ignore_index=True)
    return df.sample(frac=1,random_state=int(rng.integers(0,9999))).reset_index(drop=True)

def _grade1(df):
    ns  = _null_score(df,["age","salary","years_at_company"])
    ts  = _dtype_score(df,{"age":"numeric","salary":"numeric","years_at_company":"numeric"})
    os_ = _outlier_score(df,["salary"])
    ds  = _dup_score(df)
    br  = _business_rule_score(df,{"salary":"min:0,max:500000","age":"min:18,max:100"})
    return round(0.25*ns+0.20*ts+0.20*os_+0.15*ds+0.20*br, 4)

TASK_1=Task("task_1",
    "Employee dataset (300 rows). 22% null ages, salary outliers, tenure as string, "
    "30 dupes. Business rules: salary 0-500k, age 18-100.",
    max_steps=15, generate=_gen1, grade=_grade1,
    canonical_column_names={"years_at_company":"years_at_company"},
    business_rules={"salary":"min:0,max:500000","age":"min:18,max:100"})


# ── TASK 2 — E-Commerce  (Medium) ────────────────────────────────────────────

def _gen2(rng):
    n=500
    amounts=rng.lognormal(mean=4.5,sigma=0.8,size=n)
    quantities=rng.integers(1,20,size=n).astype(float)
    ratings=rng.choice([1.,2.,3.,4.,5.],size=n)
    cats=rng.choice(["Electronics","Clothing","Food","Books","Sports"],size=n)
    ts=pd.date_range("2023-01-01",periods=n,freq="2h").astype(str).tolist()
    tracking=[f"HASH-{rng.integers(0,99999):05d}" for _ in range(n)]
    amounts[rng.choice(n,size=25,replace=False)]=rng.choice([-500,50_000,999_999],size=25)
    quantities[rng.choice(n,size=int(n*.28),replace=False)]=np.nan
    ratings=ratings.astype(float); ratings[rng.choice(n,size=int(n*.15),replace=False)]=np.nan
    df=pd.DataFrame({"order_id":[f"ORD-{i:05d}" for i in range(n)],
                     "order_amount":amounts,"qty":quantities,"customer_rating":ratings,
                     "category":cats,"created_at":ts,"internal_hash":tracking})
    dupes=df.sample(n=40,random_state=int(rng.integers(0,9999)))
    df=pd.concat([df,dupes],ignore_index=True)
    return df.sample(frac=1,random_state=int(rng.integers(0,9999))).reset_index(drop=True)

def _grade2(df):
    ns  = _null_score(df,["order_amount","qty","customer_rating"])
    ts  = _dtype_score(df,{"order_amount":"numeric","qty":"numeric","customer_rating":"numeric"})
    os_ = _outlier_score(df,["order_amount"])
    ds  = _dup_score(df)
    drop= 0.0 if "internal_hash" in df.columns else 0.05
    br  = _business_rule_score(df,{"order_amount":"min:0,max:100000","customer_rating":"min:1,max:5"})
    return round(min(0.20*ns+0.20*ts+0.20*os_+0.15*ds+0.20*br+drop,1.0),4)

TASK_2=Task("task_2",
    "E-commerce orders (500 rows). 28% null qty, 15% null ratings, amount outliers, "
    "40 dupes, internal_hash irrelevant. Business rules: amount 0-100k, rating 1-5.",
    max_steps=18, generate=_gen2, grade=_grade2,
    canonical_column_names={"qty":"quantity"},
    irrelevant_columns=["internal_hash"],
    business_rules={"order_amount":"min:0,max:100000","customer_rating":"min:1,max:5"})


# ── TASK 3 — Healthcare  (Hard) ───────────────────────────────────────────────

def _gen3(rng):
    n=800
    ages=rng.integers(18,90,size=n).astype(float)
    genders=rng.choice(["M","F","Other"],size=n,p=[0.48,0.48,0.04])
    weights=rng.normal(75,15,n); heights=rng.normal(170,10,n)
    systolic=rng.normal(120,18,n); glucose=rng.normal(100,25,n)
    cholest=rng.normal(190,35,n)
    diag=rng.choice(["Hypertension","Diabetes","Healthy","Obesity","Hyperlipidemia"],size=n)
    admit=pd.date_range("2020-01-01",periods=n,freq="6h").astype(str).tolist()
    admin=[f"note_{rng.integers(0,999)}" for _ in range(n)]
    ages[rng.choice(n,size=int(n*.25),replace=False)]=np.nan
    weights[rng.choice(n,size=30,replace=False)]=rng.choice([-10,5,500,999],size=30)
    glucose=glucose.tolist()
    for i in rng.choice(n,size=int(n*.18),replace=False):
        glucose[i]=rng.choice(["N/A","pending","---","HIGH","??"])
    glucose=pd.array(glucose,dtype=object)
    cholest=cholest.astype(float)
    cholest[rng.choice(n,size=int(n*.12),replace=False)]=np.nan
    cholest[rng.choice(n,size=15,replace=False)]=rng.choice([-50,1500,2000],size=15)
    systolic=systolic.astype(float)
    systolic[rng.choice(n,size=int(n*.08),replace=False)]=np.nan
    df=pd.DataFrame({"patient_age":ages,"gender":genders,"weight_kg":weights,"height_cm":heights,
                     "systolic_bp":systolic,"glucose_mgdl":glucose,"cholesterol":cholest,
                     "diagnosis":diag,"admission_date":admit,"admin_notes":admin})
    dupes=df.sample(n=60,random_state=int(rng.integers(0,9999)))
    df=pd.concat([df,dupes],ignore_index=True)
    return df.sample(frac=1,random_state=int(rng.integers(0,9999))).reset_index(drop=True)

def _grade3(df):
    ns  = _null_score(df,["patient_age","systolic_bp","cholesterol"])
    ts  = _dtype_score(df,{"patient_age":"numeric","weight_kg":"numeric",
                            "glucose_mgdl":"numeric","cholesterol":"numeric","systolic_bp":"numeric"})
    os_ = _outlier_score(df,["weight_kg","cholesterol"])
    ds  = _dup_score(df)
    glc = float(pd.to_numeric(df["glucose_mgdl"],errors="coerce").notna().mean()) if "glucose_mgdl" in df.columns else 0.0
    drop= 0.0 if "admin_notes" in df.columns else 0.05
    br  = _business_rule_score(df,{"patient_age":"min:18,max:100","weight_kg":"min:20,max:300",
                                    "cholesterol":"min:50,max:600","systolic_bp":"min:60,max:250"})
    return round(min(0.20*ns+0.15*ts+0.15*os_+0.10*ds+0.10*glc+0.20*br+drop,1.0),4)

TASK_3=Task("task_3",
    "Healthcare records (800 rows). Mixed per-column corruption. Business rules: "
    "physiological bounds on age/weight/cholesterol/systolic.",
    max_steps=20, generate=_gen3, grade=_grade3,
    irrelevant_columns=["admin_notes"],
    business_rules={"patient_age":"min:18,max:100","weight_kg":"min:20,max:300",
                    "cholesterol":"min:50,max:600","systolic_bp":"min:60,max:250"})


TASK_REGISTRY: dict[str, Task] = {
    "task_1":TASK_1, "task_2":TASK_2, "task_3":TASK_3
}
