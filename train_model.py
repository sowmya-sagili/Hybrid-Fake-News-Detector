"""
train_model.py  –  Trains and saves the Hybrid Fake News Detection model.
Run once:  python train_model.py
Produces:  models/model.joblib, models/tfidf.joblib, models/config.json
"""

import os
import sys
import json
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── ensure models/ exists ────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

# ── imports ──────────────────────────────────────────────────────────────────
try:
    import numpy as np
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    )
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score
    )
    log.info("All ML packages imported successfully.")
except ImportError as e:
    log.error(f"Import failed: {e}")
    log.error("Run:  .venv/bin/pip install scikit-learn joblib numpy")
    sys.exit(1)

# ── synthetic training data ───────────────────────────────────────────────────
REAL_TEMPLATES = [
    "Scientists at {uni} published a peer-reviewed study showing {finding}. "
    "The research, funded by {org}, involved {n} participants and was reviewed by "
    "independent experts. According to lead researcher Dr. {name}, the results "
    "suggest {conclusion}. Data is available in the supplementary materials.",

    "The {govt} announced new {policy} legislation on {day}. According to official "
    "press releases, the policy aims to {aim}. Experts from {uni} and {think_tank} "
    "reviewed the proposal. Critics and supporters have expressed mixed reactions.",

    "Reuters reports that {company} posted quarterly earnings of ${amount} billion, "
    "beating analyst expectations by {pct}%. CEO {name} cited {reason} as the primary "
    "driver of growth. The stock rose {rise}% in after-hours trading on the news.",

    "Health officials from the {agency} confirmed {n} new cases of {disease} in "
    "{region}. Officials recommend standard precautions including hand-washing and "
    "staying home when ill. Vaccines remain the most effective prevention measure "
    "according to peer-reviewed literature.",

    "A new report by {think_tank} analyzes the economic impact of {policy} on "
    "middle-income households. Using data from {n} counties, the study finds that "
    "{finding}. The methodology and raw data are publicly available for review.",
]

FAKE_TEMPLATES = [
    "DOCTORS DON'T WANT YOU TO KNOW THIS!!! {miracle} cures {disease} in just "
    "{days} days! Big Pharma is hiding this ancient remedy because it would "
    "destroy their profits!!! Share this before it gets DELETED!!!",

    "BREAKING: Deep state operatives have been caught {action}! The mainstream "
    "media won't report this BOMBSHELL story! Wake up sheeple — the government "
    "is lying to you! Share NOW before this gets censored!!!",

    "This one WEIRD TRICK discovered by a {city} mom ELIMINATES {problem} overnight! "
    "Doctors HATE her! The medical establishment doesn't want you to know this "
    "simple secret. Click here before they take this down!!!",

    "EXPOSED: {celebrity} secretly {action} and the elite are covering it up! "
    "The shadow government doesn't want you to see this shocking revelation! "
    "Share before deleted! They are coming for us all!!!",

    "MIRACLE: Scientists REFUSE to explain how {product} DESTROYS {disease} in "
    "48 hours! Ancient remedy SUPPRESSED by Big Pharma for decades! "
    "This unbelievable discovery will change everything! SHARE IMMEDIATELY!!!",
]

FILL = {
    "uni": ["Harvard University", "MIT", "Oxford University", "Stanford University", "Johns Hopkins"],
    "finding": ["significant correlation", "reduced risk by 30%", "improved outcomes", "no causal link"],
    "org": ["the NIH", "NSF", "WHO", "the European Research Council"],
    "n": ["1,200", "5,000", "320", "12,000", "800"],
    "name": ["Jane Smith", "John Lee", "Maria Garcia", "David Kim", "Sarah Chen"],
    "conclusion": ["further research is warranted", "the hypothesis is supported", "results are preliminary"],
    "govt": ["U.S. Congress", "UK Parliament", "Canadian government", "European Commission"],
    "policy": ["healthcare reform", "climate change", "economic stimulus", "education funding"],
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "aim": ["reduce costs", "expand access", "improve safety standards", "boost innovation"],
    "think_tank": ["Brookings Institution", "RAND Corporation", "Pew Research", "Urban Institute"],
    "company": ["Apple", "Amazon", "Microsoft", "Google", "Tesla"],
    "amount": ["12.4", "8.7", "22.1", "5.3", "34.6"],
    "pct": ["4", "7", "12", "2", "9"],
    "reason": ["strong consumer demand", "cost reductions", "market expansion", "product innovation"],
    "rise": ["3.2", "5.1", "1.8", "7.4", "2.6"],
    "agency": ["CDC", "FDA", "WHO", "NHS", "ECDC"],
    "disease": ["influenza", "COVID-19", "measles", "diabetes", "hypertension", "cancer"],
    "region": ["the Northeast", "Europe", "Southeast Asia", "the Midwest", "sub-Saharan Africa"],
    "miracle": ["turmeric", "lemon juice", "coconut oil", "baking soda", "apple cider vinegar"],
    "days": ["3", "7", "14", "2", "5"],
    "action": ["rigging elections", "poisoning water supplies", "suppressing cures", "stealing funds"],
    "city": ["Ohio", "Texas", "Florida", "Michigan", "California"],
    "problem": ["belly fat", "joint pain", "hair loss", "vision problems", "memory loss"],
    "celebrity": ["a top politician", "a famous actor", "a billionaire", "a world leader"],
    "product": ["this natural extract", "this herbal formula", "this ancient tea", "this root powder"],
}

import random
random.seed(42)
np.random.seed(42)


def fill_template(template):
    """Fill a template with random values."""
    import re
    result = template
    for key, values in FILL.items():
        pattern = "{" + key + "}"
        while pattern in result:
            result = result.replace(pattern, random.choice(values), 1)
    return result


def generate_dataset(n_real=5000, n_fake=5000):
    texts, labels = [], []
    for _ in range(n_real):
        tmpl = random.choice(REAL_TEMPLATES)
        texts.append(fill_template(tmpl))
        labels.append(0)  # 0 = REAL
    for _ in range(n_fake):
        tmpl = random.choice(FAKE_TEMPLATES)
        texts.append(fill_template(tmpl))
        labels.append(1)  # 1 = FAKE
    return texts, labels


log.info("Generating synthetic training data...")
texts, labels = generate_dataset(n_real=5000, n_fake=5000)
log.info(f"  Total samples: {len(texts)} ({labels.count(0)} REAL, {labels.count(1)} FAKE)")

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
log.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ── TF-IDF ───────────────────────────────────────────────────────────────────
log.info("Fitting TF-IDF vectorizer (8000 features)...")
tfidf = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\b[a-zA-Z]{2,}\b",
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
log.info("  TF-IDF fitted.")

# ── Ensemble model ────────────────────────────────────────────────────────────
log.info("Training 5-model ensemble...")

lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
sgd = SGDClassifier(loss="modified_huber", max_iter=1000, random_state=42, tol=1e-3)
svc_base = LinearSVC(C=1.0, max_iter=2000, random_state=42)
svc = CalibratedClassifierCV(svc_base, cv=3)
rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
gb  = GradientBoostingClassifier(n_estimators=100, random_state=42)

ensemble = VotingClassifier(
    estimators=[("lr", lr), ("sgd", sgd), ("svc", svc), ("rf", rf), ("gb", gb)],
    voting="soft",
)

log.info("  Fitting ensemble (this may take 1-2 minutes)...")
ensemble.fit(X_train_tfidf, y_train)
log.info("  Ensemble fitted.")

# ── Evaluate ──────────────────────────────────────────────────────────────────
log.info("Evaluating on test set...")
y_pred = ensemble.predict(X_test_tfidf)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

# Per-class accuracy
real_mask = [i for i, l in enumerate(y_test) if l == 0]
fake_mask = [i for i, l in enumerate(y_test) if l == 1]
real_acc = accuracy_score([y_test[i] for i in real_mask], [y_pred[i] for i in real_mask])
fake_acc = accuracy_score([y_test[i] for i in fake_mask], [y_pred[i] for i in fake_mask])

log.info(f"  Accuracy:  {acc:.4f}")
log.info(f"  Precision: {prec:.4f}")
log.info(f"  Recall:    {rec:.4f}")
log.info(f"  F1-score:  {f1:.4f}")
log.info(f"  REAL acc:  {real_acc:.4f}")
log.info(f"  FAKE acc:  {fake_acc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
log.info("Saving model files to models/...")

model_path = os.path.join("models", "model.joblib")
tfidf_path = os.path.join("models", "tfidf.joblib")
config_path = os.path.join("models", "config.json")

joblib.dump(ensemble, model_path, compress=3)
log.info(f"  Saved: {model_path}  ({os.path.getsize(model_path):,} bytes)")

joblib.dump(tfidf, tfidf_path, compress=3)
log.info(f"  Saved: {tfidf_path}  ({os.path.getsize(tfidf_path):,} bytes)")

config = {
    "accuracy": round(acc, 10),
    "precision": round(prec, 10),
    "recall": round(rec, 10),
    "f1_score": round(f1, 10),
    "real_accuracy": round(real_acc, 10),
    "fake_accuracy": round(fake_acc, 10),
    "threshold": 0.5,
    "model_type": "Advanced Ensemble (LR+SGD+SVC+RF+GB)",
    "total_training_samples": len(X_train),
    "total_datasets": 1,
    "tfidf_features": 8000,
}
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
log.info(f"  Saved: {config_path}")

log.info("")
log.info("=" * 60)
log.info("  MODEL TRAINING COMPLETE — ready to run app.py")
log.info("=" * 60)
log.info(f"  Final Accuracy: {acc*100:.2f}%")
log.info("")
