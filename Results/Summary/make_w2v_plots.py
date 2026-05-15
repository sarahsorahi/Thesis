"""
این کد فقط cosine heatmap و confusion matrix رو برای Word2Vec می‌سازه.

اجرا:
    py -3.11 make_w2v_plots.py --data am_corpus_final.csv --w2v cc.fa.300.vec
"""

import argparse
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("results_v2")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--w2v',  type=str, required=True)
    args = parser.parse_args()

    # ── Load splits ───────────────────────────────────────────────
    train_df = pd.read_csv(OUTPUT_DIR / "train_split.csv", encoding='utf-8-sig')
    test_df  = pd.read_csv(OUTPUT_DIR / "test_split.csv",  encoding='utf-8-sig')
    full_df  = pd.concat([train_df, test_df]).reset_index(drop=True)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # ── Load Word2Vec model ───────────────────────────────────────
    from gensim.models import KeyedVectors
    print(f"Loading model: {args.w2v}")
    model = KeyedVectors.load_word2vec_format(args.w2v, binary=False)
    print(f"✓ Model loaded: {len(model)} words")

    # ── Extract vectors ───────────────────────────────────────────
    def get_vec(row):
        token = str(row.get('token', row.get('lemma_guess', ''))).strip()
        if token and token in model:
            return model[token]
        words = str(row['sentence']).split()
        for w in words:
            if w.endswith('م') and w in model:
                return model[w]
        valid = [model[w] for w in words if w in model]
        return np.mean(valid, axis=0) if valid else np.zeros(model.vector_size)

    print("Extracting vectors...")
    full_vecs = np.array([get_vec(row) for _, row in tqdm(full_df.iterrows(), total=len(full_df))])
    train_vecs = full_vecs[:len(train_df)]
    test_vecs  = full_vecs[len(train_df):]

    # ── SVM ───────────────────────────────────────────────────────
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

    le = LabelEncoder()
    le.fit(full_df['category'].tolist())
    y_train = le.transform(train_df['category'].tolist())
    y_test  = le.transform(test_df['category'].tolist())

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train_vecs)
    X_test  = scaler.transform(test_vecs)

    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro-F1: {f1_score(y_test, y_pred, average='macro'):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    classes = le.classes_.tolist()

    # ── Cosine similarity ─────────────────────────────────────────
    from sklearn.metrics.pairwise import cosine_similarity

    cats = ['verbal', 'possessive', 'copular']
    labels_arr = np.array(full_df['category'].tolist())
    centroids = {c: full_vecs[labels_arr == c].mean(0) for c in cats}

    matrix = np.zeros((3, 3))
    for i, c1 in enumerate(cats):
        for j, c2 in enumerate(cats):
            matrix[i, j] = cosine_similarity(
                centroids[c1].reshape(1, -1),
                centroids[c2].reshape(1, -1)
            )[0, 0]

    print("\nCosine similarities:")
    for i, c1 in enumerate(cats):
        for j, c2 in enumerate(cats):
            if i != j:
                print(f"  {c1} vs {c2}: {matrix[i,j]:.4f}")

    # ── Plot cosine heatmap ───────────────────────────────────────
    cat_labels = ['Verbal (رفتم)', 'Possessive (کتابم)', 'Copular (خوشحالم)']

    fig_cos = px.imshow(
        matrix,
        labels=dict(x="Category", y="Category", color="Cosine Similarity"),
        x=cat_labels, y=cat_labels,
        color_continuous_scale='Blues',
        zmin=0.6, zmax=1.0,
        title="Cosine Similarity: Word2Vec (pre-trained)",
        text_auto='.3f',
    )
    fig_cos.update_layout(
        width=520, height=460,
        font=dict(family="Times New Roman", size=13),
        title=dict(font=dict(size=15, color='#1F3864'), x=0.5),
    )
    cos_path = OUTPUT_DIR / "cosine_word2vec_pretrained.html"
    fig_cos.write_html(str(cos_path))
    print(f"\n✓ Saved: {cos_path}")

    # ── Plot confusion matrix ─────────────────────────────────────
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=classes, y=classes,
        color_continuous_scale='Blues',
        title="Confusion Matrix: Word2Vec (pre-trained)",
        text_auto=True,
    )
    fig_cm.update_layout(
        width=520, height=460,
        font=dict(family="Times New Roman", size=13),
        title=dict(font=dict(size=15, color='#1F3864'), x=0.5),
    )
    cm_path = OUTPUT_DIR / "confusion_word2vec_pretrained.html"
    fig_cm.write_html(str(cm_path))
    print(f"✓ Saved: {cm_path}")

    print("\n✓ Done! Open both HTML files in your browser.")


if __name__ == "__main__":
    main()
