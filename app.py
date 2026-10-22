"""Streamlit interface for the locally trained MoleculeNet BBBP model."""
from pathlib import Path
import pandas as pd
import streamlit as st
from rdkit.Chem import Draw
from predictor import load_artifacts, predict_one, predict_batch
from features import InvalidSmilesError

st.set_page_config(page_title="BBB Penetration Predictor", page_icon="🧬", layout="wide")
st.markdown("""<style>
.stApp{background:#f6f8fa;color:#102a43}.block-container{max-width:1180px;padding-top:2rem}
[data-testid="stMetric"]{background:white;border:1px solid #e3eaee;padding:16px;border-radius:12px}
.hero{padding:22px 26px;background:#fff;border:1px solid #e3eaee;border-left:5px solid #168b8b;border-radius:12px;margin-bottom:20px}
</style>""",unsafe_allow_html=True)

@st.cache_resource
def artifacts(): return load_artifacts()

st.title("BBB Penetration Predictor")
st.markdown('<div class="hero"><b>Estimate blood–brain barrier (BBB) class from molecular structure.</b><br>Enter a SMILES string to calculate RDKit descriptors and score it with the locally trained Random Forest model.</div>',unsafe_allow_html=True)
try:
    model, meta = artifacts()
except Exception as exc:
    model = meta = None
    st.error(f"Model is not ready: {exc}")
    st.info("From the project folder, install dependencies, then run `python train_model.py`.")

tabs=st.tabs(["Predict a molecule","Batch prediction","Model performance","Data & methodology"])
with tabs[0]:
    examples={"Caffeine":"Cn1c(=O)c2c(ncn2C)n(C)c1=O","Aspirin":"CC(=O)Oc1ccccc1C(=O)O","Nicotine":"CN1CCC[C@H]1c1cccnc1","Acetaminophen":"CC(=O)Nc1ccc(O)cc1"}
    choice=st.selectbox("Example molecule",["Choose an example…",*examples])
    default=examples.get(choice,"")
    smiles=st.text_input("Molecular SMILES",value=default,placeholder="e.g. CCO",help="SMILES is a text representation of molecular structure.")
    if st.button("Predict BBB class",type="primary",disabled=model is None):
        try:
            result=predict_one(smiles,model,meta)
            st.session_state["prediction"]=(smiles,result)
        except InvalidSmilesError as exc: st.error(f"Invalid SMILES: {exc}")
        except Exception as exc: st.error(f"Prediction failed: {exc}")
    if st.session_state.get("prediction"):
        smi,res=st.session_state["prediction"]
        st.subheader("Prediction")
        st.success(f"Predicted {res['label']}")
        st.warning("Computational estimate only; this prediction is not experimentally confirmed and is not a clinical recommendation.")
        st.caption(res["applicability"]["warning"] + " The applicability check is a training-descriptor heuristic, not a validated domain boundary.")
        if res.get("probabilities"):
            cols=st.columns(len(res["probabilities"]))
            for col,(label,p) in zip(cols,res["probabilities"].items()): col.metric(label,f"{p:.1%}")
            st.caption("Probabilities are model estimates and have not been calibrated.")
        left,right=st.columns([1,1.4])
        with left: st.image(Draw.MolToImage(res["mol"],size=(420,320)),caption="RDKit 2D structure",width="stretch")
        with right:
            st.markdown("#### Molecular properties")
            names={"molecular_weight":"Molecular weight (Da)","logp":"LogP","tpsa":"TPSA (Å²)","h_bond_donors":"H-bond donors","h_bond_acceptors":"H-bond acceptors","rotatable_bonds":"Rotatable bonds","ring_count":"Ring count","aromatic_rings":"Aromatic rings","formal_charge":"Formal charge"}
            for k,label in names.items(): st.write(f"**{label}:** {res['descriptors'][k]:.2f}")
        with st.expander("All descriptors and interpretation"):
            st.dataframe(pd.DataFrame([res["descriptors"]]).T.rename(columns={0:"Value"}),use_container_width=True)
            st.write("MW reflects molecular size; LogP estimates lipophilicity; TPSA and H-bond counts describe polar surface and interactions; rotatable bonds approximate flexibility; rings/aromaticity describe topology; charge can affect distribution. None alone determines BBB penetration.")
        st.caption("Research and educational estimate only. Not medical advice, experimental evidence, or a clinical recommendation.")
with tabs[1]:
    st.write("Upload a CSV with a `smiles` column. Every source row is retained; failures include an error message.")
    upload=st.file_uploader("CSV file",type=["csv"])
    if upload:
        try:
            batch=pd.read_csv(upload)
            if "smiles" not in batch: st.error("CSV is missing the required `smiles` column.")
            elif model is not None:
                output=predict_batch(batch,model,meta)
                st.dataframe(output,width="stretch")
                st.download_button("Download prediction results",output.to_csv(index=False).encode(),"bbbp_predictions.csv","text/csv")
        except Exception as exc: st.error(f"Could not process CSV: {exc}")
with tabs[2]:
    if meta:
        st.subheader("Held-out scaffold test"); met=meta["test_metrics"]
        cols=st.columns(5)
        for c,(label,key) in zip(cols,[("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1","f1"),("ROC-AUC","roc_auc")]): c.metric(label,"Undefined" if met[key] is None else f"{met[key]:.3f}")
        st.caption("All three models use the same held-out scaffold test set. No test-set tuning or model selection was performed. ROC-AUC is undefined when the test set contains one class.")
        a,b=st.columns(2)
        for col,name,title in [(a,"confusion_matrix.png","Random Forest confusion matrix"),(b,"model_comparison_test.png","Same held-out test comparison")]:
            p=Path(__file__).parent/"reports"/name
            with col:
                st.markdown(f"#### {title}")
                if p.exists(): st.image(str(p),width="stretch")
                else: st.info("Run training to generate this chart.")
        for name,title in [("roc_curve.png","Random Forest ROC curve"),("feature_importance.png","Random Forest feature importance")]:
            p=Path(__file__).parent/"reports"/name
            st.markdown(f"#### {title}")
            if p.exists(): st.image(str(p),width=700)
            elif name=="roc_curve.png": st.info("ROC curve is generated only when both test classes are represented.")
        st.write("**Split:**",meta["training_config"]["split"]); st.json(meta["split_stats"])
        st.write("**Dataset summary:**"); st.json(meta["dataset_stats"])
        st.write("**Random Forest classification report (held-out test):**")
        st.dataframe(pd.DataFrame(met["classification_report"]).T,width="stretch")
        comparison = pd.DataFrame(meta["heldout_model_metrics"]).T[["accuracy","precision","recall","f1","roc_auc"]]
        comparison.index.name = "model"
        st.write("**Model comparison (same held-out test rows):**")
        st.dataframe(comparison,width="stretch")
        st.write("**Training-only cross-validation:**"); st.dataframe(pd.DataFrame(meta["cv_results"]).T,width="stretch")
    else: st.info("Train the models to view evaluation.")
with tabs[3]:
    st.markdown("""### Data and method
Source: [MoleculeNet BBBP](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv). The target is `p_np` (1 = BBB permeable, 0 = non-permeable).

Cleaning validates the schema, drops missing/non-binary labels and invalid SMILES, canonicalizes structures, removes duplicate structures, and excludes a canonical structure entirely if its labels conflict. Descriptor failures are excluded from training and counted.

Fourteen deterministic RDKit descriptors are used: molecular weight, LogP, TPSA, H-bond donors/acceptors, rotatable bonds, aromatic/ring counts, fraction sp3, heavy atoms, formal charge, molar refractivity, heteroatoms, and aliphatic rings.

Models: Random Forest (primary), scaled Logistic Regression, and scaled RBF SVM. All three are compared on the same held-out test set; the RF remains primary by specification, not because of test-set selection. Hyperparameters are fixed and CV is stratified on training data only. The final test set randomly holds out 20% of Bemis–Murcko scaffold groups, including one shared empty scaffold group for acyclic molecules. Label proportions are reported but do not drive group assignment. Test data never informs fitting or tuning. Scaffold generalization is harder than random splitting among familiar chemical families.

### Limitations
BBB penetration is complex biology and cannot be reliably determined from a few descriptors. BBBP may have class imbalance and experimental inconsistencies. Scaffold testing reduces some structural leakage but does not eliminate bias. RF probabilities are model estimates and have not been evaluated for calibration. The descriptor-range and standardized-distance applicability warning is a simple heuristic, not a validated applicability-domain method. Predictions outside the training distribution can be unreliable. This research tool cannot replace experiments or expert review and makes no claim about clinical effectiveness, safety, or human brain exposure.""")
