import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
from PIL import Image
import glob

st.set_page_config(
    page_title="Cyber OMR Scorer",
    page_icon="🤖",
    layout="wide"
)

cyber_theme_css = """
/* --- General App Styling --- */
body {
    background-color: #0d0c22;
    color: #e0e0ff;
}
h1 {
    font-family: 'Consolas', 'Courier New', monospace;
    color: #a855f7;
    text-align: center;
    text-shadow: 0 0 8px #a855f7, 0 0 16px rgba(168, 85, 247, 0.5);
}
.st-emotion-cache-16idsys p {
    text-align: center;
    color: #c7d2fe;
    font-size: 1.1rem;
}
[data-testid="stSidebar"] {
    background-color: rgba(13, 12, 34, 0.8);
    backdrop-filter: blur(5px);
    border-right: 1px solid #a855f7;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #6366f1;
    background-color: rgba(30, 27, 75, 0.5);
    border-radius: 15px;
    padding: 25px;
}
[data-testid="stFileUploader"] label {
    color: #a855f7;
    font-size: 1.2rem;
    font-weight: bold;
}
[data-testid="stDownloadButton"] button, .stButton button {
    background: linear-gradient(45deg, #a855f7, #6366f1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 28px;
    font-weight: bold;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 0 15px rgba(168, 85, 247, 0.6);
}
[data-testid="stDownloadButton"] button:hover, .stButton button:hover {
    box-shadow: 0 0 25px #6366f1;
    transform: scale(1.05);
}
[data-testid="stAlert"] {
    background-color: rgba(49, 46, 129, 0.7);
    border: 1px solid #a855f7;
    border-radius: 10px;
    color: white;
    font-size: 1.1em;
}
.results-box {
    background-color: rgba(30, 27, 75, 0.6);
    border: 2px solid #a855f7;
    border-radius: 20px;
    padding: 30px;
    margin-top: 40px;
    box-shadow: 0 0 25px rgba(168, 85, 247, 0.5);
    backdrop-filter: blur(5px);
}
.results-box [data-testid="stDataFrame"] {
    border: 1px solid #4f46e5;
}
"""
st.markdown(f"<style>{cyber_theme_css}</style>", unsafe_allow_html=True)

ANSWER_KEYS_DIR = "./answer_keys"
RESULTS_DIR = "./results"
os.makedirs(ANSWER_KEYS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_answer_keys():
    return [os.path.basename(f) for f in glob.glob(f"{ANSWER_KEYS_DIR}/*.json")]

def load_answer_key(filename):
    try:
        with open(os.path.join(ANSWER_KEYS_DIR, filename), "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def detect_bubbles(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)
        if 150 < w * h < 3500 and 0.75 <= aspect <= 1.25:
            bubble_contours.append(cnt)
    return sorted(bubble_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

def is_filled(thresh_img, contour, fill_threshold=0.45):
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    filled = cv2.countNonZero(cv2.bitwise_and(thresh_img, thresh_img, mask=mask))
    total = cv2.countNonZero(mask)
    return (filled / total) > fill_threshold if total > 0 else False

def map_bubbles_to_answers(bubbles, thresh_img, options=4):
    answers = {}
    total_questions = len(answer_key_flat)
    num_bubbles_expected = total_questions * options
    
    if len(bubbles) < num_bubbles_expected * 0.9:
        st.warning(f"Detected only {len(bubbles)} bubbles, expected around {num_bubbles_expected}. Results may be inaccurate.")

    for i in range(0, min(len(bubbles), num_bubbles_expected), options):
        question_num = i // options + 1
        question_bubbles = sorted(bubbles[i:i+options], key=lambda c: cv2.boundingRect(c)[0])
        filled_options = [chr(65+j) for j, cnt in enumerate(question_bubbles) if is_filled(thresh_img, cnt)]
        
        if len(filled_options) == 1:
            answers[f"Q{question_num}"] = filled_options[0]
        elif not filled_options:
            answers[f"Q{question_num}"] = "-"
        else:
            answers[f"Q{question_num}"] = ",".join(filled_options)
    return answers

def score_student(student_answers, answer_key, correct_marks, incorrect_marks):
    scores = {}
    total = 0
    details = {}
    for subject, q_dict in answer_key.items():
        sub_score = 0
        for q, correct_ans in q_dict.items():
            student_ans = student_answers.get(q, "-")
            correct_ans_list = [ans.strip() for ans in correct_ans.split(',')]
            
            is_correct = False
            if student_ans != "-" and student_ans in correct_ans_list:
                is_correct = True

            if is_correct:
                sub_score += correct_marks
                details[q] = 'correct'
            elif student_ans == "-":
                details[q] = 'unanswered'
            else:
                sub_score += incorrect_marks
                details[q] = 'incorrect'
        
        scores[subject] = sub_score
        total += sub_score
    scores["total"] = total
    return scores, details

def draw_annotations(img, bubbles, answer_details, answer_key_flat, options=4):
    annotated_img = img.copy()
    total_questions = len(answer_key_flat)
    num_bubbles_expected = total_questions * options

    bubble_map = {f"Q{i//options + 1}-{chr(65 + (i % options))}": c for i, c in enumerate(bubbles)}

    for q_num in range(1, total_questions + 1):
        q_key = f"Q{q_num}"
        status = answer_details.get(q_key)
        correct_ans = answer_key_flat.get(q_key, "").split(',')

        for opt_idx, option in enumerate(['A', 'B', 'C', 'D']):
            bubble_key = f"{q_key}-{option}"
            if bubble_key in bubble_map:
                x, y, w, h = cv2.boundingRect(bubble_map[bubble_key])
                color = (100, 100, 100)
                
                if status == 'correct' and option in correct_ans:
                    color = (0, 255, 0)
                elif status == 'incorrect':
                    color = (0, 0, 255)
                elif status == 'unanswered' and option in correct_ans:
                    color = (255, 165, 0)

                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
    return annotated_img

st.sidebar.title("Configuration")
st.sidebar.header("Scoring Rules")
correct_marks = st.sidebar.number_input("Marks for Correct Answer", min_value=0.0, value=1.0, step=0.25)
incorrect_marks = st.sidebar.number_input("Marks for Incorrect Answer (Negative)", max_value=0.0, value=0.0, step=-0.25)

st.sidebar.header("Answer Key Management")
available_keys = get_answer_keys()
selected_key_file = st.sidebar.selectbox("Select Answer Key", available_keys, index=0 if available_keys else None)

if st.sidebar.button("Delete Selected Key", use_container_width=True):
    if selected_key_file:
        os.remove(os.path.join(ANSWER_KEYS_DIR, selected_key_file))
        st.sidebar.success(f"Deleted '{selected_key_file}'")
        st.rerun()

with st.sidebar.expander("Create or Edit Answer Key"):
    new_key_name = st.text_input("New Key Filename (e.g., setB.json)", value=selected_key_file or "")
    
    template_key = {
        "subject1": {"Q1": "A", "Q2": "B"},
        "subject2": {"Q3": "C", "Q4": "D"}
    }
    key_content_to_edit = load_answer_key(selected_key_file) if selected_key_file else template_key
    
    edited_key_str = st.text_area("Answer Key JSON", value=json.dumps(key_content_to_edit, indent=4), height=300)

    if st.button("Save Answer Key", use_container_width=True):
        if new_key_name and new_key_name.endswith(".json"):
            try:
                new_key_data = json.loads(edited_key_str)
                with open(os.path.join(ANSWER_KEYS_DIR, new_key_name), "w") as f:
                    json.dump(new_key_data, f, indent=4)
                st.success(f"Saved key '{new_key_name}'!")
                st.rerun()
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please check your syntax.")
        else:
            st.warning("Filename must not be empty and should end with .json")

st.title("🤖 Automated OMR Evaluation System 🤖")
st.markdown("<p>Upload OMR sheet images to detect and score answers automatically in a neon-drenched, cybernetic future.</p>", unsafe_allow_html=True)

answer_key = load_answer_key(selected_key_file) if selected_key_file else {}
answer_key_flat = {q: a for subj in answer_key.values() for q, a in subj.items()}

if not answer_key:
    st.warning("No answer key selected or the file is empty. Please create or select one from the sidebar.")
else:
    uploaded_files = st.file_uploader(
        "Upload OMR sheets here...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="You can upload one or more OMR images."
    )

    if uploaded_files:
        all_results = []
        st.write("---")
        
        for uploaded_file in uploaded_files:
            st.subheader(f"Processing: `{uploaded_file.name}`")
            col1, col2 = st.columns(2)
            
            try:
                img = Image.open(uploaded_file).convert("RGB")
                img_cv = np.array(img)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                
                with col1:
                    st.image(img, caption="Original Sheet", use_column_width=True)

                thresh = preprocess_image(img_cv)
                bubbles = detect_bubbles(thresh)
                student_answers = map_bubbles_to_answers(bubbles, thresh)
                
                if student_answers:
                    scores, details = score_student(student_answers, answer_key, correct_marks, incorrect_marks)
                    result = {"student_id": uploaded_file.name.split(".")[0], **scores, **student_answers}
                    all_results.append(result)
                    
                    annotated_image = draw_annotations(img_cv, bubbles, details, answer_key_flat)
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                    with col2:
                        st.image(annotated_image_rgb, caption="Processed & Verified Sheet", use_column_width=True)
                        st.metric("Total Score", f"{scores.get('total', 0)}")
                else:
                    col2.error("Could not process this sheet. Check bubble detection warnings.")

            except Exception as e:
                st.error(f"An error occurred while processing {uploaded_file.name}: {e}")
            st.write("---")

        if all_results:
            st.markdown('<div class="results-box">', unsafe_allow_html=True)
            st.header("Consolidated Results")
            df_results = pd.DataFrame(all_results)
            score_cols = ["student_id", "total"] + [s for s in scores if s != 'total']
            answer_cols = sorted([q for q in df_results.columns if q.startswith('Q')], key=lambda x: int(x[1:]))
            df_results = df_results[score_cols + answer_cols]

            st.dataframe(df_results)

            csv_file_path = os.path.join(RESULTS_DIR, "omr_scored_results.csv")
            df_results.to_csv(csv_file_path, index=False)

            with open(csv_file_path, "rb") as file:
                st.download_button(
                    label="Download Results as CSV",
                    data=file,
                    file_name="omr_scored_results.csv",
                    mime="text/csv"
                )
            st.markdown('</div>', unsafe_allow_html=True)

