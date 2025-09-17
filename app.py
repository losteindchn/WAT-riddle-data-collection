import streamlit as st
import json
import numpy as np
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os, gzip, base64

# ------------------ Load riddles ------------------
def load_riddles():
    if "riddles" in st.secrets:
        return json.loads(st.secrets["riddles"])
    else:
        with open("riddles.json", "r", encoding="utf-8") as f:
            return json.load(f)

# ------------------ Load embedding ------------------
def load_embedding():
    if "embedding_base64" in st.secrets:
        compressed = base64.b64decode(st.secrets["embedding_base64"])
        data = gzip.decompress(compressed).decode("utf-8").splitlines()
        return data
    else:
        with open("edge_list_ZH.inf_coord", "r") as f:
            return f.readlines()

# ------------------ Model ------------------
class SimpleConnectionModel:
    def __init__(self, coord_lines):
        lines = iter(coord_lines)
        for _ in range(8):
            next(lines)
        self.beta = float(next(lines).strip().split()[3])
        self.mu = float(next(lines).strip().split()[3])
        self.R = float(next(lines).strip().split()[3])

        # Names, kappa, theta
        arr = np.loadtxt(coord_lines[11:], dtype=str)  # 从第12行开始是数据
        self.names = arr[:,0]
        self.kappa = arr[:,1].astype(float)
        self.theta = arr[:,2].astype(float)
        self.name_to_idx = {name: i for i, name in enumerate(self.names)}

        self.LARGE_NUMBER = 1e10

    def raw_hyperbolic_distance(self, v1, v2):
        i1 = self.name_to_idx.get(v1, -1)
        i2 = self.name_to_idx.get(v2, -1)
        if i1 == -1 or i2 == -1:
            return self.LARGE_NUMBER
        delta_theta = np.pi - np.abs(np.pi - np.abs(self.theta[i1] - self.theta[i2]))
        distance = (self.R * delta_theta) / (self.mu * self.kappa[i1] * self.kappa[i2])
        return np.clip(distance, 0, self.LARGE_NUMBER)

    def scale_probability(self, raw_p, stretch_factor=200, min_val=0.05, max_val=0.95):
        raw_p = np.clip(raw_p, 1e-12, 1 - 1e-12)
        transformed = 1 / (1 + np.exp(-stretch_factor * (raw_p - 0.002)))
        return np.clip(min_val + (max_val - min_val) * transformed, min_val, max_val)

    def connection_probability(self, v1, v2, return_raw=False):
        try:
            dist = self.raw_hyperbolic_distance(v1, v2)
            raw_prob = 1 / (1 + dist ** self.beta)
            scaled_prob = self.scale_probability(raw_prob)
            return (raw_prob, scaled_prob) if return_raw else scaled_prob
        except:
            return (1e-8, 0.05) if return_raw else 0.05

# ------------------ Google Sheets ------------------
def init_gsheet():
    if "gcp_service_account" in st.secrets:
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            creds_dict,
            ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        )
    else:
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            "my-project-bayesian-app-977783cbe754.json",
            ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        )
    client = gspread.authorize(creds)
    sheet = client.open_by_url(
        "https://docs.google.com/spreadsheets/d/1wWT-7zbYjA3fdY8L_-OS8wJ7MdQ6UYiOLu-swYKUPBk/edit#gid=0"
    ).sheet1
    return sheet

# ------------------ Load ------------------
riddles = load_riddles()
embedding_lines = load_embedding()
model = SimpleConnectionModel(embedding_lines)
sheet = init_gsheet()

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="海龟汤联想实验", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = "intro"
    st.session_state.index = 0
    st.session_state.records = []
    st.session_state.participant_id = ""

if st.session_state.page == "intro":
    st.title("🧠 海龟汤联想实验")
    st.session_state.participant_id = st.text_input("请输入实验编号或随机ID")
    if st.button("开始实验"):
        if st.session_state.participant_id.strip():
            st.session_state.page = "prior"
        else:
            st.warning("请输入ID后才能开始。")

elif st.session_state.page == "prior":
    idx = st.session_state.index
    data = riddles[idx]
    st.markdown(f"### 🧩 谜面 {idx+1}")
    st.markdown(data["riddle_text"])
    st.markdown(f"🔹 锚点词：**{data['anchor_word']}**")
    prior = st.slider("你的先验概率", 0.0, 1.0, 0.5, step=0.01)

    if st.button("下一步 ➡️"):
        st.session_state.current_prior = prior
        st.session_state.page = "update"

elif st.session_state.page == "update":
    idx = st.session_state.index
    data = riddles[idx]
    a_word = data["phase1_samples"]
    c_words = data["answer_pool"]

    raw_ac_probs, scaled_ac_probs = zip(*[model.connection_probability(a_word, c, return_raw=True) for c in c_words])
    mean_raw = np.mean(raw_ac_probs)
    mean_scaled = np.mean(scaled_ac_probs)

    st.markdown(f"### 🧩 谜面 {idx+1}（更新阶段）")
    st.markdown(f"更新词：**{a_word}** → 平均连接概率：**{mean_scaled:.3f}**")
    updated = st.slider("更新后的概率", 0.0, 1.0, 0.5, step=0.01)
    conf = st.slider("信心程度", 0.0, 1.0, 0.5, step=0.01)

    if st.button("提交并进入下一题"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.records.append([
            st.session_state.participant_id, idx+1, data["riddle_text"], data["anchor_word"], a_word,
            ",".join(c_words), st.session_state.current_prior,
            mean_raw, mean_scaled,
            updated, conf, timestamp
        ])
        sheet.append_row(st.session_state.records[-1])
        st.session_state.index += 1

        if st.session_state.index >= len(riddles):
            st.success("🎉 你已完成全部谜题！感谢参与！")
            st.dataframe(pd.DataFrame(
                st.session_state.records,
                columns=["participant_id", "index", "riddle", "B", "A", "C_pool",
                         "prior", "A-C_prob_raw", "A-C_prob_scaled",
                         "posterior", "confidence", "timestamp"]
            ))
        else:
            st.session_state.page = "prior"
