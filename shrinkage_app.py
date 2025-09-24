import streamlit as st
import json, numpy as np, pandas as pd, gspread, gzip, base64, time, random, os
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import math

# ------------------ Load riddles ------------------
def load_riddles():
    if "riddles_path" not in st.secrets:
        raise RuntimeError("❌ Missing riddles_path in secrets.toml")
    path = st.secrets["riddles_path"]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------ Load embedding ------------------
def load_embedding():
    if "embedding_path" not in st.secrets:
        raise RuntimeError("❌ Missing embedding_path in secrets.toml")
    path = st.secrets["embedding_path"]
    with open(path, "r", encoding="utf-8") as f:
        b64 = f.read().strip()
    compressed = base64.b64decode(b64)
    return gzip.decompress(compressed).decode("utf-8").splitlines()

# ------------------ Load shrinkage ------------------
def load_shrinkage(group):
    if "shrinkage_paths" not in st.secrets:
        return {}
    shrinkage_paths = st.secrets["shrinkage_paths"]
    if group not in shrinkage_paths:
        return {}
    path = shrinkage_paths[group]
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        b64 = f.read().strip()
    compressed = base64.b64decode(b64)
    data = gzip.decompress(compressed).decode("utf-8")
    return json.loads(data)

# ------------------ Model ------------------
class SimpleConnectionModel:
    def __init__(self, coord_lines, shrinkage_weights=None, cap=10.0):
        lines = iter(coord_lines)
        for _ in range(8): next(lines)
        self.beta = float(next(lines).strip().split()[3])
        self.mu = float(next(lines).strip().split()[3])
        self.R = float(next(lines).strip().split()[3])
        arr = np.loadtxt(coord_lines[11:], dtype=str)
        self.names, self.kappa, self.theta = arr[:,0], arr[:,1].astype(float), arr[:,2].astype(float)
        self.name_to_idx = {n:i for i,n in enumerate(self.names)}
        self.LARGE_NUMBER = 1e10

        self.shrinkage = shrinkage_weights if shrinkage_weights else {}
        self.cap = cap

    def raw_hyperbolic_distance(self,v1,v2):
        i1,i2 = self.name_to_idx.get(v1,-1), self.name_to_idx.get(v2,-1)
        if -1 in (i1,i2): return self.LARGE_NUMBER
        dtheta = np.pi - np.abs(np.pi - np.abs(self.theta[i1]-self.theta[i2]))
        return np.clip((self.R*dtheta)/(self.mu*self.kappa[i1]*self.kappa[i2]),0,self.LARGE_NUMBER)

    def connection_probability(self, v1, v2):
        # ---- 保证 self-loop 始终为 1 ----
        if v1 == v2:
            return 1.0
        try:
            dist = self.raw_hyperbolic_distance(v1, v2)
            p_raw = 1/(1+dist**self.beta)
            key = f"{v1}||{v2}"
            w = self.shrinkage.get(key, 1.0)
            if self.cap:
                w = max(min(w, self.cap), 1.0/self.cap)
            # ---- logit 融合 ----
            if p_raw <= 0:
                return 1e-12
            if p_raw >= 1:
                return 1 - 1e-12
            logit = math.log(p_raw / (1 - p_raw))
            p_new = 1 / (1 + math.exp(-(logit + math.log(w))))
            return p_new
        except Exception:
            return 1e-12

# ------------------ Google Sheets ------------------
def init_gsheet():
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"],
        ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    )
    client=gspread.authorize(creds)
    return client.open_by_url(
        "https://docs.google.com/spreadsheets/d/1wWT-7zbYjA3fdY8L_-OS8wJ7MdQ6UYiOLu-swYKUPBk/edit#gid=0"
    ).sheet1

# ------------------ Glossary ------------------
def show_glossary():
    with st.expander("📖 名词解释（点击展开）"):
        st.markdown("""
        - **锚点词**：系统提供的一个参考词，你需要判断它是否可能是谜底。  
        - **谜底词**：谜题真正的答案词。  
        - **更新词**：系统在更新阶段提供的新词，它和谜底的连接概率会影响你的判断。  
        - **连接概率**：模型计算出的两个词语之间的语义相关程度，范围在 **0.0 ~ 1.0** 之间。  
        - **输入要求**：探索词必须是**单个中文词语**（如“警察”、“书签”），不要输入句子或符号。  
        """)

# ------------------ Load ------------------
riddles = load_riddles()
sheet = init_gsheet()

# ------------------ Session state ------------------
st.set_page_config(page_title="海龟汤实验",layout="centered")
if "page" not in st.session_state:
    st.session_state.update({
        "page":"intro","section":"anchor","index":0,"records":[],
        "participant_id":"","group":"FH",
        "explore_count":0,"explore_start":None,
        "order":[], "phase1_ids":[], "phase2_ids":[]
    })

# ------------------ Intro ------------------
if st.session_state.page=="intro":
    st.title("🧠 海龟汤实验")

    # ---- 导语 ----
    st.markdown("""
    👋 欢迎参加本实验！

    在本实验中，你将会看到一系列“海龟汤”谜题。  
    - **阶段一**：系统会给你一个谜面和一个锚点词，请你判断它和谜底的关系，并填写概率。  
    - **阶段二**：你可以自由输入词语，系统会反馈这些词和谜底的“连接概率”，帮助你探索。  

    🕒 **实验时长**：大约 20 分钟左右。  
    """)

    st.session_state.participant_id=st.text_input("请输入实验编号或随机ID")
    st.session_state.group=st.selectbox("请选择你的群体",["FH","MH","FN","MN"],index=0)

    if st.button("开始实验"):
        if st.session_state.participant_id.strip():
            ids=list(range(len(riddles)))
            random.shuffle(ids)
            st.session_state.order=ids
            st.session_state.phase1_ids=ids[:8]
            st.session_state.phase2_ids=ids[8:]

            # 加载 shrinkage 和 model
            shrink_dict = load_shrinkage(st.session_state.group)
            model = SimpleConnectionModel(load_embedding(), shrinkage_weights=shrink_dict)
            st.session_state.model = model

            # ---- 展示三对示例词的连接概率 ----
            examples = [("猫","窗户"),("水","草"),("绿色","蔬菜")]
            st.markdown("### 示例：连接概率演示")
            for w1,w2 in examples:
                prob = model.connection_probability(w1,w2)
                st.write(f"**{w1}** 和 **{w2}** 的连接概率 = {prob:.4f}")

            # 记录分配结果
            sheet.append_row([st.session_state.participant_id,"ORDER",
                              ",".join(map(str,ids)),st.session_state.group,
                              datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            st.session_state.page="anchor_intro"
        else:
            st.warning("请输入ID后才能开始。")

# ------------------ Section 1 intro + checkpoint ------------------
elif st.session_state.page=="anchor_intro":
    st.subheader("阶段一：锚定任务")
    st.write("请先完成一个简单检查任务以确认注意力。请输入指定词语：**注意力**")
    show_glossary()
    check=st.text_input("请输入：")
    if st.button("继续阶段一"):
        if check.strip()=="注意力":
            st.session_state.index=0
            st.session_state.page="prior"
        else:
            st.warning("请输入正确的词。")

# ------------------ Section 2 intro + checkpoint ------------------
elif st.session_state.page=="explore_intro":
    st.subheader("阶段二：自由探索任务")
    st.write("请先完成一个简单检查任务以确认注意力。请输入指定词语：**认真**")
    show_glossary()
    check=st.text_input("请输入：")
    if st.button("继续阶段二"):
        if check.strip()=="认真":
            st.session_state.index=0
            st.session_state.page="explore"
            st.session_state.explore_start=time.time()
            st.session_state.explore_count=0
        else:
            st.warning("请输入正确的词。")

# ------------------ Phase 1: Anchor tasks ------------------
elif st.session_state.page=="prior":
    model = st.session_state.model
    idx=st.session_state.phase1_ids[st.session_state.index]; data=riddles[idx]
    st.markdown(f"### 谜面 {st.session_state.index+1}"); st.markdown(data["riddle_text"])
    st.markdown(f"🔹 锚点词：**{data['anchor_word']}**")
    show_glossary()
    prior=st.slider("你的先验概率",0.0,1.0,0.5,0.01)
    if st.button("下一步"):
        st.session_state.current_prior=prior; st.session_state.page="update"

elif st.session_state.page=="update":
    model = st.session_state.model
    idx=st.session_state.phase1_ids[st.session_state.index]; data=riddles[idx]
    a_word=data["phase1_samples"]; c_words=data["answer_pool"]
    probs=[model.connection_probability(a_word,c) for c in c_words]
    max_raw=np.max(probs)
    st.markdown(f"### 谜面 {st.session_state.index+1}（更新阶段）")
    st.write(f"更新词：**{a_word}** → 最高连接概率：**{max_raw:.6f}**")
    show_glossary()
    updated=st.slider("更新后的概率",0.0,1.0,0.5,0.01)
    conf=st.slider("信心程度",0.0,1.0,0.5,0.01)
    if st.button("提交"):
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([st.session_state.participant_id,idx,"ANCHOR",
                          data["riddle_text"],data["anchor_word"],a_word,
                          ",".join(c_words),st.session_state.current_prior,
                          max_raw,"",updated,conf,timestamp])
        st.session_state.index+=1
        if st.session_state.index>=len(st.session_state.phase1_ids):
            st.session_state.page="explore_intro"
        else:
            st.session_state.page="prior"

# ------------------ Phase 2: Exploration tasks ------------------
elif st.session_state.page=="explore":
    model = st.session_state.model
    idx=st.session_state.phase2_ids[st.session_state.index]; data=riddles[idx]
    st.markdown(f"### 谜面 {st.session_state.index+1+len(st.session_state.phase1_ids)}")
    st.markdown(data["riddle_text"])
    show_glossary()
    word=st.text_input("请输入你的探索词")
    if st.button("提交探索词"):
        if not word.strip():
            st.warning("请输入一个词。")
        else:
            probs=[model.connection_probability(word,c) for c in data["answer_pool"]]
            max_raw=np.max(probs)
            st.write(f"反馈：**{word}** 与谜底最高连接概率 = {max_raw:.6f}")
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.explore_count+=1
            sheet.append_row([st.session_state.participant_id,idx,"EXPLORE",
                              data["riddle_text"],"",word,
                              ",".join(data["answer_pool"]),"",max_raw,"",
                              "",timestamp,st.session_state.explore_count])
            # stop conditions
            if word in data["answer_pool"] or st.session_state.explore_count>=30 or (time.time()-st.session_state.explore_start>600):
                st.success("该题探索结束！")
                st.session_state.index+=1
                st.session_state.explore_count=0
                st.session_state.explore_start=time.time()
                if st.session_state.index>=len(st.session_state.phase2_ids):
                    st.success("🎉 所有谜题完成！")
                else:
                    st.session_state.page="explore"

