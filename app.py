import streamlit as st
import json, numpy as np, pandas as pd, gspread, gzip, base64, time, random
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# ------------------ Load riddles ------------------
def load_riddles():
    if "riddles" not in st.secrets:
        raise RuntimeError("❌ Missing riddles in secrets.toml")
    return json.loads(st.secrets["riddles"])

# ------------------ Load embedding ------------------
def load_embedding():
    if "embedding_base64" not in st.secrets:
        raise RuntimeError("❌ Missing embedding_base64 in secrets.toml")
    compressed = base64.b64decode(st.secrets["embedding_base64"])
    return gzip.decompress(compressed).decode("utf-8").splitlines()

# ------------------ Model ------------------
class SimpleConnectionModel:
    def __init__(self, coord_lines):
        lines = iter(coord_lines)
        for _ in range(8): next(lines)
        self.beta = float(next(lines).strip().split()[3])
        self.mu = float(next(lines).strip().split()[3])
        self.R = float(next(lines).strip().split()[3])
        arr = np.loadtxt(coord_lines[11:], dtype=str)
        self.names, self.kappa, self.theta = arr[:,0], arr[:,1].astype(float), arr[:,2].astype(float)
        self.name_to_idx = {n:i for i,n in enumerate(self.names)}
        self.LARGE_NUMBER = 1e10

    def raw_hyperbolic_distance(self,v1,v2):
        i1,i2 = self.name_to_idx.get(v1,-1), self.name_to_idx.get(v2,-1)
        if -1 in (i1,i2): return self.LARGE_NUMBER
        dtheta = np.pi - np.abs(np.pi - np.abs(self.theta[i1]-self.theta[i2]))
        return np.clip((self.R*dtheta)/(self.mu*self.kappa[i1]*self.kappa[i2]),0,self.LARGE_NUMBER)

    def connection_probability(self,v1,v2):
        try:
            dist=self.raw_hyperbolic_distance(v1,v2)
            return 1/(1+dist**self.beta)   # 只返回 raw 概率
        except:
            return 1e-12   # 出错时返回极小概率

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

# ------------------ Load ------------------
riddles=load_riddles(); model=SimpleConnectionModel(load_embedding()); sheet=init_gsheet()

# ------------------ Session state ------------------
st.set_page_config(page_title="海龟汤实验",layout="centered")
if "page" not in st.session_state:
    st.session_state.update({
        "page":"intro","section":"anchor","index":0,"records":[],
        "participant_id":"","explore_count":0,"explore_start":None,
        "order":[], "phase1_ids":[], "phase2_ids":[]
    })

# ------------------ Intro ------------------
if st.session_state.page=="intro":
    st.title("🧠 海龟汤实验")
    st.session_state.participant_id=st.text_input("请输入实验编号或随机ID")
    if st.button("开始实验"):
        if st.session_state.participant_id.strip():
            ids=list(range(len(riddles)))
            random.shuffle(ids)
            st.session_state.order=ids
            st.session_state.phase1_ids=ids[:8]
            st.session_state.phase2_ids=ids[8:]
            # 记录分配结果
            sheet.append_row([st.session_state.participant_id,"ORDER",
                              ",".join(map(str,ids)),datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            st.session_state.page="anchor_intro"
        else:
            st.warning("请输入ID后才能开始。")

# ------------------ Section 1 intro + checkpoint ------------------
elif st.session_state.page=="anchor_intro":
    st.subheader("阶段一：锚定任务")
    st.write("请先完成一个简单检查任务以确认注意力。请输入指定词语：**注意力**")
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
    idx=st.session_state.phase1_ids[st.session_state.index]; data=riddles[idx]
    st.markdown(f"### 谜面 {st.session_state.index+1}"); st.markdown(data["riddle_text"])
    st.markdown(f"🔹 锚点词：**{data['anchor_word']}**")
    prior=st.slider("你的先验概率",0.0,1.0,0.5,0.01)
    if st.button("下一步"):
        st.session_state.current_prior=prior; st.session_state.page="update"

elif st.session_state.page=="update":
    idx=st.session_state.phase1_ids[st.session_state.index]; data=riddles[idx]
    a_word=data["phase1_samples"]; c_words=data["answer_pool"]
    probs=[model.connection_probability(a_word,c) for c in c_words]
    max_raw=np.max(probs)
    st.markdown(f"### 谜面 {st.session_state.index+1}（更新阶段）")
    st.write(f"更新词：**{a_word}** → 最高连接概率：**{max_raw:.6f}**")
    updated=st.slider("更新后的概率",0.0,1.0,0.5,0.01); conf=st.slider("信心程度",0.0,1.0,0.5,0.01)
    if st.button("提交"):
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([st.session_state.participant_id,idx,"ANCHOR",
                          data["riddle_text"],data["anchor_word"],a_word,
                          ",".join(c_words),st.session_state.current_prior,
                          max_raw,"",updated,conf,timestamp])
        st.session_state.index+=1
        if st.session_state.index>=len(st.session_state.phase1_ids):
            st.session_state.page="explore_intro"
        else: st.session_state.page="prior"

# ------------------ Phase 2: Exploration tasks ------------------
elif st.session_state.page=="explore":
    idx=st.session_state.phase2_ids[st.session_state.index]; data=riddles[idx]
    st.markdown(f"### 谜面 {st.session_state.index+1+len(st.session_state.phase1_ids)}")
    st.markdown(data["riddle_text"])
    word=st.text_input("请输入你的探索词")
    if st.button("提交探索词"):
        if not word.strip(): st.warning("请输入一个词。")
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
                st.session_state.index+=1; st.session_state.explore_count=0; st.session_state.explore_start=time.time()
                if st.session_state.index>=len(st.session_state.phase2_ids):
                    st.success("🎉 所有谜题完成！")
                else: st.session_state.page="explore"

