import streamlit as st
import time
import re
import importlib
import api_client
importlib.reload(api_client)
from api_client import APIClient

st.set_page_config(
    page_title="Multimodal AI Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- MODERN PROFESSIONAL GLASSMORPHIC STYLING -----------------
st.markdown("""
<style>
    /* Global layout & typography */
    .stApp {
        background: radial-gradient(circle at top right, #131722 0%, #0d0f17 100%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Top navigation header */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 22px;
        background: rgba(22, 27, 34, 0.75);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(48, 54, 61, 0.6);
        border-radius: 14px;
        margin-bottom: 20px;
    }
    .app-title-box {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .app-logo {
        font-size: 1.8rem;
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .app-subtitle {
        color: #8b949e;
        font-size: 0.88rem;
    }

    /* Badges & Status Pills */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(56, 139, 253, 0.3);
        background: rgba(56, 139, 253, 0.12);
        color: #58a6ff;
    }
    .status-online {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #3fb950;
        box-shadow: 0 0 8px #3fb950;
    }

    /* Chat message enhancements */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 12px;
    }
    .switch-alert {
        padding: 8px 14px;
        background: rgba(210, 153, 34, 0.15);
        border-left: 3px solid #d29922;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.84rem;
        color: #e3b341;
    }
    .attachment-active-bar {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(31, 111, 235, 0.15);
        border: 1px solid rgba(56, 139, 253, 0.4);
        color: #79c0ff;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .autonomous-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(188, 140, 255, 0.12);
        border: 1px solid rgba(188, 140, 255, 0.35);
        color: #d2a8ff;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Clean popover styling */
    div[data-testid="stPopover"] > button {
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        padding: 0 !important;
        background: #21262d !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stPopover"] > button:hover {
        background: #30363d !important;
        border-color: #58a6ff !important;
        color: #58a6ff !important;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

client = APIClient()

# ----------------- SESSION STATE -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "attached_docs" not in st.session_state:
    # List of {"id": str, "name": str, "chunks": int, "key": str}
    st.session_state.attached_docs = []
if "processed_upload_keys" not in st.session_state:
    st.session_state.processed_upload_keys = set()

# Fetch operational models
models_data = client.get_available_models()
active_providers = models_data.get("active_providers", [])

# Build model selection options
model_options = ["auto"]
model_labels = {"auto": "⚡ Auto (Smart Multi-Model Router)"}

for m in models_data.get("models", []):
    if m["id"] != "mock":
        status_symbol = "🟢" if m["is_available"] else "⚪"
        model_options.append(m["id"])
        model_labels[m["id"]] = f"{status_symbol} {m['name']}"

model_options.append("mock")
model_labels["mock"] = "🧪 Offline Mock Provider"

# Check backend health
health = client.get_health()
backend_online = health.get("status") == "online"
db_type = health.get("database", {}).get("type", "sqlite").upper()

# ----------------- TOP APP HEADER (PROFESSIONAL & MINIMAL) -----------------
col_head1, col_head2 = st.columns([5, 3])
with col_head1:
    st.markdown(f"""
    <div class="app-title-box">
        <span class="app-logo">⚡ Multimodal AI</span>
        <span class="app-subtitle">| Autonomous Router · Multi-Doc RAG · Vision · Web · Art</span>
    </div>
    """, unsafe_allow_html=True)

with col_head2:
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        if backend_online:
            st.markdown(f"""
            <div style="text-align: right; padding-top: 4px;">
                <span class="status-badge"><span class="status-online"></span> {db_type} Vector Engine</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: right; padding-top: 4px;">
                <span style="color: #f85149; font-size: 0.85rem; font-weight: 600;">⚠️ Backend Offline</span>
            </div>
            """, unsafe_allow_html=True)
    with header_cols[1]:
        if st.session_state.messages:
            if st.button("🗑️ Reset", help="Clear conversation history and start fresh"):
                st.session_state.messages = []
                st.session_state.session_id = None
                st.rerun()

# ----------------- CHAT HISTORY DISPLAY -----------------
if not st.session_state.messages:
    # Modern clean welcome card
    st.markdown("""
    <div style="text-align: center; padding: 45px 20px 25px 20px;">
        <h2 style="font-weight: 700; color: #f0f6fc; margin-bottom: 8px;">How can I assist you today?</h2>
        <p style="color: #8b949e; font-size: 0.95rem; max-width: 620px; margin: 0 auto 18px auto; line-height: 1.6;">
            Autonomous multimodal assistant. Upload multiple documents for instant RAG, 
            ask coding questions, search the live web, generate AI artwork, or analyze photos.
        </p>
        <span class="autonomous-badge">🤖 Autonomous Mode: Just type your request naturally</span>
    </div>
    """, unsafe_allow_html=True)
else:
    for idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            if msg.get("switched_from"):
                st.markdown(
                    f'<div class="switch-alert">⚡ <b>Auto-switched:</b> From <code>{msg["switched_from"]}</code> '
                    f'to <code>{msg["model_used"]}</code> ({msg.get("switch_reason") or "Failover"}).</div>',
                    unsafe_allow_html=True
                )
            st.markdown(content)

            # AI Generated Image
            if msg.get("image_bytes"):
                st.image(msg["image_bytes"], caption=msg.get("image_caption", "AI Generated Artwork"), use_container_width=True)
                col_img_dl, col_img_up = st.columns([1, 1])
                with col_img_dl:
                    st.download_button(
                        label="📥 Download Artwork (.jpg)",
                        data=msg["image_bytes"],
                        file_name=f"artwork_{idx}.jpg",
                        mime="image/jpeg",
                        key=f"dl_gen_{idx}"
                    )
                with col_img_up:
                    if not msg.get("is_upscaled"):
                        if st.button("✨ AI Upscale (2x / 4K Clarity)", key=f"btn_upscale_{idx}", help="Enhance resolution and sharpen with AI Super-Resolution"):
                            with st.spinner("Applying AI Super-Resolution & Clarity Enhancement..."):
                                upscaled_bytes = client.upscale_image(msg["image_bytes"], scale_factor=2.0)
                                if upscaled_bytes:
                                    msg["image_bytes"] = upscaled_bytes
                                    msg["is_upscaled"] = True
                                    msg["image_caption"] = f"{msg.get('image_caption', 'Artwork')} [AI Super-Resolution 2K/4K]"
                                    st.toast("✨ Upscaled to razor-sharp high-resolution!", icon="🎨")
                                    st.rerun()
                                else:
                                    st.error("Upscaling failed. Please try again.")
                    else:
                        st.markdown("<div style='padding-top:6px;'><span class='status-badge' style='background:rgba(63,185,80,0.15); border-color:rgba(63,185,80,0.4); color:#3fb950;'>✨ AI Super-Resolution Active</span></div>", unsafe_allow_html=True)

            # Source citations
            if msg.get("sources"):
                with st.expander("📚 Citations & Grounded Sources", expanded=False):
                    for src in msg["sources"]:
                        if src.get("url"):
                            st.markdown(f"- 🔗 [{src.get('title', 'Web Source')}]({src['url']})")
                        else:
                            st.markdown(f"- 📄 **{src.get('title', 'Document')}** (Chunk {src.get('page', 'N/A')}): {src.get('snippet', '')}")

            # Assistant Actions: Audio TTS & Download as Document
            if role == "assistant":
                col_act1, col_act2 = st.columns([1, 4])
                with col_act1:
                    if st.button("🔊 Listen", key=f"tts_{idx}", help="Play spoken audio response"):
                        with st.spinner("Generating speech audio..."):
                            audio_data = client.synthesize_speech(content[:500])
                            if audio_data:
                                st.audio(audio_data, format="audio/mpeg")
                with col_act2:
                    st.download_button(
                        label="📥 Export (.md)",
                        data=content,
                        file_name=f"response_{idx}.md",
                        mime="text/markdown",
                        key=f"dl_{idx}",
                        help="Export response as Markdown document"
                    )

# ----------------- ATTACHED DOCUMENTS CONTEXT BAR -----------------
if st.session_state.attached_docs:
    col_docs_view, col_docs_clear = st.columns([6, 1])
    with col_docs_view:
        badges_markup = "".join([
            f'<span class="attachment-active-bar">📄 <b>{doc["name"]}</b> '
            f'<span style="opacity:0.75; font-size:0.78rem;">({doc["chunks"]} chunks)</span></span>'
            for doc in st.session_state.attached_docs
        ])
        st.markdown(f"<div style='margin-bottom: 6px;'>{badges_markup}</div>", unsafe_allow_html=True)
    with col_docs_clear:
        if st.button("✕ Clear All", key="btn_clear_all_docs", help="Detach all active documents from context"):
            st.session_state.attached_docs = []
            st.session_state.processed_upload_keys = set()
            st.rerun()

# ----------------- CHATBAR CONTROL ROW (PLUS ICON & MODEL SWITCHING) -----------------
col_attach, col_model = st.columns([1, 6])

with col_attach:
    # Plus (+) Popover containing Multi-Document Ingestion, Camera/Image, and Art Generator
    with st.popover("➕", help="Upload multiple documents (RAG), capture photo, or generate art"):
        st.markdown("### 📎 Attachments & Tools")
        tab_doc, tab_cam, tab_art = st.tabs(["📁 Documents", "📸 Camera / Photo", "🎨 AI Art"])

        # Tab 1: Multi-Document RAG Ingestion
        with tab_doc:
            st.markdown("**📁 Multiple Document Ingestion (PDF, DOCX, TXT, MD)**")
            st.caption("Upload multiple files simultaneously. All documents will be chunked, embedded, and indexed into the RAG vector store.")
            uploaded_files = st.file_uploader(
                "Select one or multiple files to index into RAG:",
                type=["pdf", "docx", "txt", "md"],
                accept_multiple_files=True,
                key="plus_multi_doc_uploader"
            )
            if uploaded_files:
                new_files = [
                    f for f in uploaded_files 
                    if f"{f.name}_{f.size}" not in st.session_state.processed_upload_keys
                ]
                if new_files:
                    with st.spinner(f"⚡ Ingesting {len(new_files)} document(s) into vector database..."):
                        success_count = 0
                        for f in new_files:
                            file_key = f"{f.name}_{f.size}"
                            file_bytes = f.read()
                            upload_res = client.upload_document(file_bytes, f.name, f.type)
                            if "error" in upload_res:
                                st.error(f"Failed to ingest '{f.name}': {upload_res['error']}")
                            else:
                                st.session_state.processed_upload_keys.add(file_key)
                                st.session_state.attached_docs.append({
                                    "id": upload_res.get("id"),
                                    "name": f.name,
                                    "chunks": upload_res.get("chunk_count", 0),
                                    "key": file_key
                                })
                                success_count += 1
                        if success_count > 0:
                            st.toast(f"✅ Indexed {success_count} document(s) into RAG!", icon="📚")
                            st.rerun()

            # Manage currently indexed documents
            if st.session_state.attached_docs:
                st.markdown("---")
                st.markdown("**Currently Active Documents:**")
                for doc_idx, doc in enumerate(st.session_state.attached_docs):
                    cd1, cd2 = st.columns([5, 1])
                    with cd1:
                        st.markdown(f"📄 **{doc['name']}** ({doc['chunks']} chunks)")
                    with cd2:
                        if st.button("✕", key=f"del_doc_{doc_idx}", help=f"Remove {doc['name']}"):
                            st.session_state.processed_upload_keys.discard(doc.get("key"))
                            st.session_state.attached_docs.pop(doc_idx)
                            st.rerun()
                if st.button("🗑️ Detach All Documents", key="btn_detach_all_popover"):
                    st.session_state.attached_docs = []
                    st.session_state.processed_upload_keys = set()
                    st.rerun()

        # Tab 2: Camera & Vision Image
        with tab_cam:
            st.markdown("**Camera Snapshot or Photo Upload**")
            cam_pic = st.camera_input("Take a live photo:", key="camera_snap")
            file_img = st.file_uploader("Or choose image file:", type=["jpg", "jpeg", "png", "webp"], key="plus_img_uploader")
            chosen_img = cam_pic or file_img

            if chosen_img:
                st.image(chosen_img, caption="Selected Visual Subject", width=260)
                v_prompt = st.text_input("Vision Question:", "Describe and analyze this image in detail.", key="plus_vision_prompt")
                if st.button("🔍 Analyze Image", key="btn_run_vision"):
                    with st.spinner("Inspecting visual details with AI Vision..."):
                        v_res = client.analyze_image(
                            image_bytes=chosen_img.read(),
                            filename=getattr(chosen_img, "name", "camera.png"),
                            content_type="image/png",
                            prompt=v_prompt,
                            model="auto"
                        )
                        if "error" in v_res:
                            st.error(v_res["error"])
                        else:
                            st.session_state.messages.append({"role": "user", "content": f"[Analyzed Image]: {v_prompt}"})
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": v_res["response"],
                                "model_used": v_res.get("model_used", "vision"),
                                "switched_from": v_res.get("switched_from")
                            })
                            st.rerun()

        # Tab 3: AI Art Generation
        with tab_art:
            st.markdown("**High-Resolution AI Art Generation**")
            st.caption("Synthesize crystal-clear, sharp, artifact-free visual artwork with AI prompt optimization and super-resolution.")
            art_prompt = st.text_input("Visual Description:", "A serene Japanese zen garden in autumn with cherry blossom petals, 8k", key="plus_art_prompt")

            col_art1, col_art2 = st.columns(2)
            with col_art1:
                art_style = st.selectbox(
                    "Style / Engine:",
                    options=["flux-realism", "flux", "flux-anime", "flux-3d", "turbo"],
                    format_func=lambda x: {
                        "flux-realism": "📸 Photorealistic & Crystal-Clear",
                        "flux": "⚡ SOTA High Fidelity (FLUX.1)",
                        "flux-anime": "🎌 Vibrant Anime / Manga",
                        "flux-3d": "🎮 3D CGI & Octane Render",
                        "turbo": "🚀 Ultra-Fast SDXL Turbo"
                    }.get(x, x),
                    key="plus_art_style"
                )
            with col_art2:
                art_res = st.selectbox(
                    "Target Resolution:",
                    options=["1024x1024", "1536x1536", "2048x2048", "1920x1080", "1080x1920"],
                    format_func=lambda x: {
                        "1024x1024": "1024×1024 (Standard 1K)",
                        "1536x1536": "1536×1536 (High-Res 2K)",
                        "2048x2048": "2048×2048 (Ultra-Res 4K)",
                        "1920x1080": "1920×1080 (16:9 Landscape)",
                        "1080x1920": "1080×1920 (9:16 Portrait)"
                    }.get(x, x),
                    key="plus_art_res"
                )

            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                opt_prompt = st.checkbox("✨ AI Prompt Clarity Optimizer", value=True, help="Automatically enriches lighting, composition, and anti-artifact safeguards.")
            with col_opt2:
                opt_upscale = st.checkbox("🔍 AI Super-Resolution (2x)", value=True, help="Upscales and sharpens fine textures using sub-pixel reconstruction without quality loss.")

            if st.button("🎨 Synthesize High-Res Artwork", key="btn_plus_art"):
                with st.spinner("Synthesizing crystal-clear artwork with AI Super-Resolution..."):
                    res_w, res_h = map(int, art_res.split("x"))
                    art_bytes = client.generate_image(
                        prompt=art_prompt,
                        width=res_w,
                        height=res_h,
                        model=art_style,
                        upscale=opt_upscale,
                        enhance_prompt=opt_prompt
                    )
                    if art_bytes:
                        st.session_state.messages.append({"role": "user", "content": f"/image {art_prompt}"})
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"🎨 Visual artwork generated for: **{art_prompt}** *({art_res}, {art_style})*",
                            "image_bytes": art_bytes,
                            "image_caption": art_prompt,
                            "model_used": f"{art_style}+ai-upscale" if opt_upscale else art_style,
                            "is_upscaled": opt_upscale
                        })
                        st.rerun()
                    else:
                        st.warning("⚠️ Image provider reached its 45-second cooldown or is busy. Please wait a few moments and try again.")

with col_model:
    # Model switching right on the chatbar row
    selected_model_id = st.selectbox(
        "Model Switching",
        options=model_options,
        format_func=lambda x: model_labels.get(x, x),
        label_visibility="collapsed",
        help="Switch AI reasoning model with automatic graceful failover."
    )

# ----------------- CHAT INPUT BAR -----------------
input_placeholder = "Ask anything, write code, search the web, ask about attached documents..."
user_prompt = st.chat_input(input_placeholder)

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # ================= AUTONOMOUS INTENT ROUTING ENGINE =================
    # Analyze user prompt on its own and decide which capability to execute

    # 1. Image Generation Intent Detection (Autonomous Decision)
    def _is_image_request(prompt_text: str) -> bool:
        p = prompt_text.strip().lower()
        if any(p.startswith(cmd) for cmd in ['/image', '/img', '/imagine', '/draw', '/paint', '/art']):
            return True
        if re.search(r'^(?:(?:generate|create|make|draw|paint|sketch|render|show\s+me)\s+(?:an?\s+|the\s+|me\s+)?(?:image|picture|photo|drawing|wallpaper|portrait|illustration|artwork)?\s*(?:of\s+)?|(?:an?\s+)?(?:image|picture|photo|drawing|wallpaper|portrait|illustration|sketch|artwork)\s+of\b|^draw\s+|^paint\s+|^sketch\s+)', p):
            return True
        if re.search(r'\b(?:image|picture|photo|wallpaper|drawing|illustration|portrait|artwork)\b$', p):
            return True
        if re.search(r'\b(?:ki\s+photo|ki\s+image|ki\s+picture|photo\s+banao|image\s+banao|picture\s+banao|draw\s+karo)\b', p):
            return True
        if any(w in p for w in ['generate image', 'create image', 'draw me', 'make image', 'paint me', 'photo of', 'image of', 'picture of']):
            return True
        return False

    is_image_intent = _is_image_request(user_prompt)

    if is_image_intent:
        clean_img_prompt = user_prompt.strip()
        clean_img_prompt = re.sub(r'^/(?:image|img|imagine|draw|paint|art)\s*', '', clean_img_prompt, flags=re.I)
        clean_img_prompt = re.sub(r'^(?:(?:generate|create|make|draw|paint|sketch|render|show\s+me)\s+(?:an?\s+|the\s+|me\s+)?(?:image|picture|photo|drawing|wallpaper|portrait|illustration|artwork)?\s*(?:of\s+)?|(?:an?\s+)?(?:image|picture|photo|drawing|wallpaper|portrait|illustration|sketch|artwork)\s+of\s+|^draw\s+|^paint\s+|^sketch\s+)', '', clean_img_prompt, flags=re.I)
        clean_img_prompt = re.sub(r'\s*(?:ki\s+photo|ki\s+image|ki\s+picture|image|picture|photo|wallpaper|drawing|artwork)$', '', clean_img_prompt, flags=re.I)
        clean_img_prompt = clean_img_prompt.strip(' :,-')
        if len(clean_img_prompt) < 2:
            clean_img_prompt = user_prompt.strip()

        with st.chat_message("assistant"):
            with st.spinner(f"🎨 Synthesizing high-res visual artwork with AI super-resolution for '{clean_img_prompt}'..."):
                low_p = clean_img_prompt.lower()
                if any(w in low_p for w in ["doremon", "doraemon", "anime", "manga", "cartoon", "naruto", "goku"]):
                    selected_engine = "flux-anime"
                elif any(w in low_p for w in ["photo", "real", "human", "face", "portrait"]):
                    selected_engine = "flux-realism"
                elif any(w in low_p for w in ["3d", "cgi", "render", "game", "pixar"]):
                    selected_engine = "flux-3d"
                else:
                    selected_engine = "flux"

                img_data = client.generate_image(
                    prompt=clean_img_prompt,
                    width=1024,
                    height=1024,
                    model=selected_engine,
                    upscale=True,
                    scale_factor=2.0,
                    enhance_prompt=True
                )
                if img_data:
                    st.image(img_data, caption=f"🎨 AI Generated: {clean_img_prompt} [AI Super-Resolution 2K/4K]", use_container_width=True)
                    st.download_button(
                        label="📥 Download High-Res Artwork (.jpg)",
                        data=img_data,
                        file_name="ai_generated_image.jpg",
                        mime="image/jpeg",
                        key=f"dl_chat_img_{len(st.session_state.messages)}"
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"🎨 Visual artwork generated for: **{clean_img_prompt}** *(AI Super-Resolution & Clarity Enhanced)*",
                        "image_bytes": img_data,
                        "image_caption": clean_img_prompt,
                        "model_used": f"{selected_engine}+ai-upscale",
                        "is_upscaled": True
                    })
                else:
                    st.warning("⚠️ Image provider reached its 45-second cooldown or is busy. Please wait a few moments and try again.")
        st.rerun()

    # 2. Emotional Distress / Crisis Support Intent Detection
    distress_regex = r"\b(suicid|kill myself|end my life|depressed|depression|hopeless|overwhelmed|panic attack|burnout|anxiety|grief|lonely|heartbroken|crying|stressed|so tired of everything|mental health)\b"
    is_emotional = bool(re.search(distress_regex, user_prompt.lower()))

    # 3. Document RAG Intent (Active multiple documents attached)
    is_rag = bool(st.session_state.attached_docs)
    target_doc_ids = [d["id"] for d in st.session_state.attached_docs] if is_rag else None

    # 4. Deep Web Search & Real-Time Research Grounding Intent Detection
    search_keywords = [
        "search", "web search", "google", "latest", "recent", "news", 
        "current events", "weather", "today", "stock price", "price", 
        "score", "release date", "trending", "who is", "who won", "breaking"
    ]
    is_search = any(k in user_prompt.lower() for k in search_keywords) or bool(
        re.search(r"\b(search\b|search the web|latest|recent|news|current events|weather|stock|price|who is|trending|today's|breaking news)\b", user_prompt.lower())
    )

    # 5. Technical Code Synthesis Intent Detection
    code_keywords = [
        "write code", "code for", "python", "javascript", "typescript", 
        "react", "html", "css", "sql", "algorithm", "function", "debug", 
        "refactor", "unit test", "api endpoint", "regex", "docker", "fastapi"
    ]
    is_code = any(k in user_prompt.lower() for k in code_keywords) or bool(
        re.search(r"\b(write code|code for|script in|python|javascript|typescript|react component|sql query|algorithm|debug|refactor|unit test|regex pattern|fastapi|dockerfile)\b", user_prompt.lower())
    )

    system_instruction_override = None
    if is_code and not is_emotional:
        system_instruction_override = (
            "You are an expert principal software engineer and systems architect. "
            "When answering, provide clean, idiomatic, modular, robust, and well-tested code "
            "with type annotations, comments, docstrings, and robust error handling. "
            "Provide clear explanations and usage examples."
        )

    # Determine descriptive spinner message reflecting autonomous choice
    if is_search:
        spinner_msg = "🌐 Autonomous Search: Grounding live web facts & citations..."
    elif is_rag:
        spinner_msg = f"📚 Autonomous RAG: Grounding answer across {len(st.session_state.attached_docs)} attached document(s)..."
    elif is_code:
        spinner_msg = "💻 Autonomous Engineering: Drafting high-precision code & architecture..."
    elif is_emotional:
        spinner_msg = "❤️ Autonomous Empathy: Providing supportive emotional reflection..."
    else:
        spinner_msg = "⚡ Thinking..."

    with st.chat_message("assistant"):
        with st.spinner(spinner_msg):
            result = client.send_chat(
                message=user_prompt,
                model=selected_model_id,
                session_id=st.session_state.session_id,
                system_prompt=system_instruction_override,
                use_rag=is_rag,
                document_ids=target_doc_ids,
                use_web_search=is_search,
                emotional_mode=is_emotional
            )

        if "error" in result:
            st.error(result["error"])
        else:
            response_text = result["response"]
            model_used = result.get("model_used", "unknown")
            switched_from = result.get("switched_from")
            switch_reason = result.get("switch_reason")
            sources = result.get("sources", [])
            st.session_state.session_id = result.get("session_id")

            if switched_from:
                st.markdown(
                    f'<div class="switch-alert">⚡ <b>Auto-switched:</b> From <code>{switched_from}</code> '
                    f'to <code>{model_used}</code> ({switch_reason or "Failover"}).</div>',
                    unsafe_allow_html=True
                )

            st.markdown(response_text)

            if sources:
                with st.expander("📚 Citations & Grounded Sources", expanded=False):
                    for src in sources:
                        if src.get("url"):
                            st.markdown(f"- 🔗 [{src.get('title', 'Web Source')}]({src['url']})")
                        else:
                            st.markdown(f"- 📄 **{src.get('title', 'Document')}**: {src.get('snippet', '')}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "model_used": model_used,
                "switched_from": switched_from,
                "switch_reason": switch_reason,
                "sources": sources
            })
            st.rerun()
