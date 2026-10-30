import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
from skimage.morphology import skeletonize
import ezdxf
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io

# =========================================================
# 1. 페이지 기본 설정
# =========================================================
st.set_page_config(page_title="PDF/도면 어노테이터", layout="wide")
st.title("📐 웹 기반 PDF/도면 주석 및 CAD 선 추출기")
st.caption("스마트폰, 태블릿 브라우저에서도 사용 가능한 도면 주석 도구입니다.")

# =========================================================
# 2. 세션 상태 초기화
# =========================================================
for key, default in [
    ("rotation_angle", 0),
    ("page_drawings", {}),
    ("current_file", None),
    ("dxf_bytes", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =========================================================
# 3. 사이드바
# =========================================================
st.sidebar.header("📁 파일 및 도구 설정")
uploaded_file = st.sidebar.file_uploader(
    "PDF 또는 이미지 파일 업로드", type=["pdf", "png", "jpg", "jpeg"]
)

stroke_width = st.sidebar.slider("선 굵기 (px)", 1, 25, 3)
stroke_color = st.sidebar.color_picker("선 색상", "#FF0000")

# =========================================================
# 4. 헬퍼 함수
# =========================================================
def render_pdf_page(doc, page_num, dpi=120, rotation=0):
    """PDF 페이지를 이미지로 렌더링 + 회전 적용"""
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    if rotation:
        img = img.rotate(-rotation, expand=True, fillcolor=(255, 255, 255))
    return img

def render_image_bytes(file_bytes, rotation=0):
    """이미지 바이트를 PIL 이미지로 + 회전 적용"""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    if rotation:
        img = img.rotate(-rotation, expand=True, fillcolor=(255, 255, 255))
    return img

def fabric_path_to_points(obj, img_h, x_offset, scale_x=1.0, scale_y=1.0):
    """
    fabric.js path 객체 → DXF 좌표 리스트
    - 캔버스 절대 좌표를 그대로 사용하여 좌표 꼬임 방지
    - CAD 좌표계(좌하단 원점)로 Y축 반전 처리
    """
    path = obj.get("path", [])
    pts = []
    
    for cmd in path:
        if not cmd:
            continue
        op = cmd[0]
        
        # M, L 커맨드 처리
        if op in ("M", "L") and len(cmd) >= 3:
            px, py = cmd[1], cmd[2]
        # Q, C 곡선 커맨드 처리 (도착점만 사용)
        elif op in ("Q", "C") and len(cmd) >= 5:
            px, py = cmd[-2], cmd[-1]
        else:
            continue

        # 스케일 적용
        canvas_x = px * scale_x
        canvas_y = py * scale_y

        # CAD 좌표 변환 (Y축 반전 및 페이지 오프셋)
        gx = canvas_x + x_offset
        gy = img_h - canvas_y

        pts.append((float(gx), float(gy)))
    return pts

def extract_centerlines_from_canvas(canvas_img_data, target_w, target_h, min_area=10):
    """
    중심선 추출 로직
    - 캔버스 데이터를 원본 도면 크기(target_w, target_h)에 맞춰 출력하여 
      화면에 선이 뭉치거나 잘려 보이는 현상을 방지합니다.
    """
    if canvas_img_data is None:
        return None, None

    # 캔버스 데이터를 원본 도면 크기로 리사이즈
    canvas_img = cv2.resize(canvas_img_data.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    # 알파 채널(투명도) 추출
    if canvas_img.shape[2] == 4:
        alpha = canvas_img[:, :, 3]
    else:
        alpha = 255 - cv2.cvtColor(canvas_img[:, :, :3], cv2.COLOR_RGB2GRAY)

    mask = (alpha > 10).astype(np.uint8) * 255
    
    # 모폴로지 정제
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 최소 면적 필터
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_clean = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(mask_clean, [c], -1, 255, -1)

    # 스켈레톤 추출
    skel = skeletonize(mask_clean > 0)
    skel_u8 = (skel * 255).astype(np.uint8)

    # 원본 도면 크기의 흰색 배경 이미지 생성 후 선 오버레이
    skel_rgb = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    skel_rgb[skel_u8 > 0] = (0, 0, 0)

    return skel_u8, skel_rgb

# =========================================================
# 5. 메인 로직
# =========================================================
if uploaded_file is None:
    st.info("좌측 사이드바에서 PDF 또는 이미지 파일을 업로드해 주세요.")
    st.stop()

file_bytes = uploaded_file.read()
file_ext = uploaded_file.name.split(".")[-1].lower()

# 새 파일 업로드 시 세션 초기화
if st.session_state.current_file != uploaded_file.name:
    st.session_state.current_file = uploaded_file.name
    st.session_state.page_drawings = {}
    st.session_state.rotation_angle = 0
    st.session_state.dxf_bytes = None

# 회전 컨트롤
st.sidebar.markdown("---")
st.sidebar.write(f"🔄 현재 회전 각도: **{st.session_state.rotation_angle}°**")
rc1, rc2 = st.sidebar.columns(2)
if rc1.button("↺ -90°"):
    st.session_state.rotation_angle = (st.session_state.rotation_angle + 270) % 360
    st.session_state.dxf_bytes = None
    st.rerun()
if rc2.button("↻ +90°"):
    st.session_state.rotation_angle = (st.session_state.rotation_angle + 90) % 360
    st.session_state.dxf_bytes = None
    st.rerun()

# 문서 로드
doc = None
if file_ext == "pdf":
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    total_pages = len(doc)
    page_num = st.sidebar.number_input("페이지 선택", min_value=1, max_value=total_pages, value=1) - 1
    base_img = render_pdf_page(doc, page_num, rotation=st.session_state.rotation_angle)
else:
    total_pages = 1
    page_num = 0
    base_img = render_image_bytes(file_bytes, rotation=st.session_state.rotation_angle)

bg_w, bg_h = base_img.size
current_page_key = f"page_{page_num}"
canvas_key = f"canvas_p{page_num}_r{st.session_state.rotation_angle}"

# =========================================================
# 6. 레이아웃
# =========================================================
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"🖼️ 도면 작업 영역 ({page_num + 1}/{total_pages})")
    st.caption("💡 선을 그린 후 툴바의 [저장 ⇡]을 누르고 아래 저장 버튼을 눌러주세요.")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=base_img,
        update_streamlit=False,
        height=bg_h,
        width=bg_w,
        drawing_mode="freedraw",
        initial_drawing=st.session_state.page_drawings.get(current_page_key),
        key=canvas_key,
        return_image_data=True,
    )

    if st.button("💾 현재 페이지 작업 저장하기"):
        target = None
        widget_val = st.session_state.get(canvas_key)
        for cand in (widget_val, canvas_result):
            if cand is None:
                continue
            if hasattr(cand, "json_data") and cand.json_data:
                target = cand.json_data
                break
            if isinstance(cand, dict):
                if cand.get("json_data"):
                    target = cand["json_data"]
                    break
                if cand.get("objects"):
                    target = cand
                    break

        if target and target.get("objects"):
            st.session_state.page_drawings[current_page_key] = target
            st.session_state.dxf_bytes = None
            st.success(f"[{page_num + 1}페이지] {len(target['objects'])}개 주석 저장 완료!")
        else:
            st.warning("⚠️ 객체가 감지되지 않았습니다. 툴바 [저장 ⇡] 후 다시 시도하세요.")

with col2:
    st.subheader("⚙️ 주요 기능")

    # [1] 중심선 추출 (추후 고도화 예정)
    if st.button("🔍 CAD 중심선 추출"):
        skel_u8, skel_rgb = extract_centerlines_from_canvas(
            canvas_result.image_data, bg_w, bg_h
        )

        if skel_u8 is None:
            st.warning("캔버스 데이터가 없습니다.")
        else:
            skel_count = int((skel_u8 > 0).sum())
            if skel_count == 0:
                st.warning("그린 선이 감지되지 않았습니다. 빨간색으로 선을 그려주세요.")
            else:
                st.success(f"스켈레톤 픽셀 {skel_count}개 인식 완료")
                st.image(skel_rgb, caption="스켈레톤 결과", use_container_width=True)

    # =========================================================
    # [2] DXF 생성 (데이터가 있는 페이지만)
    # =========================================================
    if st.button("💾 DXF 생성 (전체 페이지)"):
        doc_dxf = ezdxf.new("R2010")
        msp = doc_dxf.modelspace()
        
        # 레이어 생성
        if "ANNOTATION" not in doc_dxf.layers:
            doc_dxf.layers.add("ANNOTATION", color=1) # 빨간색
        if "PAGE_BORDER" not in doc_dxf.layers:
            doc_dxf.layers.add("PAGE_BORDER", color=5) # 파란색
        if "PAGE_TEXT" not in doc_dxf.layers:
            doc_dxf.layers.add("PAGE_TEXT", color=7) # 흰색

        drawn_count = 0
        x_offset = 0.0

        for p_idx in range(total_pages):
            # 1. 주석 데이터가 있는 페이지인지 확인
            p_data = st.session_state.page_drawings.get(f"page_{p_idx}")
            
            # 주석 데이터가 없거나 빈 객체라면 이 페이지는 건너뜀 (도곽 안 그림)
            if not p_data or not p_data.get("objects"):
                continue

            if doc is not None:
                p_img = render_pdf_page(doc, p_idx, rotation=st.session_state.rotation_angle)
            else:
                p_img = render_image_bytes(file_bytes, rotation=st.session_state.rotation_angle)
            p_w, p_h = p_img.size

            # 2. 도곽 그리기
            border = [
                (x_offset, 0),
                (x_offset + p_w, 0),
                (x_offset + p_w, p_h),
                (x_offset, p_h),
            ]
            msp.add_lwpolyline(border, close=True, dxfattribs={"layer": "PAGE_BORDER"})

            # 3. 페이지 번호 텍스트 추가 (도곽 왼쪽 위)
            text_x = x_offset + (p_w * 0.02)
            text_y = p_h - (p_h * 0.05)
            
            msp.add_text(
                f"{p_idx + 1}page",
                dxfattribs={
                    "layer": "PAGE_TEXT",
                    "height": p_h * 0.03  # 텍스트 크기
                }
            ).set_placement((text_x, text_y))

            # 4. 주석(선) 그리기
            scale_x = p_w / bg_w if bg_w > 0 else 1.0
            scale_y = p_h / bg_h if bg_h > 0 else 1.0

            for obj in p_data["objects"]:
                pts = fabric_path_to_points(obj, p_h, x_offset, scale_x=scale_x, scale_y=scale_y)
                for i in range(len(pts) - 1):
                    msp.add_line(pts[i], pts[i + 1], dxfattribs={"layer": "ANNOTATION"})
                    drawn_count += 1

            # 다음 페이지 오프셋 (20% 여백)
            x_offset += p_w * 1.2

        # 파일 저장 및 다운로드 버퍼 생성
        buf = io.StringIO()
        doc_dxf.write(buf)
        st.session_state.dxf_bytes = buf.getvalue().encode("utf-8")
        st.success(f"DXF 생성 완료: 주석 {drawn_count}개 (데이터가 있는 페이지만 출력됨)")

    # =========================================================
    # [3] DXF 다운로드 버튼
    # =========================================================
    if st.session_state.dxf_bytes:
        st.download_button(
            label="📥 DXF 다운로드 (1:1 스케일)",
            data=st.session_state.dxf_bytes,
            file_name="drawing_annotation_1to1.dxf",
            mime="application/dxf",
        )