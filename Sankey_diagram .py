import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ======================
# 1️⃣ 数据读取与字段清洗
# ======================
file_path = r"桑基图数据.xlsx" 
try:
    df = pd.read_excel(file_path)
except Exception as e:
    print(f"读取文件失败: {e}")

df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={
    "类型": "审批类型", 
    "是否有文件": "是否有文件", 
    "数字疗法": "DTx-Type", 
    "深度": "深度", 
    "组编号": "组编号"
})

# ======================
# 2️⃣ 分类规则（精确匹配 + Unclear处理）
# ======================
def get_color_and_label(row):
    has_file = str(row.get("是否有文件", "")).strip().upper()
    stype = str(row.get("审批类型", "")).strip().upper()
    dtype = str(row.get("DTx-Type", "")).strip().upper()

    # 精确判断 DTx 类型
    if dtype == "DTX":
        is_dtx = True
    elif dtype == "NON-DTX":
        is_dtx = False
    else:
        is_dtx = None  # Unclear-DTx

    # 1️⃣ De Novo 优先判断
    if "DE NOVO" in stype:
        if is_dtx is True:
            return "#C7E0B4", "De novo DTx"
        elif is_dtx is False:
            return "#B39DDB", "De novo non-DTx"

    # 2️⃣ 510(k) 逻辑
    if "510" in stype:
        # 无文件
        if has_file in ["FALSE"]:
            return "#71718F", "Missing 510(k) file"
        # 有文件
        if is_dtx is True:
            return "#9BD6E7", "510(k) DTx"
        elif is_dtx is False:
            return "#E1A7A0", "510(k) Non-DTx"

    # 兜底
    return "#CCCCCC", "Unknown"

df["颜色"], df["分类"] = zip(*df.apply(get_color_and_label, axis=1))

# ======================
# 3️⃣ 绘图参数
# ======================
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

max_devices = df.groupby("深度")["组编号"].nunique().max()
fig, ax = plt.subplots(figsize=(20, 14))

scale_factor = 1.0
y_step = 1.2
bar_height = 0.4
depths = sorted(df["深度"].unique())
pos_map = {}

# ======================
# 4️⃣ 绘制条形图与上下交错数据标签
# ======================
for i, d in enumerate(depths):
    layer_df = df[df["深度"] == d]
    cat_order = [
        "510(k) DTx",
        "510(k) Non-DTx",
        "De novo DTx",
        "De novo non-DTx",
        "Missing 510(k) file"
    ]
    counts = layer_df["分类"].value_counts().reindex(cat_order).fillna(0)
    total = len(layer_df)
    x_offset = 0
    y_pos = i * y_step
    pos_map[d] = {}

    for idx_label, (label, count) in enumerate(counts.items()):
        if count == 0: continue
        w = count * scale_factor
        color = layer_df[layer_df["分类"] == label]["颜色"].iloc[0]

        ax.barh(y_pos, w, height=bar_height, left=x_offset, color=color, 
                edgecolor='black', linewidth=0.6, alpha=0.8, zorder=4)
        
        pos_map[d][label] = {'x_start': x_offset, 'width': w, 'color': color}

        # 数据标签上下交错（已隐藏）
        # if count > 0:
        #     percentage = (count / total) * 100
        #     label_txt = f"{percentage:.1f}% ({int(count)}/{total})"
        #     if idx_label % 2 == 0:
        #         y_offset = -bar_height*0.7  # 下方
        #         va = 'top'
        #     else:
        #         y_offset = bar_height*0.7   # 上方
        #         va = 'bottom'
        #     ax.text(x_offset + w/2, y_pos + y_offset, label_txt,
        #             ha='center', va=va, fontsize=8, zorder=5,
        #             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

        x_offset += w

# ======================
# 5️⃣ 绘制血缘飘带
# ======================
def draw_sigmoid_flow(ax, x1, x2, y1, y2, w1, w2, color):
    steps = 40
    y = np.linspace(y1, y2, steps)
    t = (y - y1) / (y2 - y1)
    s = 1 / (1 + np.exp(-10 * (t - 0.5)))
    lx = x1 + (x2 - x1) * s
    rx = (x1 + w1) + ((x2 + w2) - (x1 + w1)) * s
    verts = list(zip(lx, y)) + list(zip(rx[::-1], y[::-1]))
    ax.add_patch(Polygon(verts, facecolor=color, alpha=0.3, edgecolor='none', zorder=2))

for i in range(len(depths) - 1):
    d_curr, d_next = depths[i], depths[i+1]
    curr_data = df[df["深度"] == d_curr][["组编号", "分类"]]
    next_data = df[df["深度"] == d_next][["组编号", "分类"]]
    links = pd.merge(curr_data, next_data, on="组编号", suffixes=('_src', '_dst'))
    
    flow_table = links.groupby(["分类_src", "分类_dst"]).size().reset_index(name='count')
    src_used = {label: 0 for label in pos_map[d_curr].keys()}
    dst_used = {label: 0 for label in pos_map[d_next].keys()}
    
    for _, row in flow_table.iterrows():
        s_lab, d_lab, f_count = row["分类_src"], row["分类_dst"], row["count"]
        if "510" not in s_lab or s_lab == "Missing 510(k) file":
            continue
        if s_lab in pos_map[d_curr] and d_lab in pos_map[d_next]:
            p_src = pos_map[d_curr][s_lab]
            p_dst = pos_map[d_next][d_lab]
            f_w = f_count * scale_factor
            
            x1 = p_src['x_start'] + src_used[s_lab]
            x2 = p_dst['x_start'] + dst_used[d_lab]
            
            draw_sigmoid_flow(ax, x1, x2, (i*y_step)+bar_height/2, ((i+1)*y_step)-bar_height/2, 
                              f_w, f_w, p_dst['color'])
            
            src_used[s_lab] += f_w
            dst_used[d_lab] += f_w

# ======================
# 6️⃣ 坐标轴与排版
# ======================
ax.set_yticks([i * y_step for i in range(len(depths))])
ax.set_yticklabels([str(d) for d in depths], fontsize=10)
ax.set_ylabel("Number of generations", fontsize=12)
ax.set_xlabel("Number of medical devices", fontsize=12)

ax.set_xlim(0, max_devices * 1.15) 
ax.set_ylim((len(depths)-1)*y_step + 1.2, -1.2) 

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例
legend_labels = [
    "De novo DTx",
    "De novo non-DTx",
    "510(k) DTx",
    "510(k) Non-DTx",
    "Missing 510(k) file"
]

legend_colors = [
    "#C7E0B4",
    "#B39DDB",
    "#9BD6E7",
    "#E1A7A0",
    "#71718F"
]

elements = [plt.Rectangle((0,0),1,1, facecolor=c, edgecolor='k', alpha=0.8, label=l) 
            for c, l in zip(legend_colors, legend_labels)]
ax.legend(handles=elements, loc='lower right', title="Medical device type", frameon=False)

plt.tight_layout()
plt.show()