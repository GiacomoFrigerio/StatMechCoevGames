import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(5, 5))
ax.axis('off')

# Create the payoff matrix table
cell_text = [['R', 'S'],
             ['T', 'P']]
row_labels = ['C', 'D']
col_labels = ['C', 'D']

table = ax.table(cellText=cell_text,
                 rowLabels=row_labels,
                 colLabels=col_labels,
                 loc='center',
                 cellLoc='center',
                 colWidths=[0.2]*2)

table.scale(1, 2)
table.auto_set_font_size(False)
table.set_fontsize(18)

# Correctly color S and T cells only
table[(1, 1)].set_facecolor('#add8e6')  # S cell in blue
table[(2, 0)].set_facecolor('#f08080')  # T cell in red

# Determine cell positions for accurate arrow placement
fig.canvas.draw()  # needed for accurate bbox placement
bbox_s = table[(0, 1)].get_window_extent(fig.canvas.get_renderer())
bbox_t = table[(1, 0)].get_window_extent(fig.canvas.get_renderer())
inv = ax.transData.inverted()
s_xy = inv.transform([bbox_s.x1 + 20, bbox_s.y0 + bbox_s.height / 2])
t_xy = inv.transform([bbox_t.x1 + 20, bbox_t.y0 + bbox_t.height / 2])

# Arrow: High resource → S increases
arrow_s = FancyArrowPatch((s_xy[0] - 0.12, s_xy[1]-0.08), (s_xy[0], s_xy[1]-0.08),
                          arrowstyle='-|>', color='blue', mutation_scale=20, lw=2)
ax.add_patch(arrow_s)
ax.text(s_xy[0] , s_xy[1] - 0.1, 'High resource:\nS increases', color='blue', fontsize=12, va='center')

# Arrow: Low resource → T increases
arrow_t = FancyArrowPatch((t_xy[0] - 0.35, t_xy[1]-0.08), (t_xy[0] - 0.2, t_xy[1]-0.08),
                          arrowstyle='-|>', color='red', mutation_scale=20, lw=2)
ax.add_patch(arrow_t)
ax.text(t_xy[0] - 0.62, t_xy[1]-0.08, 'Low resource:\nT increases', color='red', fontsize=12, va='center')

#ax.set_title('Payoff Matrix Modulated by Environmental Feedback')

plt.tight_layout()
plt.savefig("PayoffMatrix", dpi=60)
plt.show()
