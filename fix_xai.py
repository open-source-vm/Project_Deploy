import re

with open("utils/xai.py", "r") as f:
    code = f.read()

# 1. Update plot titles
code = code.replace('"Global Feature Importance"', '"SoH Prediction Feature Importance"')
code = code.replace('"RL Action Influence per Feature"', '"RL Charging Decision Feature Influence"')

# 2. Add tight_layout to all matplotlib plots where missing
# We can just blindly add plt.tight_layout() before return fig if it's not already there
def insert_tight_layout(text):
    lines = text.split("\n")
    for i in range(len(lines)):
        if "return fig" in lines[i] and "plt.tight_layout()" not in lines[i-1]:
            # Check if tight_layout is nearby, if not inject it
            chunk = "\n".join(lines[max(0, i-5):i])
            if "tight_layout" not in chunk:
                indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
                lines.insert(i, indent + "plt.tight_layout()")
    return "\n".join(lines)

code = insert_tight_layout(code)

with open("utils/xai.py", "w") as f:
    f.write(code)

print("XAI UPDATED")
