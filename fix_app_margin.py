import re

with open("app.py", "r") as f:
    code = f.read()

header_title = """'<div class="card-title">🔋 Battery Health</div>'"""
if header_title in code:
    code = code.replace(header_title, """'<div class="card-title" style="margin-bottom: 20px;">🔋 Battery Health</div>'""")

header_decision = """'<div class="card-title">⚡ Charging Decision (RL Policy)</div>'"""
if header_decision in code:
    code = code.replace(header_decision, """'<div class="card-title" style="margin-bottom: 20px;">⚡ Charging Decision (RL Policy)</div>'""")

header_inputs = """'<div class="card-title">🔧 Battery Sensor Inputs</div>'"""
if header_inputs in code:
    code = code.replace(header_inputs, """'<div class="card-title" style="margin-bottom: 20px;">🔧 Battery Sensor Inputs</div>'""")

header_reasoning = """'<div class="card-title">🤖 AI Reasoning</div>'"""
if header_reasoning in code:
    code = code.replace(header_reasoning, """'<div class="card-title" style="margin-bottom: 20px;">🤖 AI Reasoning</div>'""")

with open("app.py", "w") as f:
    f.write(code)
print("APP MARGINS UPDATED")
