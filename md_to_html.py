import markdown
import sys
import os

with open("Elevvo_NLP_Internship_Portfolio.md", "r", encoding="utf-8") as f:
    text = f.read()

html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])

css = """
<style>
body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 2em; }
h1, h2, h3 { border-bottom: 1px solid #eaecef; padding-bottom: .3em; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
th, td { border: 1px solid #dfe2e5; padding: 6px 13px; }
th { background-color: #f6f8fa; }
code { background-color: #f6f8fa; padding: .2em .4em; border-radius: 3px; font-family: monospace; }
pre { background-color: #f6f8fa; padding: 16px; border-radius: 3px; overflow: auto; margin-bottom: 1em; }
img { max-width: 100%; }
</style>
"""

html_content = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}</head><body>{html_body}</body></html>"

with open("Elevvo_NLP_Internship_Portfolio.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("HTML generated successfully.")
